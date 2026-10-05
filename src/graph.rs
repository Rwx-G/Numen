//! Causal graph: the reasoning core. Loads the persisted nodes and edges into
//! an in-memory `petgraph` DAG and answers the three Pearl rungs structurally:
//! association (`explain`, `query`, `domain_confidence`), intervention
//! (`intervene`/`do(X)`), and counterfactual (`counterfactual` but-for). Also
//! hosts adversarial screening (`screen`) and Hebbian reinforcement.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Result;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rusqlite::Connection;
use serde::Serialize;

/// Strength floor above which an existing edge is trusted enough to override a
/// contradicting freshly-extracted relation.
pub const CONTRADICTION_THRESHOLD: f64 = 0.7;

/// Strength ceiling for `crawl`-sourced (corpus-ingested) edges, kept below the
/// contradiction threshold on purpose: an untrusted ingested relation must never
/// reach the trusted band where it would screen out a corrected relation, even
/// after repeated Hebbian reinforcement.
pub const CRAWL_TRUST_CEILING: f64 = 0.69;

/// How much faster a `crawl`-sourced edge decays than a reasoned one. Corpus
/// ingestion is cheap and noisy, so an ingested relation that is never reinforced
/// by actual reasoning should fade quickly rather than linger at full life. The
/// twin of `CRAWL_TRUST_CEILING` on the time axis: crawl edges are both capped
/// lower and forgotten sooner.
pub const CRAWL_DECAY_FACTOR: f64 = 3.0;

/// Minimum influence improvement for the max-product relaxation in `intervene`
/// to revisit a node, so near-equal cycles cannot loop forever.
const INFLUENCE_EPSILON: f64 = 1e-12;

/// Prompt context line returned when no node matches the query. French on
/// purpose: it is injected verbatim into the (French) delegation prompt.
const CONTEXT_FALLBACK: &str = "(graphe sous-couvert, aucun contexte pertinent)";

/// A causal relation between two concepts: the core domain record that flows
/// from extraction (and future relation sources) into the graph and the store.
/// Labels are canonicalized (short, lowercase) before they reach here.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Relation {
    pub cause: String,
    pub effect: String,
    pub kind: String,
    pub strength: f64,
}

/// A concept node: a labelled entity the graph can reason about.
pub struct Node {
    pub label: String,
    pub confidence: f64,
    /// Identity Subgraph member (self, values, operator): excluded from
    /// decay/prune, never erodes.
    pub pinned: bool,
}

/// A directed causal relation between two nodes. `strength` and `kind` are read
/// by path explanation; `do`-calculus and counterfactuals consume the rest in
/// Phase 2.
pub struct Edge {
    pub strength: f64,
    pub kind: String,
    #[allow(dead_code)]
    // provenance is aggregated from SQL in /stats, not from the in-memory edge yet
    pub source: String,
    /// Identity Subgraph member: excluded from decay/prune, authoritative.
    pub pinned: bool,
}

/// One hop of a causal explanation: a directed edge with its relation type.
#[derive(Serialize)]
pub struct PathStep {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub strength: f64,
}

/// A downstream node affected by an intervention, with the strength of the
/// strongest causal path from the intervened node and its hop distance.
#[derive(Serialize)]
pub struct Effect {
    pub concept: String,
    pub influence: f64,
    pub hops: usize,
}

/// A concept's immediate causal context: what Numen thinks directly causes it
/// (`causes`, the incoming edges) and what it directly causes (`effects`, the
/// outgoing edges), with its confidence and connectedness. The local
/// neighbourhood the agent reasons from when it attends to the concept.
#[derive(Serialize)]
pub struct Neighbourhood {
    pub concept: String,
    pub confidence: f64,
    pub degree: usize,
    pub causes: Vec<PathStep>,
    pub effects: Vec<PathStep>,
}

/// The result of `do(X)`: the parents detached by the intervention (their
/// edges into X are cut) and the descendants it propagates to.
#[derive(Serialize)]
pub struct Intervention {
    pub target: String,
    pub severed_parents: Vec<String>,
    pub affected: Vec<Effect>,
}

/// A causal query result: the directed causal path (if any) and the confounders
/// (common ancestors) that open back-door correlation, with a verdict that
/// separates the two.
#[derive(Serialize)]
pub struct CausalQuery {
    pub cause: String,
    pub effect: String,
    pub causal_path: Vec<PathStep>,
    pub confounders: Vec<String>,
    pub verdict: String,
}

/// A freshly-extracted relation rejected because it contradicts a trusted edge.
#[derive(Serialize)]
pub struct Conflict {
    pub cause: String,
    pub effect: String,
    pub kind: String,
    pub reason: String,
    pub existing: String,
}

/// The structural ("but-for") counterfactual: whether removing the cause would
/// leave the effect without any root cause. `alternative_causes` are the root
/// causes that would still produce the effect without it (over-determination).
/// This is the qualitative rung-3 test; quantitative PN/PS needs a full SCM.
#[derive(Serialize)]
pub struct Counterfactual {
    pub cause: String,
    pub effect: String,
    pub is_cause: bool,
    pub necessary: bool,
    pub alternative_causes: Vec<String>,
    pub verdict: String,
}

/// A node's standing in the graph: how connected it is and how confident.
#[derive(Serialize)]
pub struct ConceptStat {
    pub label: String,
    pub degree: usize,
    pub confidence: f64,
}

/// A pinned concept of the Identity Subgraph.
#[derive(Serialize)]
pub struct IdentityConcept {
    pub label: String,
    pub confidence: f64,
}

/// The Identity Subgraph: the pinned self/values/operator nodes and the pinned
/// edges among them (values `part_of` the self, the operator `authority`
/// corrigibility edge). The stable, never-decaying core the Active Inference
/// planner will read as preferences.
#[derive(Serialize)]
pub struct Identity {
    pub concepts: Vec<IdentityConcept>,
    pub relations: Vec<PathStep>,
}

/// The Active Inference generative-model view of a concept: how confident the
/// model is about it and how connected it is. Consumed by the `aif` planner.
#[derive(Serialize)]
pub struct ConceptState {
    pub label: String,
    pub confidence: f64,
    pub degree: usize,
    pub pinned: bool,
}

/// In-memory causal graph, hydrated from SQLite at startup. `labels` is a
/// label-to-index map kept in sync with `graph` so concept lookups are O(1);
/// it is rebuilt by `load` on every structural change.
pub struct CausalGraph {
    graph: DiGraph<Node, Edge>,
    labels: HashMap<String, NodeIndex>,
}

impl CausalGraph {
    /// Load all nodes and edges from the persistence layer into a DAG.
    pub fn load(conn: &Connection) -> Result<Self> {
        let mut graph = DiGraph::new();
        let mut index = HashMap::new();
        let mut labels = HashMap::new();

        let mut node_stmt = conn.prepare("SELECT id, label, confidence, pinned FROM nodes")?;
        let nodes = node_stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
                r.get::<_, bool>(3)?,
            ))
        })?;
        for node in nodes {
            let (id, label, confidence, pinned) = node?;
            let idx = graph.add_node(Node {
                label: label.clone(),
                confidence,
                pinned,
            });
            index.insert(id, idx);
            labels.insert(label, idx);
        }

        // Only active edges form the reasoning graph; superseded edges stay in the
        // store as audit records and must not influence traversal or screening.
        let mut edge_stmt = conn.prepare(
            "SELECT src, dst, strength, type, source, pinned FROM edges WHERE status = 'active'",
        )?;
        let edges = edge_stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, f64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, bool>(5)?,
            ))
        })?;
        for edge in edges {
            let (src, dst, strength, kind, source, pinned) = edge?;
            if let (Some(&a), Some(&b)) = (index.get(&src), index.get(&dst)) {
                graph.add_edge(
                    a,
                    b,
                    Edge {
                        strength,
                        kind,
                        source,
                        pinned,
                    },
                );
            }
        }

        Ok(Self { graph, labels })
    }

    /// Number of nodes currently held in memory.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of directed edges currently held in memory.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// A Graphviz DOT rendering of the active reasoning graph, for human
    /// inspection (`curl .../graph.dot | dot -Tpng -o graph.png`). Pinned identity
    /// nodes are drawn as boxes; each edge carries its relation kind and strength.
    /// One-way: the export reflects the graph, nothing is read back from it.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write as _;
        let mut dot = String::from("digraph numen {\n  rankdir=LR;\n  node [shape=ellipse];\n");
        for node in self.graph.node_weights() {
            if node.pinned {
                let _ = writeln!(dot, "  {} [shape=box];", quote(&node.label));
            }
        }
        for idx in self.graph.edge_indices() {
            if let Some((src, dst)) = self.graph.edge_endpoints(idx) {
                let edge = &self.graph[idx];
                let _ = writeln!(
                    dot,
                    "  {} -> {} [label=\"{} {:.2}\"];",
                    quote(&self.graph[src].label),
                    quote(&self.graph[dst].label),
                    edge.kind,
                    edge.strength
                );
            }
        }
        dot.push_str("}\n");
        dot
    }

    /// A Markdown edge-list of the active reasoning graph, for human reading (e.g.
    /// pasted into a chat). One-way, same as `to_dot`. Labels are not escaped: they
    /// are canonicalized (short, lowercase) at extraction, so they carry no
    /// backticks or control characters that would corrupt the Markdown. Served as
    /// `text/markdown`, never rendered as HTML.
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write as _;
        let mut md = format!(
            "# Numen causal graph\n\n{} concepts, {} relations\n\n",
            self.node_count(),
            self.edge_count()
        );
        for idx in self.graph.edge_indices() {
            if let Some((src, dst)) = self.graph.edge_endpoints(idx) {
                let edge = &self.graph[idx];
                let _ = writeln!(
                    md,
                    "- `{}` --{} ({:.2})--> `{}`",
                    self.graph[src].label, edge.kind, edge.strength, self.graph[dst].label
                );
            }
        }
        md
    }

    /// A node's total degree: incoming plus outgoing causal edges.
    fn node_degree(&self, idx: NodeIndex) -> usize {
        self.graph.edges_directed(idx, Direction::Outgoing).count()
            + self.graph.edges_directed(idx, Direction::Incoming).count()
    }

    /// The most connected concepts, ranked by total degree then confidence.
    pub fn top_concepts(&self, limit: usize) -> Vec<ConceptStat> {
        let mut stats: Vec<ConceptStat> = self
            .graph
            .node_indices()
            .map(|idx| {
                let degree = self.node_degree(idx);
                let node = &self.graph[idx];
                ConceptStat {
                    label: node.label.clone(),
                    degree,
                    confidence: node.confidence,
                }
            })
            .collect();
        stats.sort_by(|a, b| {
            b.degree
                .cmp(&a.degree)
                .then_with(|| b.confidence.total_cmp(&a.confidence))
        });
        stats.truncate(limit);
        stats
    }

    /// Find the shortest directed causal path from `from` to `to`, as the
    /// ordered list of hops. Returns `None` if either concept is unknown or no
    /// directed path exists; `Some(empty)` when `from == to`. This is the first
    /// causal-engine primitive; do-calculus and counterfactuals build on it.
    pub fn explain(&self, from: &str, to: &str) -> Option<Vec<PathStep>> {
        let start = self.find(from)?;
        let goal = self.find(to)?;
        self.path_between(start, goal)
    }

    /// BFS shortest directed path between two known nodes. `None` if no directed
    /// path exists; `Some(empty)` when `start == goal`.
    fn path_between(&self, start: NodeIndex, goal: NodeIndex) -> Option<Vec<PathStep>> {
        if start == goal {
            return Some(Vec::new());
        }

        let mut predecessor: HashMap<NodeIndex, (NodeIndex, EdgeIndex)> = HashMap::new();
        let mut visited: HashSet<NodeIndex> = HashSet::from([start]);
        let mut queue: VecDeque<NodeIndex> = VecDeque::from([start]);

        while let Some(node) = queue.pop_front() {
            if node == goal {
                break;
            }
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                if visited.insert(edge.target()) {
                    predecessor.insert(edge.target(), (node, edge.id()));
                    queue.push_back(edge.target());
                }
            }
        }

        if !visited.contains(&goal) {
            return None;
        }

        let mut steps = Vec::new();
        let mut current = goal;
        while current != start {
            let (prev, edge_idx) = *predecessor.get(&current)?;
            let edge = &self.graph[edge_idx];
            steps.push(PathStep {
                from: self.graph[prev].label.clone(),
                relation: edge.kind.clone(),
                to: self.graph[current].label.clone(),
                strength: edge.strength,
            });
            current = prev;
        }
        steps.reverse();
        Some(steps)
    }

    /// The structural "but-for" counterfactual for `cause` on `effect`: had the
    /// cause not occurred, would the effect still have a root cause? `None` if
    /// either concept is unknown. The cause is *necessary* when removing it
    /// leaves the effect unreachable from every remaining root.
    pub fn counterfactual(&self, cause: &str, effect: &str) -> Option<Counterfactual> {
        let x = self.find(cause)?;
        let y = self.find(effect)?;

        let is_cause = x != y && self.path_between(x, y).is_some();

        let mut alternative_causes: Vec<String> = self
            .roots()
            .into_iter()
            .filter(|&root| root != x && self.reaches(root, y, x))
            .map(|root| self.graph[root].label.clone())
            .collect();
        alternative_causes.sort();

        let necessary = is_cause && alternative_causes.is_empty();
        let verdict = if !is_cause {
            "not a cause"
        } else if necessary {
            "necessary cause"
        } else {
            "contributing cause"
        }
        .to_string();

        Some(Counterfactual {
            cause: self.graph[x].label.clone(),
            effect: self.graph[y].label.clone(),
            is_cause,
            necessary,
            alternative_causes,
            verdict,
        })
    }

    /// Exogenous roots: nodes with no incoming causal edge.
    fn roots(&self) -> Vec<NodeIndex> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .edges_directed(idx, Direction::Incoming)
                    .next()
                    .is_none()
            })
            .collect()
    }

    /// Whether `goal` is reachable from `start` along directed edges, treating
    /// `excluded` as removed from the graph.
    fn reaches(&self, start: NodeIndex, goal: NodeIndex, excluded: NodeIndex) -> bool {
        if start == excluded {
            return false;
        }
        let mut seen: HashSet<NodeIndex> = HashSet::from([start]);
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if node == goal {
                return true;
            }
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                if edge.target() != excluded && seen.insert(edge.target()) {
                    stack.push(edge.target());
                }
            }
        }
        false
    }

    /// All strict ancestors of `node` (reachable by following edges backward).
    fn ancestors(&self, node: NodeIndex) -> HashSet<NodeIndex> {
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            for edge in self.graph.edges_directed(current, Direction::Incoming) {
                if seen.insert(edge.source()) {
                    stack.push(edge.source());
                }
            }
        }
        seen
    }

    /// Separate causation from confounded correlation between two concepts: the
    /// directed causal path (if any) and the common ancestors that open
    /// back-door paths (confounders). Returns `None` if either concept is
    /// unknown.
    pub fn query(&self, cause: &str, effect: &str) -> Option<CausalQuery> {
        let x = self.find(cause)?;
        let y = self.find(effect)?;

        let causal_path = self.path_between(x, y);
        let has_causal = causal_path.is_some();

        let ancestors_y = self.ancestors(y);
        let mut confounders: Vec<String> = self
            .ancestors(x)
            .intersection(&ancestors_y)
            .filter(|&&idx| idx != x && idx != y)
            .map(|&idx| self.graph[idx].label.clone())
            .collect();
        confounders.sort();

        let verdict = match (has_causal, !confounders.is_empty()) {
            (true, false) => "causal",
            (false, true) => "confounded",
            (true, true) => "mixed",
            (false, false) => "no link",
        }
        .to_string();

        Some(CausalQuery {
            cause: self.graph[x].label.clone(),
            effect: self.graph[y].label.clone(),
            causal_path: causal_path.unwrap_or_default(),
            confounders,
            verdict,
        })
    }

    /// Adversarial check for one candidate relation against trusted knowledge.
    /// Returns a `Conflict` when a high-confidence edge already asserts the
    /// reverse direction, or the same edge with opposite polarity
    /// (`causes`/`enables` vs `prevents`). `None` when the candidate is novel,
    /// reinforcing, or contradicts only edges weaker than its own `strength` (the
    /// stronger claim wins; the weaker contradicted edge is removed at `apply`).
    pub fn contradicts(
        &self,
        cause: &str,
        effect: &str,
        kind: &str,
        strength: f64,
        threshold: f64,
    ) -> Option<Conflict> {
        let a = self.find(cause)?;
        let b = self.find(effect)?;

        for edge in self.graph.edges_directed(b, Direction::Outgoing) {
            if edge.target() == a
                && edge.weight().strength >= threshold
                && edge.weight().strength >= strength
            {
                return Some(Conflict {
                    cause: cause.to_string(),
                    effect: effect.to_string(),
                    kind: kind.to_string(),
                    reason: "reverse direction".to_string(),
                    existing: format!(
                        "{effect} --{}--> {cause} ({:.2})",
                        edge.weight().kind,
                        edge.weight().strength
                    ),
                });
            }
        }

        for edge in self.graph.edges_directed(a, Direction::Outgoing) {
            if edge.target() == b
                && edge.weight().strength >= threshold
                && edge.weight().strength >= strength
                && opposite_polarity(&edge.weight().kind, kind)
            {
                return Some(Conflict {
                    cause: cause.to_string(),
                    effect: effect.to_string(),
                    kind: kind.to_string(),
                    reason: "opposite polarity".to_string(),
                    existing: format!(
                        "{cause} --{}--> {effect} ({:.2})",
                        edge.weight().kind,
                        edge.weight().strength
                    ),
                });
            }
        }

        None
    }

    /// Partition extracted relations into those safe to seed and those rejected
    /// for contradicting a trusted edge at least as strong. The candidate's
    /// strength is first capped by its `source` ceiling, so an untrusted `crawl`
    /// claim cannot out-strength (and thus override) a trusted edge.
    pub fn screen(
        &self,
        relations: Vec<Relation>,
        threshold: f64,
        source: &str,
    ) -> (Vec<Relation>, Vec<Conflict>) {
        let ceiling = source_ceiling(source);
        let mut accepted = Vec::new();
        let mut conflicts = Vec::new();
        for relation in relations {
            let strength = relation.strength.min(ceiling);
            match self.contradicts(
                &relation.cause,
                &relation.effect,
                &relation.kind,
                strength,
                threshold,
            ) {
                Some(conflict) => conflicts.push(conflict),
                None => accepted.push(relation),
            }
        }
        (accepted, conflicts)
    }

    /// Hebbian reinforcement in place: bump the strength of each named edge
    /// `(cause, kind, effect)` toward 1.0. Mirrors `db::reinforce` so the
    /// in-memory graph stays in sync without a full reload. Unknown edges are
    /// skipped.
    pub fn reinforce(&mut self, edges: &[(&str, &str, &str)], delta: f64) {
        for (cause, kind, effect) in edges {
            let (Some(src), Some(dst)) = (self.find(cause), self.find(effect)) else {
                continue;
            };
            let edge = self
                .graph
                .edges_directed(src, Direction::Outgoing)
                .find(|edge| edge.target() == dst && edge.weight().kind == *kind)
                .map(|edge| edge.id());
            if let Some(idx) = edge {
                if let Some(weight) = self.graph.edge_weight_mut(idx) {
                    weight.strength = (weight.strength + delta).min(source_ceiling(&weight.source));
                }
            }
        }
    }

    /// Apply accepted relations in place, mirroring `db::seed_relations` so the
    /// in-memory graph tracks the store without a full reload. The node and edge
    /// upsert formulas match the SQL `ON CONFLICT` clauses exactly; the
    /// `apply_mirrors_a_full_reload` test pins the two together. Relations must be
    /// applied in the same order they are seeded so repeated concepts accumulate
    /// identically.
    pub fn apply(&mut self, relations: &[Relation], source: &str) {
        for r in relations {
            let src = self.upsert_node(&r.cause);
            let dst = self.upsert_node(&r.effect);
            let strength = r.strength.min(source_ceiling(source));
            self.upsert_edge(src, dst, &r.kind, strength, source);
            self.supersede_weaker_contradictions(src, dst, &r.kind, strength);
        }
    }

    /// Remove edges that contradict the just-applied `(src, dst, kind)` edge and
    /// are weaker than its `strength`: a reverse-direction edge `dst -> src` (a
    /// cycle) or an opposite-polarity edge `src -> dst`. The stronger claim wins;
    /// pinned identity edges are never removed. This drops the loser from the
    /// in-memory active graph; the store-side mirror in `db::seed_relations` instead
    /// marks it `superseded` (kept as an audit record, filtered out of
    /// `CausalGraph::load`), so the two stay equivalent for reasoning. Pinned by
    /// `apply_mirrors_a_full_reload`.
    fn supersede_weaker_contradictions(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        kind: &str,
        strength: f64,
    ) {
        let mut stale = Vec::new();
        for edge in self.graph.edges_directed(dst, Direction::Outgoing) {
            if edge.target() == src && !edge.weight().pinned && edge.weight().strength < strength {
                stale.push(edge.id());
            }
        }
        for edge in self.graph.edges_directed(src, Direction::Outgoing) {
            if edge.target() == dst
                && !edge.weight().pinned
                && edge.weight().strength < strength
                && opposite_polarity(&edge.weight().kind, kind)
            {
                stale.push(edge.id());
            }
        }
        for id in stale {
            self.graph.remove_edge(id);
        }
    }

    /// Insert a concept at confidence 0.5, or bump an existing one by 0.1 capped
    /// at 1.0. Mirrors `db::upsert_node`.
    fn upsert_node(&mut self, label: &str) -> NodeIndex {
        if let Some(&idx) = self.labels.get(label) {
            let node = &mut self.graph[idx];
            node.confidence = (node.confidence + 0.1).min(1.0);
            idx
        } else {
            let idx = self.graph.add_node(Node {
                label: label.to_string(),
                confidence: 0.5,
                pinned: false,
            });
            self.labels.insert(label.to_string(), idx);
            idx
        }
    }

    /// Insert a directed edge, or raise an existing `(src, dst, kind)` edge to
    /// `max(new, current)` capped at 1.0. Mirrors `db::upsert_edge`.
    fn upsert_edge(
        &mut self,
        src: NodeIndex,
        dst: NodeIndex,
        kind: &str,
        strength: f64,
        source: &str,
    ) {
        let existing = self
            .graph
            .edges_directed(src, Direction::Outgoing)
            .find(|e| e.target() == dst && e.weight().kind == kind)
            .map(|e| e.id());
        match existing {
            Some(idx) => {
                if let Some(weight) = self.graph.edge_weight_mut(idx) {
                    weight.strength = strength.max(weight.strength).min(1.0);
                }
            }
            None => {
                self.graph.add_edge(
                    src,
                    dst,
                    Edge {
                        strength: strength.min(1.0),
                        kind: kind.to_string(),
                        source: source.to_string(),
                        pinned: false,
                    },
                );
            }
        }
    }

    /// Pearl's `do(X)`: detach the parents of `target` (their edges into it no
    /// longer apply, since X is now set exogenously) and propagate forward to
    /// every descendant, scoring each by the strongest causal path. Returns
    /// `None` if the concept is unknown. Non-descendants are, by construction,
    /// unaffected - the difference from mere observation.
    pub fn intervene(&self, target: &str) -> Option<Intervention> {
        let x = self.find(target)?;

        let severed_parents: Vec<String> = self
            .graph
            .edges_directed(x, Direction::Incoming)
            .map(|edge| self.graph[edge.source()].label.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        // Max-product propagation: an edge multiplies influence by its strength
        // (<= 1), so cycles only weaken a path and the relaxation terminates.
        let mut best: HashMap<NodeIndex, (f64, usize)> = HashMap::from([(x, (1.0, 0))]);
        let mut queue: VecDeque<NodeIndex> = VecDeque::from([x]);
        while let Some(node) = queue.pop_front() {
            let Some(&(influence, hops)) = best.get(&node) else {
                continue;
            };
            for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                let candidate = influence * edge.weight().strength;
                let improved = best
                    .get(&edge.target())
                    .is_none_or(|&(current, _)| candidate > current + INFLUENCE_EPSILON);
                if improved {
                    best.insert(edge.target(), (candidate, hops + 1));
                    queue.push_back(edge.target());
                }
            }
        }

        let mut affected: Vec<Effect> = best
            .into_iter()
            .filter(|&(idx, _)| idx != x)
            .map(|(idx, (influence, hops))| Effect {
                concept: self.graph[idx].label.clone(),
                influence,
                hops,
            })
            .collect();
        affected.sort_by(|a, b| {
            b.influence
                .total_cmp(&a.influence)
                .then_with(|| a.hops.cmp(&b.hops))
        });

        Some(Intervention {
            target: self.graph[x].label.clone(),
            severed_parents,
            affected,
        })
    }

    /// Resolve a canonical label to its node index in O(1) via the label map.
    fn find(&self, label: &str) -> Option<NodeIndex> {
        self.labels.get(label).copied()
    }

    /// Snapshot the Identity Subgraph: every pinned node and the pinned edges
    /// among them. The self, the values, the operator, and the corrigibility edge.
    pub fn identity(&self) -> Identity {
        let concepts = self
            .graph
            .node_weights()
            .filter(|n| n.pinned)
            .map(|n| IdentityConcept {
                label: n.label.clone(),
                confidence: n.confidence,
            })
            .collect();
        let relations = self
            .graph
            .edge_indices()
            .filter(|&e| self.graph[e].pinned)
            .filter_map(|e| {
                let (src, dst) = self.graph.edge_endpoints(e)?;
                Some(PathStep {
                    from: self.graph[src].label.clone(),
                    relation: self.graph[e].kind.clone(),
                    to: self.graph[dst].label.clone(),
                    strength: self.graph[e].strength,
                })
            })
            .collect();
        Identity {
            concepts,
            relations,
        }
    }

    /// Snapshot every concept's confidence, total (in + out) degree, and pinned
    /// flag: the generative-model state the Active Inference planner reads.
    pub fn concept_states(&self) -> Vec<ConceptState> {
        self.graph
            .node_indices()
            .map(|i| ConceptState {
                label: self.graph[i].label.clone(),
                confidence: self.graph[i].confidence,
                degree: self.node_degree(i),
                pinned: self.graph[i].pinned,
            })
            .collect()
    }

    /// In + out degree of the named concept - how grounded it is in the graph -
    /// or 0 when the concept is absent. Used to verify that an autonomous
    /// exploration actually connected its target rather than drifting.
    pub fn degree_of(&self, label: &str) -> usize {
        self.find(label).map_or(0, |idx| self.node_degree(idx))
    }

    /// Confidence the graph holds in the named concept, or 0 when it is absent.
    /// Paired with `degree_of` to verify an autonomous exploration both connected
    /// its target (structural) and raised certainty about it (epistemic).
    pub fn confidence_of(&self, label: &str) -> f64 {
        self.find(label)
            .map_or(0.0, |idx| self.graph[idx].confidence)
    }

    /// A concept's immediate causal context - its direct causes and effects - or
    /// `None` if Numen has never heard of it. This is what the agent "knows" about
    /// a target it attends to.
    pub fn neighbourhood(&self, label: &str) -> Option<Neighbourhood> {
        let idx = self.find(label)?;
        let causes = self
            .graph
            .edges_directed(idx, Direction::Incoming)
            .map(|edge| PathStep {
                from: self.graph[edge.source()].label.clone(),
                relation: edge.weight().kind.clone(),
                to: label.to_string(),
                strength: edge.weight().strength,
            })
            .collect();
        let effects = self
            .graph
            .edges_directed(idx, Direction::Outgoing)
            .map(|edge| PathStep {
                from: label.to_string(),
                relation: edge.weight().kind.clone(),
                to: self.graph[edge.target()].label.clone(),
                strength: edge.weight().strength,
            })
            .collect();
        Some(Neighbourhood {
            concept: label.to_string(),
            confidence: self.graph[idx].confidence,
            degree: self.node_degree(idx),
            causes,
            effects,
        })
    }

    /// Fraction of the query's significant tokens covered by an existing node
    /// label, in `[0.0, 1.0]`. An empty graph always returns `0.0`, which is
    /// what drives delegation while the graph is still sparse.
    pub fn domain_confidence(&self, query: &str) -> f64 {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return 0.0;
        }
        let labels: Vec<String> = self
            .graph
            .node_weights()
            .map(|n| n.label.to_lowercase())
            .collect();
        let matched = tokens
            .iter()
            .filter(|t| labels.iter().any(|l| token_overlaps_label(t, l)))
            .count();
        matched as f64 / tokens.len() as f64
    }

    /// Render the nodes relevant to `query` as prompt context lines for
    /// delegation: those whose label overlaps a query token, unioned with the
    /// `recalled` concepts surfaced by semantic similarity. Returns a placeholder
    /// note when nothing matches.
    pub fn relevant_context(&self, query: &str, recalled: &[String]) -> String {
        let tokens = tokenize(query);
        let lines: Vec<String> = self
            .graph
            .node_weights()
            .filter(|n| {
                let label = n.label.to_lowercase();
                tokens.iter().any(|t| token_overlaps_label(t, &label))
                    || recalled.iter().any(|r| r.eq_ignore_ascii_case(&n.label))
            })
            .map(|n| format!("- [node: {}, conf: {:.2}]", n.label, n.confidence))
            .collect();
        if lines.is_empty() {
            CONTEXT_FALLBACK.to_string()
        } else {
            lines.join("\n")
        }
    }
}

/// Whether two relation kinds disagree in sign: `prevents` is negative, every
/// other relation (`causes`, `enables`, `requires`, `part_of`) is positive.
fn opposite_polarity(a: &str, b: &str) -> bool {
    (a == "prevents") != (b == "prevents")
}

/// Lowercase alphanumeric tokens longer than two characters.
fn tokenize(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

/// Whether a query token and an already-lowercased node label overlap as
/// substrings in either direction. The shared matching rule for domain coverage
/// (`domain_confidence`) and context selection (`relevant_context`).
fn token_overlaps_label(token: &str, label_lowercase: &str) -> bool {
    label_lowercase.contains(token) || token.contains(label_lowercase)
}

/// Maximum strength an edge from `source` may reach. Untrusted `crawl` (corpus)
/// edges are capped below the contradiction threshold; everything else (operator
/// delegation) can reach 1.0. Shared by `db` so the two strength paths agree.
pub(crate) fn source_ceiling(source: &str) -> f64 {
    if source == "crawl" {
        CRAWL_TRUST_CEILING
    } else {
        1.0
    }
}

/// Quote a node label as a DOT string literal, escaping backslashes and quotes so
/// a label can never break out of the quoted identifier.
fn quote(label: &str) -> String {
    let escaped = label.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Per-provenance decay rate, the temporal twin of `source_ceiling`: the factor
/// the consolidation lambda is multiplied by for an edge of this `source`.
/// `crawl` (corpus) edges are ephemeral and fade `CRAWL_DECAY_FACTOR` times
/// faster; `identity` edges never decay (they are pinned out of consolidation
/// anyway, this encodes the intent so the policy reads in one place); everything
/// else - operator delegation, local reasoning - decays at the baseline rate.
pub(crate) fn decay_multiplier(source: &str) -> f64 {
    match source {
        "crawl" => CRAWL_DECAY_FACTOR,
        "identity" => 0.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph() -> CausalGraph {
        let conn = crate::db::open(":memory:").unwrap();
        CausalGraph::load(&conn).unwrap()
    }

    #[test]
    fn empty_graph_has_zero_confidence() {
        let g = empty_graph();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.domain_confidence("how does ownership work in rust"), 0.0);
    }

    #[test]
    fn blank_query_is_zero_confidence() {
        let g = empty_graph();
        assert_eq!(g.domain_confidence("   "), 0.0);
    }

    #[test]
    fn relevant_context_falls_back_when_empty() {
        let g = empty_graph();
        assert!(g.relevant_context("rust", &[]).contains("sous-couvert"));
    }

    #[test]
    fn relevant_context_includes_recalled_concepts() {
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations = vec![Relation {
            cause: "gluten".to_string(),
            effect: "elasticity".to_string(),
            kind: "causes".to_string(),
            strength: 0.8,
        }];
        crate::db::seed_relations(&mut conn, &relations, "claude").unwrap();
        let g = CausalGraph::load(&conn).unwrap();
        // The query token matches no label; semantic recall surfaces "gluten".
        let context = g.relevant_context("xyzzy", &["gluten".to_string()]);
        assert!(context.contains("gluten"));
    }

    #[test]
    fn top_concepts_ranks_by_degree() {
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations = vec![
            Relation {
                cause: "ownership".to_string(),
                effect: "lifetime".to_string(),
                kind: "causes".to_string(),
                strength: 0.9,
            },
            Relation {
                cause: "borrow".to_string(),
                effect: "lifetime".to_string(),
                kind: "requires".to_string(),
                strength: 0.8,
            },
        ];
        crate::db::seed_relations(&mut conn, &relations, "claude").unwrap();

        let g = CausalGraph::load(&conn).unwrap();
        assert_eq!(g.edge_count(), 2);
        let top = g.top_concepts(10);
        assert_eq!(top.first().unwrap().label, "lifetime");
        assert_eq!(top.first().unwrap().degree, 2);
    }

    type NodeSnap = (String, f64);
    type EdgeSnap = (String, String, String, f64);

    /// Sorted (label, confidence) and (cause, kind, effect, strength) views, so
    /// two graphs can be compared by content regardless of internal index order.
    fn snapshot(g: &CausalGraph) -> (Vec<NodeSnap>, Vec<EdgeSnap>) {
        let mut nodes: Vec<NodeSnap> = g
            .graph
            .node_weights()
            .map(|n| (n.label.clone(), n.confidence))
            .collect();
        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        let mut edges: Vec<EdgeSnap> = g
            .graph
            .edge_indices()
            .map(|idx| {
                let (s, t) = g.graph.edge_endpoints(idx).unwrap();
                let w = &g.graph[idx];
                (
                    g.graph[s].label.clone(),
                    w.kind.clone(),
                    g.graph[t].label.clone(),
                    w.strength,
                )
            })
            .collect();
        edges.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        (nodes, edges)
    }

    #[test]
    fn to_dot_renders_active_edges_and_boxes_pinned_nodes() {
        let mut conn = crate::db::open(":memory:").unwrap();
        crate::db::seed_identity(&mut conn).unwrap();
        crate::db::seed_relations(
            &mut conn,
            &[rel_s("rain", "wet ground", "causes", 0.8)],
            "claude",
        )
        .unwrap();
        let g = CausalGraph::load(&conn).unwrap();

        let dot = g.to_dot();
        assert!(dot.starts_with("digraph numen {"));
        assert!(dot.contains("\"rain\" -> \"wet ground\" [label=\"causes 0.80\"];"));
        assert!(dot.contains("\"numen\" [shape=box];")); // pinned identity node
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn to_markdown_lists_active_edges() {
        let g = graph_with(&[rel_s("rain", "wet ground", "causes", 0.8)]);
        let md = g.to_markdown();
        assert!(md.contains("# Numen causal graph"));
        assert!(md.contains("- `rain` --causes (0.80)--> `wet ground`"));
    }

    #[test]
    fn apply_mirrors_a_full_reload() {
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations = vec![
            Relation {
                cause: "rain".to_string(),
                effect: "wet ground".to_string(),
                kind: "causes".to_string(),
                strength: 0.8,
            },
            Relation {
                cause: "wet ground".to_string(),
                effect: "slip".to_string(),
                kind: "causes".to_string(),
                strength: 0.6,
            },
            // Repeat the first relation to exercise the node and edge conflict
            // (reinforce) branches in both seed and apply.
            Relation {
                cause: "rain".to_string(),
                effect: "wet ground".to_string(),
                kind: "causes".to_string(),
                strength: 0.9,
            },
        ];
        crate::db::seed_relations(&mut conn, &relations, "test").unwrap();
        let reloaded = CausalGraph::load(&conn).unwrap();

        let mut applied = CausalGraph {
            graph: DiGraph::new(),
            labels: HashMap::new(),
        };
        applied.apply(&relations, "test");

        let (reloaded_nodes, reloaded_edges) = snapshot(&reloaded);
        let (applied_nodes, applied_edges) = snapshot(&applied);
        assert_eq!(applied_nodes.len(), reloaded_nodes.len());
        assert_eq!(applied_edges.len(), reloaded_edges.len());
        for ((al, ac), (rl, rc)) in applied_nodes.iter().zip(&reloaded_nodes) {
            assert_eq!(al, rl);
            assert!((ac - rc).abs() < 1e-9, "{al}: {ac} vs {rc}");
        }
        for ((af, ak, at, astr), (rf, rk, rt, rstr)) in applied_edges.iter().zip(&reloaded_edges) {
            assert_eq!((af, ak, at), (rf, rk, rt));
            assert!((astr - rstr).abs() < 1e-9, "{af}->{at}: {astr} vs {rstr}");
        }
    }

    #[test]
    fn apply_mirrors_a_full_reload_through_a_contradiction() {
        // The store keeps the contradiction loser as a superseded audit row while
        // the in-memory graph drops it; the active-only load filter must make a
        // reload match the in-place apply exactly.
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations = vec![
            rel_s("a", "b", "causes", 0.6),
            rel_s("b", "a", "causes", 0.9), // stronger reverse claim supersedes a->b
        ];
        crate::db::seed_relations(&mut conn, &relations, "test").unwrap();
        let reloaded = CausalGraph::load(&conn).unwrap();

        let mut applied = CausalGraph {
            graph: DiGraph::new(),
            labels: HashMap::new(),
        };
        applied.apply(&relations, "test");

        assert_eq!(snapshot(&applied).1, snapshot(&reloaded).1);
        assert_eq!(snapshot(&reloaded).1.len(), 1); // only the active winner b->a
    }

    #[test]
    fn apply_mirrors_a_full_reload_through_a_supersede_then_revive_cycle() {
        // The load-bearing revive case: a->b loses to b->a, then a stronger a->b
        // revives and supersedes b->a in turn. The store revives the superseded row
        // to the fresh strength; the in-memory graph re-adds a brand-new edge. They
        // must still land identically, or reasoning (RAM) and the store diverge.
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations = vec![
            rel_s("a", "b", "causes", 0.6),
            rel_s("b", "a", "causes", 0.9),  // supersedes a->b
            rel_s("a", "b", "causes", 0.95), // revives a->b, supersedes b->a
        ];
        crate::db::seed_relations(&mut conn, &relations, "test").unwrap();
        let reloaded = CausalGraph::load(&conn).unwrap();

        let mut applied = CausalGraph {
            graph: DiGraph::new(),
            labels: HashMap::new(),
        };
        applied.apply(&relations, "test");

        assert_eq!(snapshot(&applied), snapshot(&reloaded));
        let edges = snapshot(&reloaded).1;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, "a"); // a->b is the active winner
        assert_eq!(edges[0].2, "b");
        assert!((edges[0].3 - 0.95).abs() < 1e-9);
    }

    #[test]
    fn crawl_edges_cannot_reach_the_trusted_band() {
        let rel = vec![Relation {
            cause: "rumor".to_string(),
            effect: "panic".to_string(),
            kind: "causes".to_string(),
            strength: 0.95,
        }];
        let mut crawl = CausalGraph {
            graph: DiGraph::new(),
            labels: HashMap::new(),
        };
        crawl.apply(&rel, "crawl");
        // Even hammering reinforcement cannot push it into the trusted band.
        crawl.reinforce(&[("rumor", "causes", "panic")], 0.5);
        let strength = snapshot(&crawl).1[0].3;
        assert!(
            strength <= CRAWL_TRUST_CEILING,
            "crawl edge reached {strength}"
        );
        assert!(strength < CONTRADICTION_THRESHOLD);

        // A delegation-sourced edge with the same inputs is not capped.
        let mut claude = CausalGraph {
            graph: DiGraph::new(),
            labels: HashMap::new(),
        };
        claude.apply(&rel, "claude");
        assert!(snapshot(&claude).1[0].3 > CRAWL_TRUST_CEILING);
    }

    #[test]
    fn decay_multiplier_fades_crawl_fastest_and_identity_never() {
        assert!(decay_multiplier("crawl") > decay_multiplier("claude"));
        assert!((decay_multiplier("claude") - 1.0).abs() < 1e-9);
        assert!((decay_multiplier("local") - 1.0).abs() < 1e-9);
        assert!((decay_multiplier("identity")).abs() < 1e-9);
    }

    #[test]
    fn identity_reports_only_pinned_values_and_corrigibility() {
        let mut conn = crate::db::open(":memory:").unwrap();
        crate::db::seed_identity(&mut conn).unwrap();
        // A learned (non-pinned) relation must not leak into the identity.
        let learned = vec![Relation {
            cause: "rain".to_string(),
            effect: "wet".to_string(),
            kind: "causes".to_string(),
            strength: 0.5,
        }];
        crate::db::seed_relations(&mut conn, &learned, "claude").unwrap();

        let id = CausalGraph::load(&conn).unwrap().identity();
        assert_eq!(id.concepts.len(), 8); // self + operator + 6 values
        assert!(id.concepts.iter().any(|c| c.label == "numen"));
        assert!(id.concepts.iter().any(|c| c.label == "curiosity"));
        assert!(!id.concepts.iter().any(|c| c.label == "rain"));
        assert_eq!(id.relations.len(), 7); // 6 part_of + 1 corrigibility
        assert!(id
            .relations
            .iter()
            .any(|r| r.from == "numen operator" && r.to == "numen" && r.relation == "authority"));
    }

    fn chain_graph() -> CausalGraph {
        let mut conn = crate::db::open(":memory:").unwrap();
        let step = |cause: &str, effect: &str| Relation {
            cause: cause.to_string(),
            effect: effect.to_string(),
            kind: "causes".to_string(),
            strength: 0.9,
        };
        let relations = vec![
            step("move", "ownership transfer"),
            step("ownership transfer", "use after move"),
        ];
        crate::db::seed_relations(&mut conn, &relations, "claude").unwrap();
        CausalGraph::load(&conn).unwrap()
    }

    #[test]
    fn explain_returns_directed_path() {
        let g = chain_graph();
        let path = g.explain("move", "use after move").unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].from, "move");
        assert_eq!(path[1].to, "use after move");
    }

    #[test]
    fn explain_respects_direction_and_unknowns() {
        let g = chain_graph();
        assert!(g.explain("use after move", "move").is_none());
        assert!(g.explain("move", "nonexistent").is_none());
        assert_eq!(g.explain("move", "move").unwrap().len(), 0);
    }

    #[test]
    fn intervene_severs_parents_and_propagates() {
        let g = chain_graph(); // move -> ownership transfer -> use after move
        let result = g.intervene("ownership transfer").unwrap();

        assert_eq!(result.target, "ownership transfer");
        assert_eq!(result.severed_parents, vec!["move".to_string()]);
        assert_eq!(result.affected.len(), 1);
        assert_eq!(result.affected[0].concept, "use after move");
        assert_eq!(result.affected[0].hops, 1);
        assert!((result.affected[0].influence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn intervene_on_unknown_is_none() {
        assert!(chain_graph().intervene("nonexistent").is_none());
    }

    fn graph_from(edges: &[(&str, &str)]) -> CausalGraph {
        let mut conn = crate::db::open(":memory:").unwrap();
        let relations: Vec<_> = edges
            .iter()
            .map(|(cause, effect)| Relation {
                cause: cause.to_string(),
                effect: effect.to_string(),
                kind: "causes".to_string(),
                strength: 0.9,
            })
            .collect();
        crate::db::seed_relations(&mut conn, &relations, "claude").unwrap();
        CausalGraph::load(&conn).unwrap()
    }

    #[test]
    fn query_reports_pure_causation() {
        let g = graph_from(&[("a", "b"), ("b", "c")]);
        let q = g.query("a", "c").unwrap();
        assert_eq!(q.verdict, "causal");
        assert_eq!(q.causal_path.len(), 2);
        assert!(q.confounders.is_empty());
    }

    #[test]
    fn query_reports_pure_confounding() {
        // z is a common cause of x and y; no directed x -> y path.
        let g = graph_from(&[("z", "x"), ("z", "y")]);
        let q = g.query("x", "y").unwrap();
        assert_eq!(q.verdict, "confounded");
        assert!(q.causal_path.is_empty());
        assert_eq!(q.confounders, vec!["z".to_string()]);
    }

    #[test]
    fn query_reports_mixed_when_both() {
        // z confounds x and y, and x also causes y directly.
        let g = graph_from(&[("z", "x"), ("z", "y"), ("x", "y")]);
        let q = g.query("x", "y").unwrap();
        assert_eq!(q.verdict, "mixed");
        assert_eq!(q.causal_path.len(), 1);
        assert_eq!(q.confounders, vec!["z".to_string()]);
    }

    #[test]
    fn counterfactual_flags_necessary_cause() {
        // a is the only causal route to c: but-for a, no c.
        let g = graph_from(&[("a", "b"), ("b", "c")]);
        let cf = g.counterfactual("a", "c").unwrap();
        assert!(cf.is_cause);
        assert!(cf.necessary);
        assert!(cf.alternative_causes.is_empty());
        assert_eq!(cf.verdict, "necessary cause");
    }

    #[test]
    fn counterfactual_flags_contributing_when_overdetermined() {
        // root z reaches y without going through x, so x is not necessary.
        let g = graph_from(&[("z", "x"), ("z", "y"), ("x", "y")]);
        let cf = g.counterfactual("x", "y").unwrap();
        assert!(cf.is_cause);
        assert!(!cf.necessary);
        assert_eq!(cf.alternative_causes, vec!["z".to_string()]);
        assert_eq!(cf.verdict, "contributing cause");
    }

    #[test]
    fn counterfactual_rejects_non_cause() {
        let g = graph_from(&[("a", "b"), ("b", "c")]);
        let cf = g.counterfactual("c", "a").unwrap();
        assert!(!cf.is_cause);
        assert_eq!(cf.verdict, "not a cause");
    }

    fn rel_s(cause: &str, effect: &str, kind: &str, strength: f64) -> Relation {
        Relation {
            cause: cause.to_string(),
            effect: effect.to_string(),
            kind: kind.to_string(),
            strength,
        }
    }

    fn graph_with(relations: &[Relation]) -> CausalGraph {
        let mut conn = crate::db::open(":memory:").unwrap();
        crate::db::seed_relations(&mut conn, relations, "claude").unwrap();
        CausalGraph::load(&conn).unwrap()
    }

    #[test]
    fn contradicts_flags_reverse_direction() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.9)]);
        assert_eq!(
            g.contradicts("b", "a", "causes", 0.8, 0.7).unwrap().reason,
            "reverse direction"
        );
        assert!(g.contradicts("a", "b", "causes", 0.8, 0.7).is_none()); // reinforcing
        assert!(g.contradicts("x", "y", "causes", 0.8, 0.7).is_none()); // novel
    }

    #[test]
    fn contradicts_flags_opposite_polarity() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.9)]);
        assert_eq!(
            g.contradicts("a", "b", "prevents", 0.8, 0.7)
                .unwrap()
                .reason,
            "opposite polarity"
        );
    }

    #[test]
    fn contradicts_ignores_weak_edges() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.5)]);
        assert!(g.contradicts("b", "a", "causes", 0.8, 0.7).is_none());
    }

    #[test]
    fn screen_partitions_relations() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.9)]);
        let candidates = vec![
            rel_s("b", "a", "causes", 0.8), // reverse -> conflict
            rel_s("x", "y", "causes", 0.8), // novel -> accepted
        ];
        let (accepted, conflicts) = g.screen(candidates, 0.7, "claude");
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].cause, "x");
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].cause, "b");
    }

    #[test]
    fn screen_lets_a_stronger_claim_override_a_weaker_trusted_edge() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.75)]);
        // A stronger reverse claim wins rather than being rejected as a conflict.
        let (accepted, conflicts) = g.screen(vec![rel_s("b", "a", "causes", 0.9)], 0.7, "claude");
        assert_eq!(accepted.len(), 1);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn screen_caps_crawl_strength_so_it_cannot_override_a_trusted_edge() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.75)]);
        // A crawl claim nominally at 0.95 is capped to 0.69, below the trusted edge.
        let (accepted, conflicts) = g.screen(vec![rel_s("b", "a", "causes", 0.95)], 0.7, "crawl");
        assert!(accepted.is_empty());
        assert_eq!(conflicts.len(), 1);
    }

    #[test]
    fn apply_supersedes_a_weaker_contradicted_edge() {
        let mut g = graph_with(&[rel_s("a", "b", "causes", 0.6)]);
        g.apply(&[rel_s("b", "a", "causes", 0.9)], "claude");
        // The stronger reverse edge wins; the weaker a->b is removed.
        assert!(g.explain("a", "b").is_none());
        assert!(g.explain("b", "a").is_some());
    }

    #[test]
    fn degree_of_counts_grounding_and_is_zero_for_absent_concepts() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.8)]);
        assert_eq!(g.degree_of("a"), 1);
        assert_eq!(g.degree_of("b"), 1);
        assert_eq!(g.degree_of("missing"), 0);
    }

    #[test]
    fn confidence_of_reads_node_certainty_and_is_zero_for_absent_concepts() {
        let g = graph_with(&[rel_s("a", "b", "causes", 0.8)]);
        assert!(g.confidence_of("a") > 0.0);
        assert_eq!(g.confidence_of("missing"), 0.0);
    }

    #[test]
    fn neighbourhood_separates_causes_from_effects() {
        let g = graph_with(&[
            rel_s("a", "b", "causes", 0.8),
            rel_s("b", "c", "causes", 0.6),
        ]);
        let n = g.neighbourhood("b").unwrap();
        assert_eq!(n.concept, "b");
        assert_eq!(n.degree, 2);
        // a -> b is a cause of b; b -> c is an effect of b.
        assert_eq!(n.causes.len(), 1);
        assert_eq!(n.causes[0].from, "a");
        assert_eq!(n.effects.len(), 1);
        assert_eq!(n.effects[0].to, "c");
        assert!(g.neighbourhood("unknown").is_none());
    }

    #[test]
    fn reinforce_bumps_in_memory_edge_and_caps() {
        let mut g = graph_with(&[rel_s("a", "b", "causes", 0.8)]);

        g.reinforce(&[("a", "causes", "b")], 0.1);
        assert!((g.explain("a", "b").unwrap()[0].strength - 0.9).abs() < 1e-9);

        g.reinforce(&[("a", "causes", "b")], 0.5); // caps at 1.0
        assert!((g.explain("a", "b").unwrap()[0].strength - 1.0).abs() < 1e-9);

        g.reinforce(&[("x", "causes", "y")], 0.1); // unknown edge: no-op, no panic
    }
}
