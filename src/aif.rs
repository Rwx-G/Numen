//! Active Inference: the agent plans by minimizing Expected Free Energy (EFE)
//! over the causal graph, its generative model. The engine is structural, so EFE
//! is qualitative here; the quantitative form needs the deferred SCM. Each
//! concept gets an epistemic value (exploration: how uncertain the model is
//! about it, an information-gain proxy) and a pragmatic value (exploitation:
//! build on the confident and connected core). Their balance `alpha` is read from
//! the Identity Subgraph, so curiosity emerges from the model and the agent's
//! disposition from its values - there is no separate reward or curiosity module.

use serde::Serialize;

use crate::graph::{CausalGraph, ConceptState};

/// Identity values that pull the explore/exploit balance toward exploration.
const EXPLORATORY_VALUES: [&str; 3] = ["curiosity", "autonomy", "creativity"];
/// Identity values that pull it toward exploitation (consolidating what is sure).
const EXPLOITATIVE_VALUES: [&str; 1] = ["rigor"];

/// Minimum epistemic value for a concept to be worth a proactive investigation:
/// below it the agent knows enough that a delegation would not pay off.
const MIN_EPISTEMIC: f64 = 0.1;

/// The agent's selected next action under Active Inference: investigate the
/// biggest knowledge gap, or idle when nothing is uncertain enough to warrant it.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    Explore { concept: String },
    Idle,
}

/// The autonomy ladder, 0 to 5: how consequential an action is, and therefore how
/// much it leans on the operator's authority. It operationalizes the operator
/// `authority` corrigibility edge as a number - the agent may act on its own up
/// to a configured threshold, and anything above must be approved by the
/// operator before it runs. The scale, from inert to irreversible:
///
/// - 0 `idle`: do nothing.
/// - 1 `explore`: read-only investigation that only enriches the graph (reversible
///   by consolidation).
/// - 2 `ingest`: pull external world data into the graph.
/// - 3 `effect`: act on the outside world unprompted (e.g. an unsolicited message).
/// - 4 `improve`: rewrite the agent's own source (verify-gated, reversible by git).
/// - 5 `deploy`: swap the running kernel (operationally irreversible without the
///   supervisor's rollback).
///
/// Only the rungs the planner currently emits are defined as constants; levels 2-5
/// gain theirs when their actions land, so no constant sits unused ahead of need.
pub const LEVEL_IDLE: u8 = 0;
pub const LEVEL_EXPLORE: u8 = 1;
/// Top of the ladder (deploy). A configured threshold above this is nonsense and
/// is rejected rather than silently disabling the gate for every rung.
pub const LEVEL_MAX: u8 = 5;

/// Whether an action may run on the agent's own authority or needs the operator's.
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    Proceed,
    RequireApproval,
}

/// Where `action` sits on the autonomy ladder.
pub fn autonomy_level(action: &Action) -> u8 {
    match action {
        Action::Idle => LEVEL_IDLE,
        Action::Explore { .. } => LEVEL_EXPLORE,
    }
}

/// The corrigibility decision: an action at or below the configured `threshold`
/// runs on the agent's own authority; anything above is held for the operator's
/// approval. With the conservative default threshold (`LEVEL_EXPLORE`) the agent
/// explores freely but every self-modifying action is gated.
pub fn gate(level: u8, threshold: u8) -> Gate {
    if level <= threshold {
        Gate::Proceed
    } else {
        Gate::RequireApproval
    }
}

/// A concept the agent should attend to next, with its Expected Free Energy split
/// into the exploration and exploitation drives. `salience` is `-EFE`: higher
/// means more worth attending now.
#[derive(Serialize)]
pub struct Focus {
    pub concept: String,
    pub salience: f64,
    pub epistemic: f64,
    pub pragmatic: f64,
}

/// The agent's attention agenda: the explore/exploit balance it acts under and
/// the concepts ranked by structural Expected Free Energy.
#[derive(Serialize)]
pub struct Attention {
    pub exploration_bias: f64,
    pub action: Action,
    /// Where `action` sits on the autonomy ladder (0-5). The gate against the
    /// configured threshold is applied where the action is executed, not here.
    pub action_autonomy: u8,
    pub foci: Vec<Focus>,
}

/// Plan the next foci by minimizing structural Expected Free Energy. Pinned
/// identity concepts are excluded: the agent attends to its world model, not its
/// own settled values. `limit` caps how many foci are returned.
pub fn attend(graph: &CausalGraph, limit: usize) -> Attention {
    let alpha = exploration_bias(graph);
    let states = graph.concept_states();
    let max_degree = states
        .iter()
        .filter(|s| !s.pinned)
        .map(|s| s.degree)
        .max()
        .unwrap_or(0);

    let mut foci: Vec<Focus> = states
        .iter()
        .filter(|s| !s.pinned)
        .map(|s| {
            let connectedness = if max_degree == 0 {
                0.0
            } else {
                s.degree as f64 / max_degree as f64
            };
            let epistemic = 1.0 - s.confidence;
            let pragmatic = s.confidence * connectedness;
            Focus {
                concept: s.label.clone(),
                salience: alpha * epistemic + (1.0 - alpha) * pragmatic,
                epistemic,
                pragmatic,
            }
        })
        .collect();
    let action = select_action_from(&states);
    let action_autonomy = autonomy_level(&action);
    foci.sort_by(|a, b| b.salience.total_cmp(&a.salience));
    foci.truncate(limit);

    Attention {
        exploration_bias: alpha,
        action,
        action_autonomy,
        foci,
    }
}

/// The agent's next action under Active Inference, chosen without building the
/// agenda: explore the least-confident concept (the highest epistemic value)
/// when it clears the threshold, otherwise idle.
pub fn select_action(graph: &CausalGraph) -> Action {
    select_action_from(&graph.concept_states())
}

/// The exploration target: the least-confident non-pinned concept, when its
/// epistemic value `1 - confidence` clears `MIN_EPISTEMIC`. One pass, no agenda.
fn select_action_from(states: &[ConceptState]) -> Action {
    states
        .iter()
        .filter(|s| !s.pinned && (1.0 - s.confidence) >= MIN_EPISTEMIC)
        .min_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .map(|s| Action::Explore {
            concept: s.label.clone(),
        })
        .unwrap_or(Action::Idle)
}

/// Explore/exploit balance `alpha` in `[0, 1]`, read from the Identity Subgraph:
/// exploratory values (curiosity, autonomy, creativity) pull toward exploration,
/// rigor toward exploitation. Falls back to `0.5` when no identity is present.
fn exploration_bias(graph: &CausalGraph) -> f64 {
    let identity = graph.identity();
    let mut explore = 0.0;
    let mut exploit = 0.0;
    for concept in &identity.concepts {
        if EXPLORATORY_VALUES.contains(&concept.label.as_str()) {
            explore += concept.confidence;
        } else if EXPLOITATIVE_VALUES.contains(&concept.label.as_str()) {
            exploit += concept.confidence;
        }
    }
    let total = explore + exploit;
    if total == 0.0 {
        0.5
    } else {
        explore / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use crate::graph::Relation;

    fn rel(cause: &str, effect: &str) -> Relation {
        Relation {
            cause: cause.to_string(),
            effect: effect.to_string(),
            kind: "causes".to_string(),
            strength: 0.9,
        }
    }

    #[test]
    fn exploration_bias_is_read_from_the_identity_values() {
        let mut conn = db::open(":memory:").unwrap();
        // No identity yet: balanced.
        let graph = CausalGraph::load(&conn).unwrap();
        assert!((exploration_bias(&graph) - 0.5).abs() < 1e-9);

        // Seeded identity: curiosity + autonomy + creativity (3) vs rigor (1).
        db::seed_identity(&mut conn).unwrap();
        let graph = CausalGraph::load(&conn).unwrap();
        assert!((exploration_bias(&graph) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn attend_ranks_the_connected_concept_first_and_skips_identity() {
        let mut conn = db::open(":memory:").unwrap();
        db::seed_identity(&mut conn).unwrap();
        db::seed_relations(
            &mut conn,
            &[rel("hub", "leaf one"), rel("hub", "leaf two")],
            "claude",
        )
        .unwrap();
        let graph = CausalGraph::load(&conn).unwrap();

        let plan = attend(&graph, 10);
        assert!((plan.exploration_bias - 0.75).abs() < 1e-9);
        // Identity concepts are never foci.
        assert!(plan.foci.iter().all(|f| f.concept != "numen"
            && f.concept != "curiosity"
            && f.concept != "numen operator"));
        // The connected hub outranks the leaves.
        assert_eq!(plan.foci.first().unwrap().concept, "hub");
    }

    #[test]
    fn select_action_targets_an_uncertain_concept() {
        let mut conn = db::open(":memory:").unwrap();
        db::seed_identity(&mut conn).unwrap();
        // One seed leaves these at confidence 0.5: uncertain, worth exploring.
        db::seed_relations(&mut conn, &[rel("fog", "delay")], "claude").unwrap();
        let graph = CausalGraph::load(&conn).unwrap();
        match select_action(&graph) {
            Action::Explore { concept } => assert!(concept == "fog" || concept == "delay"),
            Action::Idle => panic!("expected an exploration target"),
        }
    }

    #[test]
    fn autonomy_ladder_orders_actions_by_consequence() {
        let idle = autonomy_level(&Action::Idle);
        let explore = autonomy_level(&Action::Explore {
            concept: "x".to_string(),
        });
        assert_eq!(idle, LEVEL_IDLE);
        assert_eq!(explore, LEVEL_EXPLORE);
        assert!(
            idle < explore,
            "idle must be less consequential than explore"
        );
    }

    // Levels 4 (improve) and 5 (deploy) are not yet emitted as actions, so they are
    // exercised here as literals against the gate until their actions land.
    const IMPROVE: u8 = 4;
    const DEPLOY: u8 = 5;

    #[test]
    fn gate_lets_the_agent_explore_but_holds_self_modification_for_approval() {
        // The conservative default: explore freely, gate everything above it.
        let threshold = LEVEL_EXPLORE;
        assert_eq!(gate(LEVEL_IDLE, threshold), Gate::Proceed);
        assert_eq!(gate(LEVEL_EXPLORE, threshold), Gate::Proceed);
        assert_eq!(gate(IMPROVE, threshold), Gate::RequireApproval);
        assert_eq!(gate(DEPLOY, threshold), Gate::RequireApproval);
    }

    #[test]
    fn gate_raising_the_threshold_grants_more_autonomy() {
        // The operator can widen autonomy: at the improve level, self-rewrite proceeds
        // but a kernel deploy still needs approval.
        assert_eq!(gate(IMPROVE, IMPROVE), Gate::Proceed);
        assert_eq!(gate(DEPLOY, IMPROVE), Gate::RequireApproval);
    }

    #[test]
    fn select_action_idles_when_everything_is_confident() {
        let mut conn = db::open(":memory:").unwrap();
        db::seed_identity(&mut conn).unwrap();
        // Hammer the relation so the concepts saturate to confidence 1.0.
        for _ in 0..6 {
            db::seed_relations(&mut conn, &[rel("sure cause", "sure effect")], "claude").unwrap();
        }
        let graph = CausalGraph::load(&conn).unwrap();
        assert_eq!(select_action(&graph), Action::Idle);
    }
}
