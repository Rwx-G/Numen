//! Numen's self-critique. It reads its own metrics and asks the LLM to rank the
//! axes along which it should improve its own source - performance (slow paths),
//! relevance (too much delegation), or reasoning quality (ungrounded exploration).
//! Read-only: it proposes axes confined to the self-modifiable modules; acting on
//! one (the autonomous improve/deploy loop) is a separate, gated increment.

use serde::{Deserialize, Serialize};

use crate::agent::Failure;
use crate::metrics::Snapshot;
use crate::{evolve, llm, AppState};

/// One improvement Numen proposes for itself, grounded in a metric signal.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct ImprovementAxis {
    /// A self-modifiable module: `graph`, `aif`, `router`, or `extract`.
    pub target: String,
    /// `performance`, `relevance`, or `quality`.
    pub kind: String,
    /// The metric evidence motivating the axis.
    pub signal: String,
    /// A concrete, single-file improvement goal, usable as an `improve` goal.
    pub goal: String,
}

/// The self-metrics Numen saw plus the LLM-ranked axes it derived from them.
#[derive(Serialize)]
pub struct Critique {
    pub metrics: Snapshot,
    pub axes: Vec<ImprovementAxis>,
}

#[derive(Deserialize)]
struct AxisList {
    axes: Vec<ImprovementAxis>,
}

/// Snapshot the metrics, ask the LLM to audit them into ranked improvement axes,
/// and keep only those targeting a self-modifiable module. A parse failure yields
/// no axes rather than an error - the critique is advisory.
pub async fn critique(state: &AppState) -> Result<Critique, Failure> {
    let metrics = state.metrics.snapshot();
    let prompt = critique_prompt(&metrics);
    let answer = {
        let _permit = state.llm_permit().await?;
        llm::claude(&prompt).await.map_err(Failure::upstream)?
    };
    let axes = parse_axes(&answer)
        .into_iter()
        .filter_map(|axis| {
            writable_target(&axis.target).map(|target| ImprovementAxis { target, ..axis })
        })
        .collect();
    Ok(Critique { metrics, axes })
}

/// Normalize an LLM-named target (`graph`, `src/graph.rs`, `graph.rs`) to a bare
/// module name, returning `None` unless it is on the writable allowlist. The
/// allowlist is the same one `evolve` enforces, so the critique can never propose
/// editing a gate or the supervisor.
fn writable_target(target: &str) -> Option<String> {
    let name = target
        .trim()
        .trim_start_matches("src/")
        .trim_end_matches(".rs");
    evolve::is_writable(&format!("src/{name}.rs")).then(|| name.to_string())
}

/// Pull the JSON object out of the LLM answer: prefer a ```json fence, else the
/// span from the first `{` to the last `}`.
fn extract_json(text: &str) -> &str {
    if let Some(start) = text.find("```json") {
        let after = &text[start + "```json".len()..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    match (text.find('{'), text.rfind('}')) {
        (Some(a), Some(b)) if a <= b => &text[a..=b],
        _ => "{}",
    }
}

fn parse_axes(answer: &str) -> Vec<ImprovementAxis> {
    serde_json::from_str::<AxisList>(extract_json(answer))
        .map(|list| list.axes)
        .unwrap_or_default()
}

/// Build the self-audit prompt: the metrics, the modules Numen may touch, and the
/// JSON contract for the ranked axes.
fn critique_prompt(m: &Snapshot) -> String {
    format!(
        "You are Numen performing a self-critique to decide what to improve in your own Rust \
         source. Your self-metrics since startup:\n\
         - investigations: {inv} (delegated to Claude: {del}, delegation ratio {dr:.2}; a high \
         ratio means the local causal graph and router cover the domain poorly - a relevance gap)\n\
         - average generation latency: {ms:.0} ms (a performance signal)\n\
         - proactive explorations: {pg} grounded, {pu} ungrounded (grounding ratio {gr:.2}; low \
         means weak reasoning quality)\n\
         - consolidations: {cons}, edges pruned: {pruned}\n\n\
         You may modify ONLY these modules: graph (the petgraph causal engine, traversals, decay), \
         aif (Active Inference / Expected Free Energy planning), router (the local-vs-delegate \
         meta-router and prompt building), extract (parsing the LLM causal block, concept \
         canonicalization). For each weakness the metrics reveal, propose one improvement axis. \
         Rank them best-first; propose none if the metrics show no clear weakness.\n\n\
         Respond with ONLY a JSON object in a ```json fence, no prose around it:\n\
         {{\"axes\": [{{\"target\": \"graph|aif|router|extract\", \
         \"kind\": \"performance|relevance|quality\", \
         \"signal\": \"the metric evidence\", \
         \"goal\": \"a concrete single-file improvement\"}}]}}",
        inv = m.investigations,
        del = m.delegations,
        dr = m.delegation_ratio,
        ms = m.avg_investigate_ms,
        pg = m.proactive_grounded,
        pu = m.proactive_ungrounded,
        gr = m.grounding_ratio,
        cons = m.consolidations,
        pruned = m.edges_pruned,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_target_normalizes_and_filters_to_the_allowlist() {
        assert_eq!(writable_target("graph"), Some("graph".to_string()));
        assert_eq!(writable_target("src/aif.rs"), Some("aif".to_string()));
        assert_eq!(writable_target("router.rs"), Some("router".to_string()));
        assert_eq!(writable_target(" extract "), Some("extract".to_string()));
        // Off-list modules (a gate, the supervisor, persistence) are refused.
        assert_eq!(writable_target("evolve"), None);
        assert_eq!(writable_target("src/db.rs"), None);
        assert_eq!(writable_target("routes"), None);
    }

    #[test]
    fn parse_axes_extracts_a_fenced_json_object() {
        let answer = "Here is my critique.\n```json\n{\"axes\":[\
            {\"target\":\"graph\",\"kind\":\"performance\",\"signal\":\"slow traversal\",\"goal\":\"cache degrees\"}]}\n\
            ```\nThat is all.";
        let axes = parse_axes(answer);
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0].target, "graph");
        assert_eq!(axes[0].kind, "performance");
    }

    #[test]
    fn parse_axes_falls_back_to_a_bare_object_and_tolerates_garbage() {
        let bare = "{\"axes\":[{\"target\":\"router\",\"kind\":\"relevance\",\"signal\":\"s\",\"goal\":\"g\"}]}";
        assert_eq!(parse_axes(bare).len(), 1);
        assert!(parse_axes("no json here").is_empty());
    }
}
