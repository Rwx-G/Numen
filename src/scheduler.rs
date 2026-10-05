//! Offline compute lane. A background task runs memory consolidation on a
//! timer, but only while the agent is idle - interactive work takes priority,
//! matching the brief's online/offline scheduler (consolidation happens during
//! the quiet "sleep" periods, not under load).

use std::time::Duration;

use anyhow::Result;

use crate::{db, graph, AppState};

/// Decay rate for the semantic graph: ~180-day half-life, per day. Episodic
/// memory forgets far faster, expiring after the TTL below.
pub const SEMANTIC_DECAY_LAMBDA: f64 = 0.003_85;
pub const EDGE_PRUNE_FLOOR: f64 = 0.05;
pub const EPISODE_TTL_DAYS: f64 = 30.0;

/// Run one consolidation pass against the store and refresh the in-memory graph.
/// Shared by the `/consolidate` endpoint and the background scheduler.
pub fn consolidate_once(state: &AppState) -> Result<db::ConsolidationStats> {
    // Hold the graph write lock across the DB transaction and the in-memory
    // refresh so a concurrent seed cannot interleave (lock order: graph, db).
    let mut graph = state.graph_write();
    let mut conn = state.db();
    let stats = db::consolidate(
        &mut conn,
        SEMANTIC_DECAY_LAMBDA,
        EDGE_PRUNE_FLOOR,
        EPISODE_TTL_DAYS,
    )?;
    *graph = graph::CausalGraph::load(&conn)?;
    Ok(stats)
}

/// Background loop: every `tick`, consolidate only if the agent has been idle
/// for at least `idle_threshold`; otherwise defer so interactive work is never
/// contended.
pub async fn run(
    state: AppState,
    tick: Duration,
    idle_threshold: Duration,
    proactive: bool,
    autonomous: bool,
    booted_from_deploy: bool,
    feeds: Vec<String>,
) {
    let mut interval = tokio::time::interval(tick);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Self-improvement history (rate-limit + per-module cooldowns). If this kernel
    // booted from a self-deploy, open the rate-limit window so it cannot immediately
    // redeploy across the restart it just made.
    let mut improve_history = crate::selfimprove::History::default();
    if booted_from_deploy {
        improve_history.mark_deployed(crate::agent::unix_now());
    }

    loop {
        interval.tick().await;

        let idle = state.idle_for();
        if idle < idle_threshold {
            tracing::debug!(idle_secs = idle.as_secs(), "scheduler: busy, deferring");
            continue;
        }

        // Run the synchronous SQLite work off the async runtime threads.
        let result = {
            let state = state.clone();
            tokio::task::spawn_blocking(move || consolidate_once(&state)).await
        };
        match result {
            Ok(Ok(stats)) => {
                state
                    .metrics
                    .record_consolidation(stats.edges_pruned as u64);
                // Persist the lifetime self-metrics during the same quiet window so
                // they survive a self-deploy restart (the self-critique relies on
                // them not resetting on every improvement).
                {
                    let conn = state.db();
                    if let Err(err) = db::save_metrics(&conn, &state.metrics.counters()) {
                        tracing::warn!(error = %err, "scheduler: failed to persist metrics");
                    }
                }
                if stats.edges_pruned + stats.nodes_pruned + stats.episodes_expired > 0 {
                    state.record_event(
                        "consolidate",
                        format!(
                            "pruned {} edges, {} nodes, {} episodes",
                            stats.edges_pruned, stats.nodes_pruned, stats.episodes_expired
                        ),
                    );
                    tracing::info!(
                        edges_pruned = stats.edges_pruned,
                        nodes_pruned = stats.nodes_pruned,
                        episodes_expired = stats.episodes_expired,
                        "scheduler: consolidated"
                    );
                } else {
                    tracing::debug!(
                        edges_decayed = stats.edges_decayed,
                        "scheduler: decayed, nothing pruned"
                    );
                }
            }
            Ok(Err(err)) => tracing::warn!(error = %err, "scheduler: consolidation failed"),
            Err(join_err) => {
                tracing::warn!(error = %join_err, "scheduler: consolidation task panicked")
            }
        }

        // After consolidating, take one Active Inference step if proactive mode is
        // on: explore the biggest knowledge gap autonomously (bounded to one per
        // idle tick, gated so it never burns the LLM quota unasked).
        if proactive {
            crate::agent::proactive_explore(&state).await;
        }

        // When exploration is exhausted, the agent may improve its own source
        // (off by default; the decision is rate-limited and cooldown-gated, the
        // sandboxed build gate and the kill switch are the safety net).
        if autonomous {
            crate::agent::autonomous_improve(&state, &mut improve_history).await;
        }

        // Pull the configured world feeds during the same quiet window. Entries
        // are dedup'd by id, so re-fetches are cheap and only genuinely new
        // entries delegate to the LLM.
        if !feeds.is_empty() {
            crate::agent::ingest_feeds(&state, &feeds).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::Instant;

    use crate::graph::Relation;

    #[test]
    fn consolidate_once_prunes_and_refreshes_graph() {
        let conn = db::open(":memory:").unwrap();
        let causal_graph = graph::CausalGraph::load(&conn).unwrap();
        let state = AppState {
            db: Arc::new(Mutex::new(conn)),
            graph: Arc::new(RwLock::new(causal_graph)),
            backend: Arc::new(crate::llm::Backend::cloud_only()),
            embedder: Arc::new(None),
            workshop: Arc::new(tokio::sync::Mutex::new(None)),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            llm_permits: Arc::new(tokio::sync::Semaphore::new(1)),
            api_token: None,
            effector: None,
            http: reqwest::Client::new(),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            metrics: Arc::new(crate::metrics::Metrics::default()),
            events: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        };

        {
            let mut conn = state.db.lock().unwrap();
            let relation = Relation {
                cause: "a".to_string(),
                effect: "b".to_string(),
                kind: "causes".to_string(),
                strength: 0.8,
            };
            db::seed_relations(&mut conn, &[relation], "claude").unwrap();
            conn.execute("UPDATE edges SET last_accessed = '2000-01-01 00:00:00'", [])
                .unwrap();
        }

        let stats = consolidate_once(&state).unwrap();
        assert_eq!(stats.edges_pruned, 1);
        assert_eq!(state.graph.read().unwrap().edge_count(), 0);
    }
}
