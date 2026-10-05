//! Numen's self-metrics: cheap atomic counters on the hot paths that feed the
//! self-critique. They live in RAM since process start (surviving a self-deploy
//! restart is a deliberate follow-up); a `Snapshot` with derived ratios is read
//! for `/stats` (free) and `/critique` (the LLM self-audit).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::Serialize;

/// Running counters Numen keeps about its own behaviour. Updates are `Relaxed`:
/// exact cross-counter ordering does not matter for a coarse self-assessment, and
/// the instrumented hot paths must not pay for synchronization.
#[derive(Default)]
pub struct Metrics {
    local_answers: AtomicU64,
    delegations: AtomicU64,
    investigate_nanos: AtomicU64,
    proactive_grounded: AtomicU64,
    proactive_ungrounded: AtomicU64,
    consolidations: AtomicU64,
    edges_pruned: AtomicU64,
}

/// The raw cumulative counters, persisted so the lifetime self-view survives a
/// self-deploy restart (the metrics drive the self-critique, which must not forget
/// across an improvement).
#[derive(Clone, Copy, Default)]
pub struct Counters {
    pub local_answers: u64,
    pub delegations: u64,
    pub investigate_nanos: u64,
    pub proactive_grounded: u64,
    pub proactive_ungrounded: u64,
    pub consolidations: u64,
    pub edges_pruned: u64,
}

impl Metrics {
    /// Build metrics pre-seeded from persisted counters, restoring the lifetime
    /// view at startup.
    pub fn from_counters(c: Counters) -> Self {
        Self {
            local_answers: AtomicU64::new(c.local_answers),
            delegations: AtomicU64::new(c.delegations),
            investigate_nanos: AtomicU64::new(c.investigate_nanos),
            proactive_grounded: AtomicU64::new(c.proactive_grounded),
            proactive_ungrounded: AtomicU64::new(c.proactive_ungrounded),
            consolidations: AtomicU64::new(c.consolidations),
            edges_pruned: AtomicU64::new(c.edges_pruned),
        }
    }

    /// The current raw counters, for persistence.
    pub fn counters(&self) -> Counters {
        Counters {
            local_answers: self.local_answers.load(Ordering::Relaxed),
            delegations: self.delegations.load(Ordering::Relaxed),
            investigate_nanos: self.investigate_nanos.load(Ordering::Relaxed),
            proactive_grounded: self.proactive_grounded.load(Ordering::Relaxed),
            proactive_ungrounded: self.proactive_ungrounded.load(Ordering::Relaxed),
            consolidations: self.consolidations.load(Ordering::Relaxed),
            edges_pruned: self.edges_pruned.load(Ordering::Relaxed),
        }
    }

    /// Record one investigation: whether it was answered locally or delegated to
    /// Claude, and how long the generation took. `source` is the same
    /// `"local"`/`"claude"` tag `investigate` already assigns.
    pub fn record_generation(&self, source: &str, elapsed: Duration) {
        if source == "local" {
            self.local_answers.fetch_add(1, Ordering::Relaxed);
        } else {
            self.delegations.fetch_add(1, Ordering::Relaxed);
        }
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.investigate_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Record a proactive exploration's outcome: whether it grounded its target.
    pub fn record_proactive(&self, grounded: bool) {
        let counter = if grounded {
            &self.proactive_grounded
        } else {
            &self.proactive_ungrounded
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a consolidation pass and how many edges it pruned.
    pub fn record_consolidation(&self, edges_pruned: u64) {
        self.consolidations.fetch_add(1, Ordering::Relaxed);
        self.edges_pruned.fetch_add(edges_pruned, Ordering::Relaxed);
    }

    /// A point-in-time read of the counters with the ratios the critique reasons
    /// over.
    pub fn snapshot(&self) -> Snapshot {
        let local = self.local_answers.load(Ordering::Relaxed);
        let delegated = self.delegations.load(Ordering::Relaxed);
        let nanos = self.investigate_nanos.load(Ordering::Relaxed);
        let grounded = self.proactive_grounded.load(Ordering::Relaxed);
        let ungrounded = self.proactive_ungrounded.load(Ordering::Relaxed);
        let investigations = local + delegated;
        Snapshot {
            local_answers: local,
            delegations: delegated,
            investigations,
            delegation_ratio: ratio(delegated, investigations),
            avg_investigate_ms: if investigations == 0 {
                0.0
            } else {
                nanos as f64 / investigations as f64 / 1e6
            },
            proactive_grounded: grounded,
            proactive_ungrounded: ungrounded,
            grounding_ratio: ratio(grounded, grounded + ungrounded),
            consolidations: self.consolidations.load(Ordering::Relaxed),
            edges_pruned: self.edges_pruned.load(Ordering::Relaxed),
        }
    }
}

/// `part / whole`, or `0.0` when `whole` is zero.
fn ratio(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// A point-in-time view of the metrics with the derived ratios. Serialized into
/// `/stats` and handed to the self-critique.
#[derive(Serialize, Clone)]
pub struct Snapshot {
    pub local_answers: u64,
    pub delegations: u64,
    pub investigations: u64,
    /// Fraction of investigations that had to delegate to Claude. High means the
    /// local graph and router cover the domain poorly - a relevance signal.
    pub delegation_ratio: f64,
    /// Mean wall-clock of a generation, in milliseconds - a performance signal.
    pub avg_investigate_ms: f64,
    pub proactive_grounded: u64,
    pub proactive_ungrounded: u64,
    /// Fraction of proactive explorations that grounded their target - a
    /// reasoning-quality signal.
    pub grounding_ratio: f64,
    pub consolidations: u64,
    pub edges_pruned: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_counts_and_derives_ratios() {
        let m = Metrics::default();
        m.record_generation("local", Duration::from_millis(10));
        m.record_generation("claude", Duration::from_millis(30));
        m.record_generation("claude", Duration::from_millis(50));

        let s = m.snapshot();
        assert_eq!(s.investigations, 3);
        assert_eq!(s.local_answers, 1);
        assert_eq!(s.delegations, 2);
        assert!((s.delegation_ratio - 2.0 / 3.0).abs() < 1e-9);
        assert!((s.avg_investigate_ms - 30.0).abs() < 1e-6); // (10+30+50)/3 ms

        m.record_proactive(true);
        m.record_proactive(false);
        m.record_consolidation(4);
        let s = m.snapshot();
        assert!((s.grounding_ratio - 0.5).abs() < 1e-9);
        assert_eq!(s.consolidations, 1);
        assert_eq!(s.edges_pruned, 4);
    }

    #[test]
    fn snapshot_is_zeroed_with_no_activity() {
        let s = Metrics::default().snapshot();
        assert_eq!(s.investigations, 0);
        assert_eq!(s.delegation_ratio, 0.0);
        assert_eq!(s.avg_investigate_ms, 0.0);
        assert_eq!(s.grounding_ratio, 0.0);
    }
}
