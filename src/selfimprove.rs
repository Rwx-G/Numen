//! The autonomous self-improvement decision. Given the self-critique's ranked
//! axes and the recent history (when the agent last self-deployed, which modules
//! are cooling down), it decides whether to improve now and which axis to take.
//!
//! Pure decision logic, separately tested. Acting on a decision - running
//! `improve`, notifying the operator, requesting a deploy - lives in `agent` and is off
//! by default (`NUMEN_AUTONOMOUS_IMPROVE`). Times are unix seconds so the
//! rate-limit can be reasoned about across the very restart a deploy triggers.

use std::collections::HashMap;

use crate::selfcritique::ImprovementAxis;

/// Minimum gap between two autonomous self-deploys (seconds), so the agent cannot
/// rewrite itself in a tight loop. The primary loop guard, and it survives the
/// restart (the kernel marks itself deployed on a wakeup boot).
pub const DEFAULT_RATE_LIMIT_SECS: u64 = 6 * 3600;
/// How long a module is held back after it was just improved (seconds), so the
/// agent moves on instead of hammering the same axis.
pub const DEFAULT_COOLDOWN_SECS: u64 = 24 * 3600;

/// What the autonomous loop should do this tick.
#[derive(Debug, PartialEq)]
pub enum Decision {
    /// Take this axis: improve the target module toward the goal, then deploy.
    Improve(ImprovementAxis),
    /// Do nothing, with the reason (for logging).
    Hold(&'static str),
}

/// The history the decision reads and the orchestration updates: when the agent
/// last self-deployed and, per module, the unix second its cooldown lifts.
pub struct History {
    last_deploy: Option<u64>,
    cooldowns: HashMap<String, u64>,
    rate_limit_secs: u64,
    cooldown_secs: u64,
}

impl Default for History {
    fn default() -> Self {
        Self::new(DEFAULT_RATE_LIMIT_SECS, DEFAULT_COOLDOWN_SECS)
    }
}

impl History {
    pub fn new(rate_limit_secs: u64, cooldown_secs: u64) -> Self {
        Self {
            last_deploy: None,
            cooldowns: HashMap::new(),
            rate_limit_secs,
            cooldown_secs,
        }
    }

    /// Record that the agent just self-deployed (opens the rate-limit window). Used
    /// both after an autonomous deploy and at startup when the kernel booted from a
    /// self-deploy, so the rate-limit survives the restart it triggered.
    pub fn mark_deployed(&mut self, now: u64) {
        self.last_deploy = Some(now);
    }

    /// Hold `module` back until `cooldown_secs` from now, so the agent moves on
    /// instead of re-improving the same module immediately.
    pub fn cool_down(&mut self, now: u64, module: &str) {
        self.cooldowns
            .insert(module.to_string(), now + self.cooldown_secs);
    }

    /// Whether the rate-limit window has passed - checked before the (costly)
    /// critique so a rate-limited tick does not delegate to the LLM at all.
    pub fn ready(&self, now: u64) -> bool {
        !self.rate_limited(now)
    }

    fn rate_limited(&self, now: u64) -> bool {
        self.last_deploy
            .is_some_and(|t| now.saturating_sub(t) < self.rate_limit_secs)
    }

    fn cooling(&self, now: u64, module: &str) -> bool {
        self.cooldowns.get(module).is_some_and(|&until| now < until)
    }
}

/// Decide whether to self-improve now: never within the rate-limit window, and
/// only on the best-ranked axis whose module is not cooling down.
pub fn decide(now: u64, history: &History, axes: &[ImprovementAxis]) -> Decision {
    if history.rate_limited(now) {
        return Decision::Hold("rate-limited since the last self-deploy");
    }
    match axes.iter().find(|axis| !history.cooling(now, &axis.target)) {
        Some(axis) => Decision::Improve(axis.clone()),
        None if axes.is_empty() => Decision::Hold("the critique proposed no axis"),
        None => Decision::Hold("every proposed axis is cooling down"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(target: &str) -> ImprovementAxis {
        ImprovementAxis {
            target: target.to_string(),
            kind: "performance".to_string(),
            signal: "slow".to_string(),
            goal: "speed it up".to_string(),
        }
    }

    #[test]
    fn decide_holds_within_the_rate_limit_window() {
        let mut history = History::default();
        history.mark_deployed(1_000);
        assert!(!history.ready(1_000 + 3_600)); // 1h in, still rate-limited
        assert_eq!(
            decide(1_000 + 3_600, &history, &[axis("aif")]),
            Decision::Hold("rate-limited since the last self-deploy")
        );
        // 7 hours later, the window has passed.
        assert!(history.ready(1_000 + 7 * 3_600));
        assert_eq!(
            decide(1_000 + 7 * 3_600, &history, &[axis("aif")]),
            Decision::Improve(axis("aif"))
        );
    }

    #[test]
    fn decide_takes_the_top_axis_whose_module_is_not_cooling() {
        let mut history = History::default();
        history.mark_deployed(0);
        history.cool_down(0, "graph"); // graph cools down for 24h
        let now = 7 * 3_600; // past the rate limit, inside graph's cooldown
                             // graph is cooling, so the second axis wins.
        assert_eq!(
            decide(now, &history, &[axis("graph"), axis("router")]),
            Decision::Improve(axis("router"))
        );
    }

    #[test]
    fn decide_holds_when_there_is_nothing_to_do() {
        let history = History::default();
        assert_eq!(
            decide(10_000, &history, &[]),
            Decision::Hold("the critique proposed no axis")
        );
    }

    #[test]
    fn decide_holds_when_every_axis_is_cooling() {
        let mut history = History::default();
        history.mark_deployed(0);
        history.cool_down(0, "graph");
        let now = 7 * 3_600; // past the rate limit
        assert_eq!(
            decide(now, &history, &[axis("graph")]),
            Decision::Hold("every proposed axis is cooling down")
        );
    }
}
