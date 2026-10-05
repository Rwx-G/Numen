//! Self-improvement workshop: Numen reads, patches, and verifies its own source.
//!
//! The deterministic gate (`build` + `test` + `clippy` must all pass) is
//! authoritative and fail-closed: a change that does not pass is discarded, never
//! deployed. Every accepted change is a git commit on a branch, so it is
//! auditable and revertible, and deployment stays gated by the operator (the
//! corrigibility edge). The workshop is absent unless `NUMEN_SRC_DIR` points at
//! the repo and a toolchain is present, so the running agent cannot touch its own
//! source by default.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use tokio::process::Command;

/// Cap on a source file Numen will read or write, and on captured tool output.
const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_CHARS: usize = 8000;

/// Wall-clock ceilings on the subprocesses the gate runs, so a hung or
/// pathological `cargo`/`git` cannot block the workshop lock forever.
const CARGO_TIMEOUT_SECS: u64 = 1800;
const GIT_TIMEOUT_SECS: u64 = 60;

/// The ONLY files Numen may rewrite: pure reasoning and algorithm modules. This
/// is an allowlist, not a denylist - everything else (the verify gate and its
/// orchestration in `agent.rs`, the auth and endpoints in `routes.rs`, the wiring
/// and shutdown in `main.rs`, the workshop itself, the supervisor, the Cargo
/// manifests, the persistence/migrations in `db.rs`, the supply-chain policy) is
/// refused, so the kernel cannot edit out its own guardrails even via a path that
/// was not foreseen. Expand deliberately, never by the agent. The trusted
/// supervisor enforces an identical hardcoded list at deploy time (it does not
/// trust this mutable copy); keep the two in sync.
const WRITABLE: [&str; 4] = [
    "src/graph.rs",
    "src/aif.rs",
    "src/router.rs",
    "src/extract.rs",
];

/// A handle on Numen's own source tree, present only when `NUMEN_SRC_DIR` is a
/// git checkout with a `Cargo.toml`.
pub struct Workshop {
    src_dir: PathBuf,
}

/// The outcome of the verification gate: which stage failed (if any) and the
/// tail of its output, bounded.
pub struct VerifyReport {
    pub passed: bool,
    pub failed_stage: Option<String>,
    pub output: String,
}

struct StageResult {
    ok: bool,
    output: String,
}

impl Workshop {
    /// Open the workshop from `NUMEN_SRC_DIR`. `None` when unset, not a directory,
    /// or not a git checkout with a `Cargo.toml` - self-improvement is then simply
    /// unavailable rather than a hard dependency.
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var("NUMEN_SRC_DIR").ok()?;
        let src_dir = Path::new(&dir).canonicalize().ok()?;
        (src_dir.join("Cargo.toml").is_file() && src_dir.join(".git").exists())
            .then_some(Self { src_dir })
    }

    /// Read a source file, confined to the tree: the relative path may not be
    /// absolute, may not contain `..`, and must resolve inside the source root.
    pub fn read_source(&self, rel: &str) -> Result<String> {
        let path = self.confine(rel)?;
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        if bytes.len() > MAX_SOURCE_BYTES {
            bail!("source file {rel} exceeds the read cap");
        }
        String::from_utf8(bytes).context("source is not valid UTF-8")
    }

    /// Write a source file, confined to the tree under the same rules. Used to
    /// apply a proposed patch before verification.
    pub fn write_source(&self, rel: &str, content: &str) -> Result<()> {
        if !is_writable(rel) {
            bail!("{rel} is not in the self-modifiable allowlist (reasoning modules only)");
        }
        if content.len() > MAX_SOURCE_BYTES {
            bail!("proposed content for {rel} exceeds the write cap");
        }
        let path = self.confined_path(rel)?;
        std::fs::write(&path, content).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// The authoritative gate: the working tree must build, pass every test, and
    /// lint clean. Stages run in order and stop at the first failure, whose output
    /// tail is returned for diagnosis.
    pub async fn verify(&self) -> Result<VerifyReport> {
        let stages: [(&str, &[&str]); 3] = [
            ("build", &["build"]),
            ("test", &["test"]),
            (
                "clippy",
                &["clippy", "--all-targets", "--", "-D", "warnings"],
            ),
        ];
        for (stage, args) in stages {
            let result = self.cargo(args).await?;
            if !result.ok {
                return Ok(VerifyReport {
                    passed: false,
                    failed_stage: Some(stage.to_string()),
                    output: result.output,
                });
            }
        }
        Ok(VerifyReport {
            passed: true,
            failed_stage: None,
            output: String::new(),
        })
    }

    /// Start a fresh branch for a self-improvement attempt.
    pub async fn start_branch(&self, name: &str) -> Result<()> {
        self.git(&["checkout", "-b", name]).await
    }

    /// Commit only the rewritten file on the current branch. Never `git add -A`,
    /// which could stage untracked secrets (`.env`) or local artifacts.
    pub async fn commit(&self, file: &str, message: &str) -> Result<()> {
        self.git(&["add", "--", file]).await?;
        self.git(&["commit", "-m", message]).await
    }

    /// Discard an unverified attempt: revert tracked changes and drop any new
    /// files the patch introduced, leaving the tree clean.
    pub async fn discard(&self) -> Result<()> {
        self.git(&["checkout", "--", "."]).await?;
        self.git(&["clean", "-fd"]).await
    }

    /// The branch currently checked out, to return to after an attempt.
    pub async fn current_branch(&self) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(&self.src_dir)
            .output()
            .await
            .context("run git rev-parse")?;
        if !output.status.success() {
            bail!("git rev-parse failed: {}", tail(&output.stderr, 500));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Switch back to an existing branch.
    pub async fn checkout(&self, branch: &str) -> Result<()> {
        self.git(&["checkout", branch]).await
    }

    /// Delete an abandoned attempt branch.
    pub async fn delete_branch(&self, branch: &str) -> Result<()> {
        self.git(&["branch", "-D", branch]).await
    }

    /// Request a deploy of an already-verified branch by writing its name to the
    /// deploy marker. The supervisor - the trusted std-only parent - then checks it
    /// out, re-runs the gate, and swaps the binary IT builds. The kernel never
    /// builds or stages the deployed binary, so a compromised kernel cannot
    /// self-certify a deploy. The marker path matches the supervisor's env.
    pub fn request_deploy(&self, branch: &str) -> Result<()> {
        let marker = deploy_path("NUMEN_DEPLOY_MARKER", "/app/deploy.request");
        std::fs::write(&marker, branch).with_context(|| format!("write deploy marker {marker}"))?;
        Ok(())
    }

    /// Resolve `rel` against the source root and confine it: reject absolute paths
    /// and `..`, then require the canonical path to stay inside the root. For files
    /// that must already exist (reads).
    fn confine(&self, rel: &str) -> Result<PathBuf> {
        if normalize_rel(rel).is_none() {
            bail!("path {rel} is absolute or escapes the source tree");
        }
        let canonical = self
            .src_dir
            .join(rel)
            .canonicalize()
            .with_context(|| format!("resolve {rel}"))?;
        if !canonical.starts_with(&self.src_dir) {
            bail!("path {rel} resolves outside the source tree");
        }
        Ok(canonical)
    }

    /// Like `confine`, but for a path that may not exist yet (writes): the parent
    /// directory must exist and resolve inside the source root.
    fn confined_path(&self, rel: &str) -> Result<PathBuf> {
        if normalize_rel(rel).is_none() {
            bail!("path {rel} is absolute or escapes the source tree");
        }
        let path = self.src_dir.join(rel);
        let parent = path.parent().context("path has no parent directory")?;
        let parent_canonical = parent
            .canonicalize()
            .with_context(|| format!("resolve parent of {rel}"))?;
        if !parent_canonical.starts_with(&self.src_dir) {
            bail!("path {rel} resolves outside the source tree");
        }
        Ok(path)
    }

    /// Run a subprocess under a wall-clock deadline, killing it (and its children,
    /// via `kill_on_drop`) if it overruns rather than blocking the workshop.
    async fn run_tool(
        &self,
        program: &str,
        args: &[&str],
        secs: u64,
    ) -> Result<std::process::Output> {
        let future = Command::new(program)
            .args(args)
            .current_dir(&self.src_dir)
            .kill_on_drop(true)
            .output();
        match tokio::time::timeout(Duration::from_secs(secs), future).await {
            Ok(result) => result.with_context(|| format!("run {program} {}", args.join(" "))),
            Err(_) => bail!("{program} {} timed out after {secs}s", args.join(" ")),
        }
    }

    async fn cargo(&self, args: &[&str]) -> Result<StageResult> {
        let output = self.run_tool("cargo", args, CARGO_TIMEOUT_SECS).await?;
        let mut combined = tail(&output.stdout, MAX_OUTPUT_CHARS / 2);
        combined.push_str(&tail(&output.stderr, MAX_OUTPUT_CHARS / 2));
        Ok(StageResult {
            ok: output.status.success(),
            output: combined,
        })
    }

    async fn git(&self, args: &[&str]) -> Result<()> {
        let output = self.run_tool("git", args, GIT_TIMEOUT_SECS).await?;
        if !output.status.success() {
            bail!(
                "git {} failed: {}",
                args.join(" "),
                tail(&output.stderr, 500)
            );
        }
        Ok(())
    }
}

/// A deploy path from the environment (shared with the supervisor), or a default.
fn deploy_path(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Normalize a relative path lexically: collapse `.` and empty components, reject
/// `..`, absolute, root, and Windows-prefix components. Returns the clean `a/b/c`
/// form, or `None` if unsafe. The basis for confinement and the allowlist, so
/// `./src/graph.rs` or `src//graph.rs` cannot dodge the writable check.
fn normalize_rel(rel: &str) -> Option<String> {
    let path = Path::new(rel);
    if path.is_absolute() {
        return None;
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_str()?),
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// Whether Numen may rewrite this path: it must normalize cleanly and match the
/// writable allowlist exactly. Everything off the list (the gate, auth, deploy,
/// the workshop, the supervisor, Cargo, persistence) is refused.
pub(crate) fn is_writable(rel: &str) -> bool {
    normalize_rel(rel).is_some_and(|norm| WRITABLE.contains(&norm.as_str()))
}

/// The last `max_chars` characters of captured output, on a char boundary.
fn tail(bytes: &[u8], max_chars: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        text.into_owned()
    } else {
        chars[chars.len() - max_chars..].iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_rel_collapses_and_rejects_unsafe() {
        assert_eq!(
            normalize_rel("src/graph.rs").as_deref(),
            Some("src/graph.rs")
        );
        assert_eq!(
            normalize_rel("./src/graph.rs").as_deref(),
            Some("src/graph.rs")
        );
        assert_eq!(
            normalize_rel("src//graph.rs").as_deref(),
            Some("src/graph.rs")
        );
        assert!(normalize_rel("../secrets").is_none());
        assert!(normalize_rel("src/../../etc/passwd").is_none());
        assert!(normalize_rel("/etc/passwd").is_none());
    }

    #[test]
    fn tail_keeps_the_end_within_the_cap() {
        assert_eq!(tail(b"abcdef", 3), "def");
        assert_eq!(tail(b"ab", 5), "ab");
    }

    #[test]
    fn only_allowlisted_reasoning_files_are_writable() {
        assert!(is_writable("src/graph.rs"));
        assert!(is_writable("src/aif.rs"));
        // The allowlist defeats the path-form bypasses a denylist missed.
        assert!(!is_writable("./src/evolve.rs"));
        assert!(!is_writable("src//evolve.rs"));
        // The gate, auth, deploy, wiring, supervisor, manifests: all refused.
        assert!(!is_writable("src/evolve.rs"));
        assert!(!is_writable("src/agent.rs"));
        assert!(!is_writable("src/routes.rs"));
        assert!(!is_writable("src/main.rs"));
        assert!(!is_writable("src/bin/numen-supervisor.rs"));
        assert!(!is_writable("Cargo.toml"));
        assert!(!is_writable("deny.toml"));
    }
}
