//! Numen supervisor: the small, boring parent process that owns the kernel's
//! lifecycle so the kernel can rewrite itself safely.
//!
//! It spawns the kernel, health-checks it through a probation window, and on a
//! kernel-requested deploy atomically swaps in the staged binary, respawns, and
//! re-probations - rolling back to the previous binary if the new kernel does not
//! come up healthy. It also owns the kill switch. This binary is deliberately
//! std-only and separately versioned: the kernel's self-improvement workshop
//! refuses to rewrite it, so a buggy or adversarial kernel cannot disable its own
//! rollback, crash-loop guard, or stop control.

#![deny(unsafe_code)]
#![warn(clippy::all)]

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;
use std::{env, fs, thread};

/// How long the new kernel has to report healthy after a spawn, in seconds.
const PROBATION_SECS: u64 = 60;
/// Consecutive unrecoverable kernel failures before the supervisor gives up.
const MAX_CRASHES: u32 = 3;
/// Pause between respawns after a crash, in seconds.
const BACKOFF_SECS: u64 = 5;

/// Files the kernel is allowed to self-modify, hardcoded in the trusted supervisor
/// so it never depends on the kernel's (mutable) copy. A deploy branch that changes
/// anything else is refused before it is built. Must match `evolve::WRITABLE`.
const WRITABLE: [&str; 4] = [
    "src/graph.rs",
    "src/aif.rs",
    "src/router.rs",
    "src/extract.rs",
];

/// Whether a git-relative path is one the kernel may self-modify.
fn is_writable(path: &str) -> bool {
    WRITABLE.contains(&path)
}

struct Config {
    /// The runnable kernel binary the supervisor spawns and swaps.
    kernel: String,
    /// The writable source checkout the supervisor builds the kernel from.
    src_dir: String,
    /// The branch to return to after an aborted deploy.
    base_branch: String,
    /// Backup of the current kernel, restored on a failed deploy.
    backup: String,
    /// File the kernel writes the branch name into to request a deploy.
    deploy_marker: String,
    /// File the supervisor touches so the freshly swapped kernel announces itself.
    wakeup: String,
    /// File whose presence stops the supervisor; the kernel cannot create it.
    kill_switch: String,
    /// `host:port` the kernel serves `/health` on.
    health: String,
    /// The SQLite database, snapshotted before a swap so a rollback reverts a
    /// migration the new kernel may have applied, not just the binary.
    db: String,
    db_backup: String,
    /// Run the build/test/clippy gate inside a bubblewrap sandbox (no network,
    /// read-only root). On by default; `NUMEN_GATE_NO_SANDBOX` opts out.
    sandbox_gate: bool,
    /// Cargo home, bind-mounted writable inside the sandbox so the gate build can
    /// use the pre-populated registry and write its lock files.
    cargo_home: String,
    /// Cargo features the gate builds/tests/lints with (`NUMEN_BUILD_FEATURES`),
    /// e.g. `cuda` on the GPU image so a self-deploy keeps GPU inference. `None`
    /// builds the default (CPU) feature set.
    build_features: Option<String>,
}

impl Config {
    fn from_env() -> Self {
        let kernel = env_or("NUMEN_KERNEL_BIN", "/app/numen");
        let backup = format!("{kernel}.prev");
        let db = env_or("NUMEN_DB_PATH", "/data/numen.db");
        let db_backup = format!("{db}.prev");
        let cargo_home = env::var("CARGO_HOME")
            .unwrap_or_else(|_| format!("{}/.cargo", env_or("HOME", "/home/numen")));
        Self {
            src_dir: env_or("NUMEN_SRC_DIR", "/app/src"),
            base_branch: env_or("NUMEN_BASE_BRANCH", "main"),
            deploy_marker: env_or("NUMEN_DEPLOY_MARKER", "/app/deploy.request"),
            wakeup: env_or("NUMEN_WAKEUP_MARKER", "/app/wakeup"),
            kill_switch: env_or("NUMEN_KILL_SWITCH", "/app/STOP"),
            health: env_or("NUMEN_HEALTH", "127.0.0.1:8080"),
            sandbox_gate: !env_flag("NUMEN_GATE_NO_SANDBOX"),
            cargo_home,
            build_features: env::var("NUMEN_BUILD_FEATURES")
                .ok()
                .filter(|f| !f.is_empty()),
            db,
            db_backup,
            kernel,
            backup,
        }
    }
}

/// How one kernel run ended, classified for the supervisor's next move.
#[derive(Debug, PartialEq)]
enum Outcome {
    DeployRequested(String),
    CleanExit,
    Crash,
    ProbationFailed,
    KillSwitch,
}

fn main() {
    let cfg = Config::from_env();
    log(&format!("starting, kernel={}", cfg.kernel));
    let mut crashes = 0u32;

    loop {
        if Path::new(&cfg.kill_switch).exists() {
            log("kill switch present, stopping");
            break;
        }
        match run_once(&cfg) {
            Outcome::DeployRequested(branch) => {
                log(&format!(
                    "deploy requested for {branch}; verifying in the supervisor"
                ));
                match build_verified(&cfg, &branch) {
                    Ok(()) => match swap_to_built(&cfg) {
                        Ok(()) => log("staged build verified and swapped in"),
                        Err(err) => log(&format!("swap failed, keeping current kernel: {err}")),
                    },
                    Err(stage) => {
                        log(&format!("deploy gate failed at {stage}; not swapping"));
                        let _ = fs::remove_file(&cfg.deploy_marker);
                        let _ = run_tool(&cfg.src_dir, "git", &["checkout", &cfg.base_branch]);
                    }
                }
                crashes = 0;
            }
            Outcome::ProbationFailed => {
                if rollback(&cfg) {
                    log("new kernel failed health probation, rolled back to previous");
                    crashes = 0;
                } else {
                    crashes += 1;
                    log(&format!(
                        "kernel failed probation with no rollback target ({crashes}/{MAX_CRASHES})"
                    ));
                    if crashes >= MAX_CRASHES {
                        log("giving up");
                        break;
                    }
                    thread::sleep(Duration::from_secs(BACKOFF_SECS));
                }
            }
            Outcome::Crash => {
                crashes += 1;
                log(&format!("kernel crashed ({crashes}/{MAX_CRASHES})"));
                if crashes >= MAX_CRASHES {
                    log("crash loop, stopping");
                    break;
                }
                thread::sleep(Duration::from_secs(BACKOFF_SECS));
            }
            Outcome::CleanExit => {
                log("kernel exited cleanly, stopping");
                break;
            }
            Outcome::KillSwitch => {
                log("kill switch, stopping");
                break;
            }
        }
    }
}

/// Spawn the kernel, probation-check it, then wait for it to exit and classify why.
fn run_once(cfg: &Config) -> Outcome {
    let mut child = match Command::new(&cfg.kernel).spawn() {
        Ok(child) => child,
        Err(err) => {
            log(&format!("failed to spawn kernel: {err}"));
            return Outcome::Crash;
        }
    };

    if !probation(&mut child, &cfg.health) {
        let _ = child.kill();
        let _ = child.wait();
        return Outcome::ProbationFailed;
    }
    log("kernel healthy");

    let status = match child.wait() {
        Ok(status) => status,
        Err(err) => {
            log(&format!("lost track of kernel: {err}"));
            return Outcome::Crash;
        }
    };

    let kill_switch = Path::new(&cfg.kill_switch).exists();
    let deploy = Path::new(&cfg.deploy_marker).exists().then(|| {
        fs::read_to_string(&cfg.deploy_marker)
            .unwrap_or_default()
            .trim()
            .to_string()
    });
    classify_exit(kill_switch, deploy, status.success())
}

/// Decide what a finished kernel run means, in strict precedence: the kill switch
/// trumps everything, then a pending deploy request, then a clean versus crash
/// exit. Pure so the precedence is unit-testable without spawning a kernel.
fn classify_exit(kill_switch: bool, deploy: Option<String>, clean: bool) -> Outcome {
    if kill_switch {
        Outcome::KillSwitch
    } else if let Some(branch) = deploy {
        Outcome::DeployRequested(branch)
    } else if clean {
        Outcome::CleanExit
    } else {
        Outcome::Crash
    }
}

/// Poll `/health` until it answers `200` within the probation window. Fails if the
/// kernel exits during probation or never reports healthy in time.
fn probation(child: &mut Child, health: &str) -> bool {
    for _ in 0..PROBATION_SECS {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return false;
        }
        if health_ok(health) {
            return true;
        }
        thread::sleep(Duration::from_secs(1));
    }
    false
}

/// A minimal `GET /health` over a raw socket: healthy iff the status line is 200.
fn health_ok(addr: &str) -> bool {
    let Some(socket) = addr.to_socket_addrs().ok().and_then(|mut a| a.next()) else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&socket, Duration::from_secs(2)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request = format!("GET /health HTTP/1.0\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    String::from_utf8_lossy(&response)
        .lines()
        .next()
        .is_some_and(|status_line| status_line.contains(" 200"))
}

/// The supervisor's own verification: check out the proposed branch and run the
/// full gate (build + tests + clippy) here, in the trusted std-only parent, so a
/// malicious or buggy kernel cannot self-certify a binary. Only a `self/improve-*`
/// branch is accepted, and only files on the writable allowlist may change. The
/// build/test/clippy stages run sandboxed (no network, read-only root) because
/// they execute branch-controlled code - build scripts, proc-macros, test bodies.
/// The build leaves the artifact at `target/release/numen`. Returns the failing
/// stage on failure.
fn build_verified(cfg: &Config, branch: &str) -> Result<(), &'static str> {
    if !branch.starts_with("self/improve-") {
        return Err("branch");
    }
    let src = &cfg.src_dir;
    if !run_tool(src, "git", &["checkout", branch]) {
        return Err("checkout");
    }
    // Re-assert the writable allowlist at the trusted boundary: refuse to build a
    // branch that touches any file the kernel may not self-modify, so a compromised
    // kernel cannot smuggle a change to the gate, the supervisor, or a build script
    // (which runs during the gate) through a self/improve-* branch. The list is
    // hardcoded here on purpose - the trusted parent never consults the kernel's.
    match changed_files(src, &cfg.base_branch, branch) {
        Some(files) if files.iter().all(|f| is_writable(f)) => {}
        Some(_) => return Err("allowlist"),
        None => return Err("diff"),
    }
    let features = cfg.build_features.as_deref();
    if !run_gate_tool(
        cfg,
        "cargo",
        &with_features(&["build", "--release", "--bin", "numen"], features),
    ) {
        return Err("build");
    }
    if !run_gate_tool(cfg, "cargo", &with_features(&["test"], features)) {
        return Err("test");
    }
    // Features go before the `--` clippy-args separator.
    let mut clippy = with_features(&["clippy", "--all-targets"], features);
    clippy.extend(["--", "-D", "warnings"]);
    if !run_gate_tool(cfg, "cargo", &clippy) {
        return Err("clippy");
    }
    Ok(())
}

/// Append `--features <features>` to a cargo argument list when the GPU (or any
/// other) feature set is configured, so the gate builds what the image ships.
fn with_features<'a>(base: &[&'a str], features: Option<&'a str>) -> Vec<&'a str> {
    let mut args: Vec<&str> = base.to_vec();
    if let Some(features) = features {
        args.push("--features");
        args.push(features);
    }
    args
}

/// Run a gate stage. When sandboxing is on (the default), the command runs inside
/// bubblewrap with no network and a read-only root - only the checkout and the
/// cargo cache are writable - so the branch's untrusted build scripts, proc-macros
/// and tests cannot exfiltrate, persist outside the build tree, or reach the host
/// while the gate runs. Fail-closed: if `bwrap` is absent or the sandbox cannot
/// start, the spawn fails and the stage is treated as failed, so nothing deploys.
fn run_gate_tool(cfg: &Config, program: &str, args: &[&str]) -> bool {
    if !cfg.sandbox_gate {
        log("WARNING: gate sandbox disabled (NUMEN_GATE_NO_SANDBOX); build runs unconfined");
        return run_tool(&cfg.src_dir, program, args);
    }
    let sandbox = sandbox_args(&cfg.src_dir, &cfg.cargo_home, program, args);
    let refs: Vec<&str> = sandbox.iter().map(String::as_str).collect();
    run_tool(&cfg.src_dir, "bwrap", &refs)
}

/// The bubblewrap arguments (not including `bwrap` itself) that run `program args`
/// network-isolated with a read-only root, the source checkout and cargo cache
/// bound writable, and ephemeral `/tmp`. Pure so the isolation flags are testable.
fn sandbox_args(cwd: &str, cargo_home: &str, program: &str, args: &[&str]) -> Vec<String> {
    let mut argv: Vec<String> = [
        "--unshare-net",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        cwd,
        cwd,
        "--bind",
        cargo_home,
        cargo_home,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        cwd,
        "--",
        program,
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    argv.extend(args.iter().map(|a| (*a).to_string()));
    argv
}

/// Run a subprocess in `cwd`, inheriting stdio so its log streams, and report
/// whether it succeeded.
fn run_tool(cwd: &str, program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The files changed between `base` and `branch`, or `None` if git failed. Paths
/// are git-relative with forward slashes on every platform, so they compare
/// directly against the `WRITABLE` allowlist.
fn changed_files(cwd: &str, base: &str, branch: &str) -> Option<Vec<String>> {
    let output = Command::new("git")
        .args(["diff", "--name-only", &format!("{base}..{branch}")])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(String::from)
            .collect(),
    )
}

/// Back up the running kernel, copy the freshly built binary into its place, and
/// arm the wakeup marker so the new kernel announces itself on boot.
fn swap_to_built(cfg: &Config) -> std::io::Result<()> {
    let built = format!("{}/target/release/numen", cfg.src_dir);
    // Snapshot the DB before the new kernel boots (it may migrate the schema), so a
    // rollback reverts both the binary and the data.
    if Path::new(&cfg.db).exists() {
        fs::copy(&cfg.db, &cfg.db_backup)?;
    }
    fs::copy(&cfg.kernel, &cfg.backup)?;
    fs::copy(&built, &cfg.kernel)?;
    fs::write(&cfg.wakeup, "online")?;
    let _ = fs::remove_file(&cfg.deploy_marker);
    Ok(())
}

/// Restore the backed-up kernel over the current one. Returns whether a backup was
/// available to restore.
fn rollback(cfg: &Config) -> bool {
    if !Path::new(&cfg.backup).exists() {
        return false;
    }
    let restored = fs::rename(&cfg.backup, &cfg.kernel).is_ok();
    // Restore the DB snapshot too, so a migration the failed kernel applied is
    // reverted alongside the binary rather than stranding the old kernel against a
    // newer schema.
    if Path::new(&cfg.db_backup).exists() {
        let _ = fs::copy(&cfg.db_backup, &cfg.db);
    }
    restored
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Whether an environment variable is set to a truthy value (`1` or `true`).
fn env_flag(key: &str) -> bool {
    matches!(env::var(key).ok().as_deref(), Some("1") | Some("true"))
}

fn log(message: &str) {
    eprintln!("[supervisor] {message}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A fresh scratch directory under the system temp dir, unique per test tag so
    /// the parallel test runner does not let two tests collide on the same paths.
    fn scratch(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("numen-sup-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cfg_in(dir: &Path) -> Config {
        let at = |name: &str| dir.join(name).to_string_lossy().into_owned();
        Config {
            kernel: at("numen"),
            src_dir: dir.to_string_lossy().into_owned(),
            base_branch: "main".to_string(),
            backup: at("numen.prev"),
            deploy_marker: at("deploy.request"),
            wakeup: at("wakeup"),
            kill_switch: at("STOP"),
            health: "127.0.0.1:8080".to_string(),
            db: at("numen.db"),
            db_backup: at("numen.db.prev"),
            sandbox_gate: true,
            cargo_home: at(".cargo"),
            build_features: None,
        }
    }

    fn built_binary(cfg: &Config, contents: &str) {
        let release = format!("{}/target/release", cfg.src_dir);
        fs::create_dir_all(&release).unwrap();
        fs::write(format!("{release}/numen"), contents).unwrap();
    }

    #[test]
    fn classify_exit_lets_the_kill_switch_trump_a_pending_deploy() {
        let outcome = classify_exit(true, Some("self/improve-x".to_string()), true);
        assert_eq!(outcome, Outcome::KillSwitch);
    }

    #[test]
    fn classify_exit_prefers_a_deploy_request_over_a_clean_exit() {
        let outcome = classify_exit(false, Some("self/improve-x".to_string()), true);
        assert_eq!(
            outcome,
            Outcome::DeployRequested("self/improve-x".to_string())
        );
    }

    #[test]
    fn classify_exit_maps_a_bare_clean_and_crash_exit() {
        assert_eq!(classify_exit(false, None, true), Outcome::CleanExit);
        assert_eq!(classify_exit(false, None, false), Outcome::Crash);
    }

    #[test]
    fn writable_allowlist_admits_only_the_self_modifiable_files() {
        assert!(is_writable("src/graph.rs"));
        assert!(is_writable("src/aif.rs"));
        // The gate, the supervisor, the manifest, and build scripts are off-limits.
        assert!(!is_writable("src/evolve.rs"));
        assert!(!is_writable("crates/supervisor/src/main.rs"));
        assert!(!is_writable("Cargo.toml"));
        assert!(!is_writable("build.rs"));
        // A changeset passes only if every file is writable.
        let clean = ["src/graph.rs".to_string(), "src/aif.rs".to_string()];
        let tainted = ["src/graph.rs".to_string(), "build.rs".to_string()];
        assert!(clean.iter().all(|f| is_writable(f)));
        assert!(!tainted.iter().all(|f| is_writable(f)));
    }

    #[test]
    fn with_features_appends_only_when_configured() {
        assert_eq!(
            with_features(&["build", "--release"], None),
            ["build", "--release"]
        );
        assert_eq!(
            with_features(&["build", "--release"], Some("cuda")),
            ["build", "--release", "--features", "cuda"]
        );
    }

    #[test]
    fn sandbox_args_isolate_network_and_filesystem_then_run_the_command() {
        let argv = sandbox_args(
            "/app/src",
            "/home/numen/.cargo",
            "cargo",
            &["build", "--release"],
        );
        let joined = argv.join(" ");
        // No network, read-only root.
        assert!(joined.contains("--unshare-net"));
        assert!(joined.contains("--ro-bind / /"));
        // The checkout and the cargo cache are the only writable mounts.
        assert!(joined.contains("--bind /app/src /app/src"));
        assert!(joined.contains("--bind /home/numen/.cargo /home/numen/.cargo"));
        // The wrapped command follows the `--` separator, in order.
        let sep = argv.iter().position(|a| a == "--").unwrap();
        let tail: Vec<String> = argv[sep + 1..].to_vec();
        assert_eq!(tail, ["cargo", "build", "--release"].map(String::from));
    }

    #[test]
    fn build_verified_refuses_a_branch_outside_the_self_improve_namespace() {
        // The gate must reject before it ever checks out or builds, so a missing
        // src_dir is irrelevant: no subprocess should run for these branches.
        let cfg = cfg_in(Path::new("/nonexistent"));
        assert_eq!(build_verified(&cfg, "main"), Err("branch"));
        assert_eq!(build_verified(&cfg, "self/improvement"), Err("branch"));
        assert_eq!(build_verified(&cfg, "../escape"), Err("branch"));
    }

    #[test]
    fn rollback_without_a_backup_reports_no_target_and_keeps_the_kernel() {
        let cfg = cfg_in(&scratch("rollback-none"));
        fs::write(&cfg.kernel, "current").unwrap();
        assert!(!rollback(&cfg));
        assert_eq!(fs::read_to_string(&cfg.kernel).unwrap(), "current");
    }

    #[test]
    fn rollback_restores_the_previous_binary_and_database_snapshot() {
        let cfg = cfg_in(&scratch("rollback-restore"));
        fs::write(&cfg.kernel, "broken-new").unwrap();
        fs::write(&cfg.backup, "good-old").unwrap();
        fs::write(&cfg.db, "migrated").unwrap();
        fs::write(&cfg.db_backup, "pre-migration").unwrap();

        assert!(rollback(&cfg));
        assert_eq!(fs::read_to_string(&cfg.kernel).unwrap(), "good-old");
        assert_eq!(fs::read_to_string(&cfg.db).unwrap(), "pre-migration");
        assert!(!Path::new(&cfg.backup).exists());
    }

    #[test]
    fn swap_to_built_backs_up_then_installs_the_new_binary_and_snapshots_the_db() {
        let cfg = cfg_in(&scratch("swap"));
        built_binary(&cfg, "new-binary");
        fs::write(&cfg.kernel, "old-binary").unwrap();
        fs::write(&cfg.db, "live-data").unwrap();
        fs::write(&cfg.deploy_marker, "self/improve-x").unwrap();

        swap_to_built(&cfg).unwrap();

        assert_eq!(fs::read_to_string(&cfg.kernel).unwrap(), "new-binary");
        assert_eq!(fs::read_to_string(&cfg.backup).unwrap(), "old-binary");
        assert_eq!(fs::read_to_string(&cfg.db_backup).unwrap(), "live-data");
        assert_eq!(fs::read_to_string(&cfg.wakeup).unwrap(), "online");
        assert!(!Path::new(&cfg.deploy_marker).exists());
    }

    #[test]
    fn swap_to_built_skips_the_db_snapshot_when_no_database_exists_yet() {
        let cfg = cfg_in(&scratch("swap-no-db"));
        built_binary(&cfg, "new-binary");
        fs::write(&cfg.kernel, "old-binary").unwrap();

        swap_to_built(&cfg).unwrap();

        assert_eq!(fs::read_to_string(&cfg.kernel).unwrap(), "new-binary");
        assert!(!Path::new(&cfg.db_backup).exists());
    }
}
