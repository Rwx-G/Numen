#![deny(unsafe_code)]
#![warn(clippy::all)]

mod agent;
mod aif;
mod db;
mod effector;
mod embed;
mod evolve;
mod extract;
mod graph;
mod ingest;
mod llm;
mod metrics;
mod router;
mod routes;
mod scheduler;
mod selfcritique;
mod selfimprove;

use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::Semaphore;
use tracing_subscriber::EnvFilter;

/// How many recent activity events the dashboard feed keeps in memory.
const MAX_EVENTS: usize = 200;

/// A timestamped record of something Numen did - an inference, a proactive
/// exploration, a consolidation, a self-improvement step. The live activity feed
/// these form is what makes the otherwise black-box autonomous behaviour
/// observable over HTTP.
#[derive(Clone, serde::Serialize)]
pub struct Event {
    /// Unix seconds.
    pub ts: u64,
    /// A short category (`infer`, `explore`, `ingest`, `consolidate`, `improve`).
    pub kind: &'static str,
    /// A one-line human-readable description.
    pub detail: String,
}

/// Shared application state: the SQLite connection, the in-memory causal graph,
/// the LLM backend, a permit pool bounding concurrent (expensive) LLM calls, and
/// the timestamp of the last interactive request (drives the idle-aware
/// consolidation scheduler).
#[derive(Clone)]
pub struct AppState {
    db: Arc<Mutex<rusqlite::Connection>>,
    graph: Arc<RwLock<graph::CausalGraph>>,
    backend: Arc<llm::Backend>,
    /// Local embedding model for vector memory. `None` when no model is mounted,
    /// leaving the graph recalled by exact labels only.
    embedder: Arc<Option<embed::Embedder>>,
    /// Numen's own source tree for self-improvement, behind a lock that serializes
    /// attempts. `None` inside when no `NUMEN_SRC_DIR` / toolchain is available.
    workshop: Arc<tokio::sync::Mutex<Option<evolve::Workshop>>>,
    /// Fired to request a graceful shutdown (a self-deploy handoff to the
    /// supervisor); the server's shutdown future also waits on it.
    shutdown: Arc<tokio::sync::Notify>,
    llm_permits: Arc<Semaphore>,
    /// When set (`NUMEN_API_TOKEN`), every request except `/health` must carry a
    /// matching bearer token. Unset means an open API (localhost dev).
    api_token: Option<Arc<str>>,
    /// Telegram push target for proactive notifications. `None` when unconfigured.
    effector: Option<Arc<effector::Telegram>>,
    /// Shared HTTP client for the World Ingestion Layer (feed fetching).
    http: reqwest::Client,
    last_activity: Arc<Mutex<Instant>>,
    /// Self-metrics feeding the self-critique (in RAM since process start).
    metrics: Arc<metrics::Metrics>,
    /// Bounded ring of recent activity events for the dashboard's live feed.
    events: Arc<Mutex<std::collections::VecDeque<Event>>>,
}

impl AppState {
    /// Lock helpers recover from poison rather than propagating: a panic while a
    /// lock was held leaves graph state the loader can rebuild and DB state SQLite
    /// keeps consistent, so a single transient fault must not brick every endpoint.
    fn graph_read(&self) -> RwLockReadGuard<'_, graph::CausalGraph> {
        self.graph
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn graph_write(&self) -> RwLockWriteGuard<'_, graph::CausalGraph> {
        self.graph
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn db(&self) -> MutexGuard<'_, rusqlite::Connection> {
        self.db.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    fn touch_activity(&self) {
        *self
            .last_activity
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        self.last_activity
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .elapsed()
    }

    /// Push an event onto the bounded activity feed, evicting the oldest once full.
    fn record_event(&self, kind: &'static str, detail: String) {
        let mut events = self
            .events
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if events.len() >= MAX_EVENTS {
            events.pop_front();
        }
        events.push_back(Event {
            ts: agent::unix_now(),
            kind,
            detail,
        });
    }

    /// The recent activity events, newest first, for the dashboard feed.
    fn recent_events(&self) -> Vec<Event> {
        let events = self
            .events
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        events.iter().rev().cloned().collect()
    }

    /// Acquire a permit before an LLM call, capping how many expensive
    /// generations (a `claude -p` subprocess or a GPU/CPU forward) run at once so
    /// a flood of requests cannot fork-bomb or saturate the device. The semaphore
    /// is never closed, so acquisition only fails on an unreachable internal error.
    async fn llm_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        Ok(self.llm_permits.clone().acquire_owned().await?)
    }
}

/// Read a `u64` seconds value from the environment, falling back to `default`.
fn env_secs(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let db_path = std::env::var("NUMEN_DB_PATH").unwrap_or_else(|_| "/data/numen.db".to_string());
    let mut conn = db::open(&db_path)?;
    db::seed_identity(&mut conn)?;
    let metrics_counters = db::load_metrics(&conn)?;
    let causal_graph = graph::CausalGraph::load(&conn)?;
    tracing::info!(
        db = %db_path,
        nodes = causal_graph.node_count(),
        "memory loaded"
    );

    let model_path = std::env::var("NUMEN_MODEL_PATH").ok();
    let backend = llm::Backend::load(model_path.as_deref()).await?;

    let embed_dir = std::env::var("NUMEN_EMBED_MODEL_DIR").ok();
    let embedder = embed::Embedder::load(embed_dir.as_deref())?;
    tracing::info!(configured = embedder.is_some(), "embedder");

    let workshop = evolve::Workshop::from_env();
    tracing::info!(enabled = workshop.is_some(), "self-improvement workshop");

    let max_llm = std::env::var("NUMEN_MAX_CONCURRENT_LLM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4usize)
        .max(1);
    let api_token: Option<Arc<str>> = std::env::var("NUMEN_API_TOKEN").ok().map(Arc::from);
    tracing::info!(enabled = api_token.is_some(), "api authentication");
    let http = reqwest::Client::builder()
        .user_agent(concat!("numen/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let effector = effector::Telegram::from_env(http.clone()).map(Arc::new);
    tracing::info!(configured = effector.is_some(), "telegram effector");
    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
        graph: Arc::new(RwLock::new(causal_graph)),
        backend: Arc::new(backend),
        embedder: Arc::new(embedder),
        workshop: Arc::new(tokio::sync::Mutex::new(workshop)),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        llm_permits: Arc::new(Semaphore::new(max_llm)),
        api_token,
        effector,
        http,
        last_activity: Arc::new(Mutex::new(Instant::now())),
        metrics: Arc::new(metrics::Metrics::from_counters(metrics_counters)),
        events: Arc::new(Mutex::new(std::collections::VecDeque::with_capacity(
            MAX_EVENTS,
        ))),
    };

    // Backfill embeddings for any node that lacks one (a one-time pass after the
    // model first appears). A failure here is logged, not fatal: the API still
    // serves, just without semantic recall.
    if let Some(embedder) = state.embedder.as_ref() {
        let mut conn = state.db();
        match embed::backfill(embedder, &mut conn) {
            Ok(n) => tracing::info!(embedded = n, "node embeddings backfilled"),
            Err(error) => tracing::warn!(%error, "embedding backfill failed"),
        }
    }

    // If the supervisor just swapped us in (a wakeup marker is present), announce
    // the new kernel over the effector and clear the marker so it fires once.
    let wakeup = std::env::var("NUMEN_WAKEUP_MARKER").unwrap_or_else(|_| "/app/wakeup".to_string());
    let booted_from_deploy = std::path::Path::new(&wakeup).exists();
    if booted_from_deploy {
        if let Some(effector) = &state.effector {
            let message = concat!("Numen v", env!("CARGO_PKG_VERSION"), " online - I'm here.");
            let _ = effector.send(message).await;
        }
        let _ = std::fs::remove_file(&wakeup);
        tracing::info!("woke up after a self-deploy");
    }

    let tick = Duration::from_secs(env_secs("NUMEN_CONSOLIDATE_TICK_SECS", 300));
    let idle = Duration::from_secs(env_secs("NUMEN_IDLE_SECS", 60));
    // Off by default: proactive exploration delegates to the LLM on a timer, so
    // it is opt-in to avoid burning the Claude quota unasked.
    let proactive = std::env::var("NUMEN_PROACTIVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Off by default: when on, the agent may rewrite and redeploy its own source
    // once exploration is exhausted (rate-limited, cooldown-gated, sandboxed gate).
    let autonomous = std::env::var("NUMEN_AUTONOMOUS_IMPROVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    // Feeds the scheduler fetches on each idle tick, dedup'd by entry id.
    // Comma-, space-, or newline-separated; empty disables scheduled ingestion.
    let feeds: Vec<String> = std::env::var("NUMEN_FEEDS")
        .unwrap_or_default()
        .split([',', '\n', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let feed_count = feeds.len();
    tokio::spawn(scheduler::run(
        state.clone(),
        tick,
        idle,
        proactive,
        autonomous,
        booted_from_deploy,
        feeds,
    ));
    tracing::info!(
        tick_secs = tick.as_secs(),
        idle_secs = idle.as_secs(),
        feeds = feed_count,
        "consolidation scheduler started"
    );

    // Default to loopback. The container sets NUMEN_BIND=0.0.0.0 explicitly,
    // reachable only behind the published port and the bearer-token gate.
    let addr = std::env::var("NUMEN_BIND").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "Numen listening");
    let deploy = state.shutdown.clone();
    axum::serve(listener, routes::build(state))
        .with_graceful_shutdown(shutdown_signal(deploy))
        .await?;
    tracing::info!("Numen shut down gracefully");
    Ok(())
}

/// Resolve when the process is asked to stop (Ctrl-C, or SIGTERM from the
/// container or the supervisor), so the server drains in-flight requests before a
/// restart instead of being killed mid-response.
async fn shutdown_signal(deploy: Arc<tokio::sync::Notify>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(%error, "failed to install the SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    let deploy = async move {
        deploy.notified().await;
    };

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
        () = deploy => tracing::info!("self-deploy requested"),
    }
    tracing::info!("shutdown signal received, draining");
}
