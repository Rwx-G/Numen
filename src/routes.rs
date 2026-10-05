//! HTTP surface: `/` (the observability dashboard) + `/events` (its live activity
//! feed), `/health`, `/infer` (delegate + seed), `/learn` (corpus) and
//! `/ingest_feed` (RSS/Atom feeds), `/explain` `/simulate` `/query`
//! `/counterfactual` (causal
//! queries), `/consolidate` (memory maintenance), `/stats` (size + self-metrics),
//! `/critique` (the LLM self-audit into ranked improvement axes), `/identity` (the
//! pinned Identity Subgraph), `/attend` (the Active Inference agenda),
//! `/graph.dot` and `/graph.md` (one-way human-inspection exports of the active
//! graph), and `/act` (take the selected action: a self-directed investigation).
//! An activity
//! middleware stamps the idle clock that drives the consolidation scheduler.

use anyhow::anyhow;

use axum::{
    extract::{DefaultBodyLimit, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::Level;

use crate::{agent, aif, db, extract, graph, metrics, router, scheduler, selfcritique, AppState};

/// Maximum accepted request body. Bounds the memory a single POST can force the
/// server to buffer before any handler runs; `/learn` additionally caps how much
/// of an accepted body it processes.
const MAX_BODY_BYTES: usize = 1 << 20;

/// Assemble the application router with shared state.
pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/events", get(events))
        .route("/health", get(health))
        .route("/infer", post(infer))
        .route("/learn", post(learn))
        .route("/ingest_feed", post(ingest_feed))
        .route("/explain", post(explain))
        .route("/simulate", post(simulate))
        .route("/query", post(query))
        .route("/counterfactual", post(counterfactual))
        .route("/consolidate", post(consolidate))
        .route("/stats", get(stats))
        .route("/critique", post(critique))
        .route("/identity", get(identity))
        .route("/attend", get(attend))
        .route("/introspect", get(introspect))
        .route("/graph.dot", get(graph_dot))
        .route("/graph.md", get(graph_markdown))
        .route("/act", post(act))
        .route("/improve", post(improve))
        .route("/deploy", post(deploy))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            track_activity,
        ))
        // Above track_activity so an unauthorized request neither stamps the idle
        // clock nor reaches a handler, but still inside the trace span.
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        // Outermost: wrap every request in a span so a handler's logs (delegate,
        // extract, seed) correlate, and record method, path, status, and latency.
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .with_state(state)
}

/// Stamp the last-activity clock on every request except read-only observation
/// endpoints, so neither the Docker healthcheck nor a watching dashboard (which
/// polls those endpoints) keeps the agent perpetually "busy" and blocks idle-time
/// consolidation and proactive work.
async fn track_activity(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if !is_observation_path(request.uri().path()) {
        state.touch_activity();
    }
    next.run(request).await
}

/// Read-only observation endpoints the dashboard polls: they neither count as
/// interactive work (no activity stamp) nor mutate state.
fn is_observation_path(path: &str) -> bool {
    matches!(
        path,
        "/health"
            | "/"
            | "/events"
            | "/stats"
            | "/attend"
            | "/introspect"
            | "/identity"
            | "/graph.dot"
            | "/graph.md"
    )
}

/// Bearer-token gate. With `NUMEN_API_TOKEN` set, every request except the health
/// probe must present a matching `Authorization: Bearer <token>` (compared in
/// constant time). Unset means an open API, for localhost development.
async fn require_auth(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let Some(expected) = state.api_token.as_deref() else {
        return next.run(request).await;
    };
    // The health probe and the dashboard shell are always reachable; the dashboard's
    // data calls (/events, /stats, ...) still carry the token.
    if matches!(request.uri().path(), "/health" | "/") {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if presented.is_some_and(|token| constant_time_eq(token.as_bytes(), expected.as_bytes())) {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

/// Length-checked constant-time byte comparison, so token verification does not
/// leak the token through response timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
struct InferRequest {
    query: String,
}

#[derive(Serialize)]
struct InferResponse {
    answer: String,
    source: String,
    confidence: f64,
    delegated: bool,
    learned: usize,
    /// The causal relations Numen actually added this turn - what it "found".
    relations: Vec<graph::Relation>,
    conflicts: Vec<graph::Conflict>,
}

async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, AppError> {
    let outcome = agent::investigate(&state, &req.query).await?;
    Ok(Json(InferResponse {
        answer: outcome.prose,
        source: outcome.source.to_string(),
        confidence: outcome.confidence,
        delegated: outcome.source == "claude",
        learned: outcome.learned,
        relations: outcome.relations,
        conflicts: outcome.conflicts,
    }))
}

#[derive(Deserialize)]
struct LearnRequest {
    text: String,
}

#[derive(Serialize)]
struct LearnResponse {
    chunks: usize,
    learned: usize,
    conflicts: usize,
    truncated: bool,
}

/// Ingest a corpus passage through the shared learn pipeline.
async fn learn(
    State(state): State<AppState>,
    Json(req): Json<LearnRequest>,
) -> Result<Json<LearnResponse>, AppError> {
    let summary = agent::learn_text(&state, &req.text).await?;
    Ok(Json(LearnResponse {
        chunks: summary.chunks,
        learned: summary.learned,
        conflicts: summary.conflicts,
        truncated: summary.truncated,
    }))
}

#[derive(Deserialize)]
struct IngestFeedRequest {
    url: String,
}

#[derive(Serialize)]
struct IngestFeedResponse {
    entries: usize,
    new_entries: usize,
    learned: usize,
    conflicts: usize,
    truncated: bool,
}

/// World Ingestion Layer: fetch an RSS or Atom feed and learn from its new
/// entries, dedup'd by entry id so re-posting the same feed only ingests what
/// has appeared since.
async fn ingest_feed(
    State(state): State<AppState>,
    Json(req): Json<IngestFeedRequest>,
) -> Result<Json<IngestFeedResponse>, AppError> {
    let report = agent::ingest_one_feed(&state, &req.url).await?;
    Ok(Json(IngestFeedResponse {
        entries: report.entries,
        new_entries: report.new_entries,
        learned: report.learned,
        conflicts: report.conflicts,
        truncated: report.truncated,
    }))
}

#[derive(Deserialize)]
struct ExplainRequest {
    from: String,
    to: String,
}

#[derive(Serialize)]
struct ExplainResponse {
    from: String,
    to: String,
    found: bool,
    path: Vec<graph::PathStep>,
}

/// Strength added to each edge a query traverses (Hebbian reinforcement).
const HEBBIAN_DELTA: f64 = 0.05;

/// Reinforce the edges a query traversed, persist it, and refresh the in-memory
/// graph. No-op for an empty path.
async fn reinforce_path(state: &AppState, steps: &[graph::PathStep]) -> Result<(), AppError> {
    if steps.is_empty() {
        return Ok(());
    }
    let edges: Vec<(&str, &str, &str)> = steps
        .iter()
        .map(|step| (step.from.as_str(), step.relation.as_str(), step.to.as_str()))
        .collect();
    // Update store and memory atomically under the graph write lock (lock order:
    // graph, then db).
    let mut graph = state.graph_write();
    let mut conn = state.db();
    db::reinforce(&mut conn, &edges, HEBBIAN_DELTA)?;
    graph.reinforce(&edges, HEBBIAN_DELTA);
    Ok(())
}

/// Return the shortest directed causal path between two concepts. Inputs are
/// canonicalized the same way stored labels are, so `Data Races` matches
/// `data race`. Traversed edges are reinforced (Hebbian).
async fn explain(
    State(state): State<AppState>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<ExplainResponse>, AppError> {
    let from = extract::canonical_label(&req.from);
    let to = extract::canonical_label(&req.to);

    let path = {
        let graph = state.graph_read();
        graph.explain(&from, &to)
    };

    if let Some(steps) = &path {
        reinforce_path(&state, steps).await?;
    }

    Ok(Json(ExplainResponse {
        from,
        to,
        found: path.is_some(),
        path: path.unwrap_or_default(),
    }))
}

#[derive(Deserialize)]
struct SimulateRequest {
    intervention: String,
}

#[derive(Serialize)]
struct SimulateResponse {
    intervention: String,
    found: bool,
    severed_parents: Vec<String>,
    affected: Vec<graph::Effect>,
}

/// Run `do(X)` on the causal graph: report the parents detached by the
/// intervention and the descendants it propagates to, ranked by influence.
async fn simulate(
    State(state): State<AppState>,
    Json(req): Json<SimulateRequest>,
) -> Result<Json<SimulateResponse>, AppError> {
    let target = extract::canonical_label(&req.intervention);

    let result = {
        let graph = state.graph_read();
        graph.intervene(&target)
    };

    Ok(Json(match result {
        Some(intervention) => SimulateResponse {
            intervention: intervention.target,
            found: true,
            severed_parents: intervention.severed_parents,
            affected: intervention.affected,
        },
        None => SimulateResponse {
            intervention: target,
            found: false,
            severed_parents: Vec::new(),
            affected: Vec::new(),
        },
    }))
}

#[derive(Deserialize)]
struct QueryRequest {
    cause: String,
    effect: String,
}

#[derive(Serialize)]
struct QueryResponse {
    found: bool,
    #[serde(flatten)]
    query: Option<graph::CausalQuery>,
}

/// Distinguish causation from confounded correlation between two concepts.
async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    let cause = extract::canonical_label(&req.cause);
    let effect = extract::canonical_label(&req.effect);

    let result = {
        let graph = state.graph_read();
        graph.query(&cause, &effect)
    };

    if let Some(causal) = &result {
        reinforce_path(&state, &causal.causal_path).await?;
    }

    Ok(Json(QueryResponse {
        found: result.is_some(),
        query: result,
    }))
}

#[derive(Deserialize)]
struct CounterfactualRequest {
    cause: String,
    effect: String,
}

#[derive(Serialize)]
struct CounterfactualResponse {
    found: bool,
    #[serde(flatten)]
    counterfactual: Option<graph::Counterfactual>,
}

/// Structural but-for counterfactual: would removing the cause leave the effect
/// without a root cause? Reports necessary vs contributing causation.
async fn counterfactual(
    State(state): State<AppState>,
    Json(req): Json<CounterfactualRequest>,
) -> Result<Json<CounterfactualResponse>, AppError> {
    let cause = extract::canonical_label(&req.cause);
    let effect = extract::canonical_label(&req.effect);

    let result = {
        let graph = state.graph_read();
        graph.counterfactual(&cause, &effect)
    };

    Ok(Json(CounterfactualResponse {
        found: result.is_some(),
        counterfactual: result,
    }))
}

/// Run a consolidation pass on demand. The same pass runs automatically on the
/// idle-aware background scheduler.
async fn consolidate(
    State(state): State<AppState>,
) -> Result<Json<db::ConsolidationStats>, AppError> {
    let stats = tokio::task::spawn_blocking(move || scheduler::consolidate_once(&state))
        .await
        .map_err(|err| anyhow!("consolidation task panicked: {err}"))??;
    Ok(Json(stats))
}

#[derive(Serialize)]
struct StatsResponse {
    nodes: usize,
    edges: usize,
    episodes: i64,
    edges_by_source: std::collections::BTreeMap<String, i64>,
    top_concepts: Vec<graph::ConceptStat>,
    metrics: metrics::Snapshot,
}

/// How many of the most-connected concepts `/stats` returns.
const STATS_TOP_CONCEPTS: usize = 15;

/// Snapshot of the graph's size, edge provenance, and most connected concepts.
async fn stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, AppError> {
    let (nodes, edges, top_concepts) = {
        let graph = state.graph_read();
        (
            graph.node_count(),
            graph.edge_count(),
            graph.top_concepts(STATS_TOP_CONCEPTS),
        )
    };
    let store = {
        let conn = state.db();
        db::store_stats(&conn)?
    };

    Ok(Json(StatsResponse {
        nodes,
        edges,
        episodes: store.episodes,
        edges_by_source: store.edges_by_source,
        top_concepts,
        metrics: state.metrics.snapshot(),
    }))
}

/// Numen's self-critique: it reads its own metrics and asks the LLM to rank the
/// axes along which it should improve its own source. Read-only and advisory - it
/// proposes, it does not act. A POST because it costs a Claude delegation.
async fn critique(State(state): State<AppState>) -> Result<Json<selfcritique::Critique>, AppError> {
    Ok(Json(selfcritique::critique(&state).await?))
}

/// The Identity Subgraph: the pinned self, values, operator, and the
/// corrigibility edge. Stable across sessions, never decayed or pruned.
async fn identity(State(state): State<AppState>) -> Json<graph::Identity> {
    Json(state.graph_read().identity())
}

/// How many foci `/attend` returns.
const ATTEND_LIMIT: usize = 10;

/// The Active Inference attention agenda: what the agent should explore or
/// consolidate next, ranked by structural Expected Free Energy, with the
/// explore/exploit balance set by the identity values.
async fn attend(State(state): State<AppState>) -> Json<aif::Attention> {
    Json(aif::attend(&state.graph_read(), ATTEND_LIMIT))
}

#[derive(Deserialize)]
struct IntrospectQuery {
    concept: String,
}

/// What Numen "thinks" about a concept: its local causal context (the direct
/// causes and effects it has learned), how certain it is, its uncertainty (the
/// exploration pull), and how it would reason about it - answer locally or
/// delegate. Read-only and free (no LLM): the agent's mind, not an action.
#[derive(Serialize)]
struct ConceptView {
    concept: String,
    known: bool,
    confidence: f64,
    /// `1 - confidence`: the epistemic gap that makes a concept a target.
    epistemic: f64,
    degree: usize,
    /// How an investigation of this concept would route now: `local` or `delegate`.
    route: &'static str,
    causes: Vec<graph::PathStep>,
    effects: Vec<graph::PathStep>,
}

async fn introspect(
    State(state): State<AppState>,
    Query(query): Query<IntrospectQuery>,
) -> Json<ConceptView> {
    let concept = extract::canonical_label(&query.concept);
    let graph = state.graph_read();
    let routing_query = format!("causes and effects of {concept}");
    let route = match router::route(
        graph.domain_confidence(&routing_query),
        state.backend.has_local(),
    ) {
        router::Route::Local => "local",
        router::Route::Delegate => "delegate",
    };
    let view = match graph.neighbourhood(&concept) {
        Some(n) => ConceptView {
            concept: n.concept,
            known: true,
            confidence: n.confidence,
            epistemic: 1.0 - n.confidence,
            degree: n.degree,
            route,
            causes: n.causes,
            effects: n.effects,
        },
        None => ConceptView {
            concept,
            known: false,
            confidence: 0.0,
            epistemic: 1.0,
            degree: 0,
            route,
            causes: Vec::new(),
            effects: Vec::new(),
        },
    };
    Json(view)
}

/// The active reasoning graph as Graphviz DOT, for human inspection. One-way:
/// render it with `dot`, nothing is read back in.
async fn graph_dot(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/vnd.graphviz")],
        state.graph_read().to_dot(),
    )
}

/// The active reasoning graph as a Markdown edge-list, for human reading.
async fn graph_markdown(State(state): State<AppState>) -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/markdown")],
        state.graph_read().to_markdown(),
    )
}

/// The observability dashboard: a self-contained page (no external assets) that
/// polls the read-only endpoints to show, live, what Numen knows, what it wants to
/// explore next, what it is doing, and an explorer for its causal reasoning.
async fn dashboard() -> Html<String> {
    // Serve the file at NUMEN_DASHBOARD when set and readable, so the UI can be
    // live-edited (mount it, restart) without a rebuild; otherwise the copy
    // embedded at build time, so the image always works standalone.
    let html = std::env::var("NUMEN_DASHBOARD")
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .unwrap_or_else(|| include_str!("dashboard.html").to_string());
    Html(html)
}

/// The recent activity feed, newest first - the live record of what the agent did
/// (inferences, proactive explorations, consolidations, self-improvements).
async fn events(State(state): State<AppState>) -> Json<Vec<crate::Event>> {
    Json(state.recent_events())
}

#[derive(Serialize)]
struct ActResponse {
    action: &'static str,
    concept: Option<String>,
    source: Option<String>,
    learned: usize,
}

/// Take the agent's next Active Inference action: explore the biggest knowledge
/// gap (a self-directed investigation that learns) or report idle. The action is
/// selected by structural EFE; executing `Explore` runs the same investigate
/// pipeline as `/infer` on the chosen concept.
///
/// This is operator-invoked, so it runs on operator authority and is not subject
/// to the autonomy gate (which governs self-initiated actions in the proactive
/// loop). The planner only ever emits `Explore`/`Idle` today. SECURITY: once the
/// ladder can emit a rung above `Explore`, this handler must route the action
/// through `aif::gate` (or refuse rungs above `Explore`) before executing it.
async fn act(State(state): State<AppState>) -> Result<Json<ActResponse>, AppError> {
    let action = aif::select_action(&state.graph_read());
    match action {
        aif::Action::Explore { concept } => {
            let query = format!("causes and effects of {concept}");
            let outcome = agent::investigate(&state, &query).await?;
            Ok(Json(ActResponse {
                action: "explore",
                concept: Some(concept),
                source: Some(outcome.source.to_string()),
                learned: outcome.learned,
            }))
        }
        aif::Action::Idle => Ok(Json(ActResponse {
            action: "idle",
            concept: None,
            source: None,
            learned: 0,
        })),
    }
}

#[derive(Deserialize)]
struct ImproveRequest {
    goal: String,
    file: String,
}

#[derive(Serialize)]
struct ImproveResponse {
    file: String,
    verified: bool,
    branch: String,
    detail: String,
}

/// Self-improvement: Numen rewrites one of its own source files toward a goal and
/// runs the deterministic gate (build + tests + clippy). A verified change lands
/// on a branch for human review and deploy; an unverified one is discarded. Numen
/// never deploys on its own - this only produces a vetted candidate.
async fn improve(
    State(state): State<AppState>,
    Json(req): Json<ImproveRequest>,
) -> Result<Json<ImproveResponse>, AppError> {
    let report = agent::improve(&state, &req.goal, &req.file).await?;
    Ok(Json(ImproveResponse {
        file: report.file,
        verified: report.verified,
        branch: report.branch,
        detail: report.detail,
    }))
}

#[derive(Deserialize)]
struct DeployRequest {
    branch: String,
}

#[derive(Serialize)]
struct DeployResponse {
    staged: bool,
}

/// Deploy a verified self-improvement branch (human-gated): merge it, build the
/// release binary, stage it for the supervisor, and gracefully shut down so the
/// supervisor swaps in the new kernel (rolling back if it does not come up
/// healthy). The response is sent before the handoff drains the server.
async fn deploy(
    State(state): State<AppState>,
    Json(req): Json<DeployRequest>,
) -> Result<Json<DeployResponse>, AppError> {
    agent::deploy(&state, &req.branch).await?;
    Ok(Json(DeployResponse { staged: true }))
}

/// Request error mapped to an HTTP status, without leaking internals to the
/// client beyond a short message. `Upstream` is a failure of the LLM backend
/// (the `claude` subprocess or local inference), surfaced as `502` so a caller
/// can tell a model/backend outage from a genuine internal fault; everything
/// else is `Internal` (`500`).
enum AppError {
    Internal(anyhow::Error),
    Upstream(anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message, error) = match self {
            Self::Internal(error) => (StatusCode::INTERNAL_SERVER_ERROR, "internal error", error),
            Self::Upstream(error) => (StatusCode::BAD_GATEWAY, "upstream model error", error),
        };
        tracing::error!(%status, error = %error, "request failed");
        (status, message).into_response()
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Internal(error) | Self::Upstream(error) => error.fmt(f),
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        Self::Internal(err.into())
    }
}

impl From<agent::Failure> for AppError {
    fn from(failure: agent::Failure) -> Self {
        match failure {
            agent::Failure::Upstream(error) => Self::Upstream(error),
            agent::Failure::Internal(error) => Self::Internal(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, RwLock};
    use std::time::Instant;

    use axum::body::Body;
    use tokio::sync::Semaphore;
    use tower::ServiceExt;

    use super::*;
    use crate::llm;

    /// A minimal app state for handler tests: an in-memory DB with the identity
    /// seeded, a Claude-only backend, and an optional bearer token.
    fn test_state(api_token: Option<&str>) -> AppState {
        let mut conn = db::open(":memory:").unwrap();
        db::seed_identity(&mut conn).unwrap();
        let causal_graph = graph::CausalGraph::load(&conn).unwrap();
        AppState {
            db: Arc::new(Mutex::new(conn)),
            graph: Arc::new(RwLock::new(causal_graph)),
            backend: Arc::new(llm::Backend::cloud_only()),
            embedder: Arc::new(None),
            workshop: Arc::new(tokio::sync::Mutex::new(None)),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            llm_permits: Arc::new(Semaphore::new(1)),
            api_token: api_token.map(Arc::from),
            effector: None,
            http: reqwest::Client::new(),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            metrics: Arc::new(crate::metrics::Metrics::default()),
            events: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        }
    }

    fn get(uri: &str, token: Option<&str>) -> Request {
        let mut builder = Request::builder().uri(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder.body(Body::empty()).unwrap()
    }

    #[test]
    fn constant_time_eq_matches_only_identical_bytes() {
        assert!(constant_time_eq(b"secret-token", b"secret-token"));
        assert!(constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"secret-token", b"secret-toked"));
        assert!(!constant_time_eq(b"secret", b"secret-token"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    #[tokio::test]
    async fn health_is_open_even_with_auth_enabled() {
        let app = build(test_state(Some("secret")));
        let response = app.oneshot(get("/health", None)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_gate_rejects_missing_or_wrong_token_and_accepts_a_valid_one() {
        let app = build(test_state(Some("secret")));
        let missing = app.clone().oneshot(get("/identity", None)).await.unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        let wrong = app
            .clone()
            .oneshot(get("/identity", Some("nope")))
            .await
            .unwrap();
        assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
        let valid = app.oneshot(get("/identity", Some("secret"))).await.unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn open_api_serves_the_read_endpoints() {
        let app = build(test_state(None));
        for uri in [
            "/",
            "/events",
            "/identity",
            "/attend",
            "/stats",
            "/graph.dot",
            "/graph.md",
            "/health",
        ] {
            let response = app.clone().oneshot(get(uri, None)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{uri}");
        }
    }

    #[tokio::test]
    async fn dashboard_shell_loads_without_a_token_but_its_data_stays_gated() {
        let app = build(test_state(Some("secret")));
        // The shell is reachable so the page can load and prompt for the token.
        let shell = app.clone().oneshot(get("/", None)).await.unwrap();
        assert_eq!(shell.status(), StatusCode::OK);
        assert!(body_text(shell).await.contains("NUMEN"));
        // Its data feed still requires the token.
        let feed = app.oneshot(get("/events", None)).await.unwrap();
        assert_eq!(feed.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn introspect_returns_a_concept_view() {
        let app = build(test_state(None));
        // The seeded identity has 'numen'; its neighbourhood is the value edges.
        let response = app
            .oneshot(get("/introspect?concept=numen", None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_text(response).await;
        assert!(body.contains("\"concept\":\"numen\""));
        assert!(body.contains("\"known\":true"));
    }

    #[tokio::test]
    async fn critique_is_gated_by_the_token() {
        // /critique costs a Claude delegation, so it must sit behind auth. A POST
        // without the token is refused before the handler (no LLM call in the test).
        let app = build(test_state(Some("secret")));
        let request = Request::builder()
            .method("POST")
            .uri("/critique")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    async fn body_text(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn graph_exports_render_dot_and_markdown_with_their_content_types() {
        let app = build(test_state(None));

        let dot = app.clone().oneshot(get("/graph.dot", None)).await.unwrap();
        assert_eq!(dot.status(), StatusCode::OK);
        assert_eq!(
            dot.headers().get("content-type").unwrap(),
            "text/vnd.graphviz"
        );
        assert!(body_text(dot).await.starts_with("digraph numen {"));

        let md = app.oneshot(get("/graph.md", None)).await.unwrap();
        assert_eq!(md.status(), StatusCode::OK);
        assert_eq!(md.headers().get("content-type").unwrap(), "text/markdown");
        assert!(body_text(md).await.contains("# Numen causal graph"));
    }

    #[tokio::test]
    async fn auth_gate_also_covers_the_graph_exports() {
        let app = build(test_state(Some("secret")));
        for uri in ["/graph.dot", "/graph.md"] {
            let denied = app.clone().oneshot(get(uri, None)).await.unwrap();
            assert_eq!(denied.status(), StatusCode::UNAUTHORIZED, "{uri}");
        }
    }
}
