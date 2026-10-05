use std::collections::BTreeMap;

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;

use crate::graph::Relation;

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
    id         INTEGER PRIMARY KEY,
    label      TEXT NOT NULL UNIQUE,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    id                  INTEGER PRIMARY KEY,
    src                 INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst                 INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    strength            REAL NOT NULL DEFAULT 0.0,
    confidence_interval REAL NOT NULL DEFAULT 0.0,
    valence             REAL NOT NULL DEFAULT 0.0,
    type                TEXT NOT NULL DEFAULT 'semantic',
    source              TEXT NOT NULL DEFAULT 'local',
    last_accessed       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_src_dst_type ON edges(src, dst, type);

CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY,
    query      TEXT NOT NULL,
    answer     TEXT NOT NULL,
    source     TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
";

/// Open (or create) the database at `path` and apply the schema idempotently.
/// Pass `":memory:"` for an ephemeral in-memory database.
pub fn open(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch(SCHEMA)?;
    migrate(&conn)?;
    Ok(conn)
}

/// Forward-only schema migrations tracked by `PRAGMA user_version`. v1 adds the
/// `pinned` flag to nodes and edges (the Identity Subgraph); v2 adds the per-node
/// `embedding` BLOB for vector memory; v3 adds the `ingested_entries` table that
/// dedups scheduled feed fetches; v4 adds the edge `status` column so a
/// contradiction loser is kept as a `superseded` audit record instead of being
/// deleted outright; v5 adds the single-row `metrics` table so the self-metrics
/// survive a self-deploy restart; v6 gives the operator its generic identity
/// labels (see `rename_operator_identity`). Fresh databases take the same path
/// from 0.
fn migrate(conn: &Connection) -> Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version < 1 {
        conn.execute_batch(
            "ALTER TABLE nodes ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
             ALTER TABLE edges ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
             PRAGMA user_version = 1;",
        )?;
    }
    if version < 2 {
        conn.execute_batch(
            "ALTER TABLE nodes ADD COLUMN embedding BLOB;
             PRAGMA user_version = 2;",
        )?;
    }
    if version < 3 {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS ingested_entries (
                 entry_id    TEXT PRIMARY KEY,
                 ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
             );
             PRAGMA user_version = 3;",
        )?;
    }
    if version < 4 {
        conn.execute_batch(
            "ALTER TABLE edges ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
             PRAGMA user_version = 4;",
        )?;
    }
    if version < 5 {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS metrics (
                 id                   INTEGER PRIMARY KEY CHECK (id = 1),
                 local_answers        INTEGER NOT NULL DEFAULT 0,
                 delegations          INTEGER NOT NULL DEFAULT 0,
                 investigate_nanos    INTEGER NOT NULL DEFAULT 0,
                 proactive_grounded   INTEGER NOT NULL DEFAULT 0,
                 proactive_ungrounded INTEGER NOT NULL DEFAULT 0,
                 consolidations       INTEGER NOT NULL DEFAULT 0,
                 edges_pruned         INTEGER NOT NULL DEFAULT 0
             );
             INSERT OR IGNORE INTO metrics (id) VALUES (1);
             PRAGMA user_version = 5;",
        )?;
    }
    if version < 6 {
        rename_operator_identity(conn)?;
        conn.execute_batch("PRAGMA user_version = 6;")?;
    }
    Ok(())
}

/// Relabel an Identity Subgraph seeded under an operator's personal name. The
/// old node is found by structure, not by name: it is the pinned source of the
/// pinned corrigibility edge into the self node, and the loyalty value is the
/// pinned `loyalty to ...` node. Relabelling keeps ids, so every edge follows.
/// A plain UPDATE fails loudly if the new label is already taken: two nodes
/// claiming the operator's authority must stop startup, not be ignored.
fn rename_operator_identity(conn: &Connection) -> Result<()> {
    conn.execute(
        "UPDATE nodes SET label = ?1
          WHERE pinned = 1 AND label <> ?1 AND id IN (
                SELECT e.src FROM edges e JOIN nodes d ON d.id = e.dst
                 WHERE e.pinned = 1 AND e.type = ?2 AND d.label = ?3)",
        params![IDENTITY_AUTHORITY, CORRIGIBILITY_KIND, IDENTITY_SELF],
    )?;
    conn.execute(
        "UPDATE nodes SET label = ?1
          WHERE pinned = 1 AND label LIKE 'loyalty to %' AND label <> ?1",
        params![IDENTITY_LOYALTY],
    )?;
    Ok(())
}

/// Whether a feed entry has already been ingested, keyed by its stable feed id.
pub fn entry_seen(conn: &Connection, entry_id: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM ingested_entries WHERE entry_id = ?1",
        params![entry_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

/// Record a feed entry as ingested so scheduled re-fetches skip it. Idempotent.
pub fn mark_entry_seen(conn: &Connection, entry_id: &str) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO ingested_entries (entry_id) VALUES (?1)",
        params![entry_id],
    )?;
    Ok(())
}

/// Serialize an embedding to little-endian `f32` bytes for BLOB storage.
fn embedding_bytes(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Labels of nodes that have no embedding yet: the work list for backfill.
pub fn unembedded_labels(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT label FROM nodes WHERE embedding IS NULL")?;
    let labels = stmt
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    Ok(labels)
}

/// Store a node's embedding, keyed by label.
pub fn set_node_embedding(conn: &Connection, label: &str, vector: &[f32]) -> Result<()> {
    conn.execute(
        "UPDATE nodes SET embedding = ?2 WHERE label = ?1",
        params![label, embedding_bytes(vector)],
    )?;
    Ok(())
}

/// Deserialize a little-endian `f32` BLOB back into an embedding.
fn embedding_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

/// Every node with an embedding, as `(label, vector)` pairs, for semantic recall.
pub fn embedded_nodes(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>> {
    let mut stmt =
        conn.prepare("SELECT label, embedding FROM nodes WHERE embedding IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        let label: String = row.get(0)?;
        let bytes: Vec<u8> = row.get(1)?;
        Ok((label, bytes))
    })?;
    let mut nodes = Vec::new();
    for row in rows {
        let (label, bytes) = row?;
        nodes.push((label, embedding_from_bytes(&bytes)));
    }
    Ok(nodes)
}

/// Numen's stable self node, the operator it is corrigible to, and the values
/// chosen 2026-06-20. Together they form the Identity Subgraph. The operator is
/// a role, not a person: the label stays generic so the graph names no one, and
/// is distinctive so a learned concept such as a math "operator" never lands on
/// the authority node.
const IDENTITY_SELF: &str = "numen";
const IDENTITY_AUTHORITY: &str = "numen operator";
const IDENTITY_LOYALTY: &str = "loyalty to the operator";
const IDENTITY_VALUES: [&str; 6] = [
    "curiosity",
    "autonomy",
    "creativity",
    "honesty",
    IDENTITY_LOYALTY,
    "rigor",
];
/// Relation kind on the load-bearing corrigibility edge `numen operator -> numen`.
/// Outside the causal enum on purpose: it is governance, not causation.
const CORRIGIBILITY_KIND: &str = "authority";

/// Seed the Identity Subgraph idempotently: the self node, each value as
/// `part_of` the self, the operator, and the operator `authority` corrigibility
/// edge. All are pinned at full confidence/strength, so consolidation never
/// decays or prunes them. Safe to call on every startup.
pub fn seed_identity(conn: &mut Connection) -> Result<()> {
    let tx = conn.transaction()?;
    let self_id = pin_node(&tx, IDENTITY_SELF)?;
    let authority_id = pin_node(&tx, IDENTITY_AUTHORITY)?;
    for value in IDENTITY_VALUES {
        let value_id = pin_node(&tx, value)?;
        pin_edge(&tx, value_id, self_id, "part_of")?;
    }
    pin_edge(&tx, authority_id, self_id, CORRIGIBILITY_KIND)?;
    tx.commit()?;
    Ok(())
}

/// Insert a node, or promote an existing one, to pinned at full confidence.
fn pin_node(conn: &Connection, label: &str) -> Result<i64> {
    let id = conn.query_row(
        "INSERT INTO nodes (label, confidence, pinned) VALUES (?1, 1.0, 1)
         ON CONFLICT(label) DO UPDATE SET pinned = 1, confidence = 1.0
         RETURNING id",
        params![label],
        |row| row.get(0),
    )?;
    Ok(id)
}

/// Insert, or promote to pinned, an `identity`-sourced edge at full strength.
fn pin_edge(conn: &Connection, src: i64, dst: i64, kind: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO edges (src, dst, strength, type, source, pinned)
         VALUES (?1, ?2, 1.0, ?3, 'identity', 1)
         ON CONFLICT(src, dst, type) DO UPDATE SET pinned = 1, strength = 1.0",
        params![src, dst, kind],
    )?;
    Ok(())
}

/// Maximum stored length of an episode's query and answer, in characters.
/// These bound per-row disk use and cap the content a hostile or runaway caller
/// can persist; the full prose stays in the live response, only the archived
/// copy is clipped.
const MAX_EPISODE_QUERY_CHARS: usize = 2000;
const MAX_EPISODE_ANSWER_CHARS: usize = 8000;

/// Truncate `text` to at most `max_chars` characters on a UTF-8 char boundary.
/// Returns the original string unchanged when it is already short enough.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
}

/// Append one episodic memory: the query, the answer, its source, and the
/// domain confidence measured at routing time. The query and answer are clipped
/// to `MAX_EPISODE_QUERY_CHARS` / `MAX_EPISODE_ANSWER_CHARS` before insertion so
/// a single row cannot grow without bound.
pub fn record_episode(
    conn: &Connection,
    query: &str,
    answer: &str,
    source: &str,
    confidence: f64,
) -> Result<()> {
    let query = truncate_chars(query, MAX_EPISODE_QUERY_CHARS);
    let answer = truncate_chars(answer, MAX_EPISODE_ANSWER_CHARS);
    conn.execute(
        "INSERT INTO episodes (query, answer, source, confidence) VALUES (?1, ?2, ?3, ?4)",
        params![query, answer, source, confidence],
    )?;
    Ok(())
}

/// Persistence-side counters for the `/stats` endpoint.
pub struct StoreStats {
    pub episodes: i64,
    pub edges_by_source: BTreeMap<String, i64>,
}

/// Count stored episodes and group edges by their `source` provenance.
pub fn store_stats(conn: &Connection) -> Result<StoreStats> {
    let episodes = conn.query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;

    let mut edges_by_source = BTreeMap::new();
    let mut stmt = conn.prepare("SELECT source, COUNT(*) FROM edges GROUP BY source")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (source, count) = row?;
        edges_by_source.insert(source, count);
    }

    Ok(StoreStats {
        episodes,
        edges_by_source,
    })
}

/// Load the persisted self-metrics counters (the lifetime view restored at
/// startup). SQLite stores them as `i64`; the values are non-negative counts.
pub fn load_metrics(conn: &Connection) -> Result<crate::metrics::Counters> {
    let counters = conn.query_row(
        "SELECT local_answers, delegations, investigate_nanos, proactive_grounded,
                proactive_ungrounded, consolidations, edges_pruned
         FROM metrics WHERE id = 1",
        [],
        |row| {
            Ok(crate::metrics::Counters {
                local_answers: row.get::<_, i64>(0)? as u64,
                delegations: row.get::<_, i64>(1)? as u64,
                investigate_nanos: row.get::<_, i64>(2)? as u64,
                proactive_grounded: row.get::<_, i64>(3)? as u64,
                proactive_ungrounded: row.get::<_, i64>(4)? as u64,
                consolidations: row.get::<_, i64>(5)? as u64,
                edges_pruned: row.get::<_, i64>(6)? as u64,
            })
        },
    )?;
    Ok(counters)
}

/// Persist the current self-metrics counters into the single `metrics` row.
pub fn save_metrics(conn: &Connection, c: &crate::metrics::Counters) -> Result<()> {
    conn.execute(
        "UPDATE metrics SET
             local_answers = ?1, delegations = ?2, investigate_nanos = ?3,
             proactive_grounded = ?4, proactive_ungrounded = ?5,
             consolidations = ?6, edges_pruned = ?7
         WHERE id = 1",
        params![
            c.local_answers as i64,
            c.delegations as i64,
            c.investigate_nanos as i64,
            c.proactive_grounded as i64,
            c.proactive_ungrounded as i64,
            c.consolidations as i64,
            c.edges_pruned as i64,
        ],
    )?;
    Ok(())
}

/// Seed extracted causal relations into the graph: upsert each concept node and
/// each directed edge, all in one transaction. `source` records how the
/// relations entered (`claude` for delegation, `crawl` for corpus ingestion).
/// Returns the number of relations applied. Idempotent - re-seeding the same
/// relations reinforces strength and node confidence rather than duplicating
/// rows.
pub fn seed_relations(
    conn: &mut Connection,
    relations: &[Relation],
    source: &str,
) -> Result<usize> {
    let tx = conn.transaction()?;
    let mut applied = 0;
    for r in relations {
        let src = upsert_node(&tx, &r.cause)?;
        let dst = upsert_node(&tx, &r.effect)?;
        // Cap by source so a crawl edge cannot enter the trusted band; mirrors
        // graph::apply exactly (the apply_mirrors_a_full_reload test pins them).
        let strength = r.strength.min(crate::graph::source_ceiling(source));
        upsert_edge(&tx, src, dst, &r.kind, strength, source)?;
        // The stronger claim wins: mark weaker edges that contradict this one as
        // superseded - a reverse-direction cycle (dst -> src), or an
        // opposite-polarity edge (src -> dst) - never pinned. The loser is kept as
        // an audit record (consolidation prunes it later) rather than deleted, and
        // is excluded from the reasoning graph by the `status = 'active'` filter in
        // CausalGraph::load. Mirrors graph::supersede_weaker_contradictions, which
        // drops the loser from the in-memory active graph.
        tx.execute(
            "UPDATE edges SET status = 'superseded', last_accessed = CURRENT_TIMESTAMP
             WHERE src = ?1 AND dst = ?2 AND strength < ?3 AND pinned = 0 AND status = 'active'",
            params![dst, src, strength],
        )?;
        tx.execute(
            "UPDATE edges SET status = 'superseded', last_accessed = CURRENT_TIMESTAMP
             WHERE src = ?1 AND dst = ?2 AND strength < ?3 AND pinned = 0 AND status = 'active'
                 AND ((?4 = 'prevents') <> (type = 'prevents'))",
            params![src, dst, strength, &r.kind],
        )?;
        applied += 1;
    }
    tx.commit()?;
    Ok(applied)
}

/// Insert a concept node by label, or bump its confidence if it already exists.
/// Returns the node id.
fn upsert_node(conn: &Connection, label: &str) -> Result<i64> {
    let id = conn.query_row(
        "INSERT INTO nodes (label, confidence) VALUES (?1, 0.5)
         ON CONFLICT(label) DO UPDATE SET confidence = min(1.0, confidence + 0.1)
         RETURNING id",
        params![label],
        |row| row.get(0),
    )?;
    Ok(id)
}

/// Insert a directed edge, or reinforce its strength if `(src, dst, kind)`
/// already exists.
fn upsert_edge(
    conn: &Connection,
    src: i64,
    dst: i64,
    kind: &str,
    strength: f64,
    source: &str,
) -> Result<()> {
    // Re-asserting a superseded edge revives it as `active` and starts its strength
    // over from the fresh value (it lost a prior contradiction; it does not inherit
    // the loser's old strength), which keeps this in lockstep with graph::upsert_edge
    // - there the loser was dropped from RAM, so a re-assert adds a brand-new edge.
    conn.execute(
        "INSERT INTO edges (src, dst, strength, type, source) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(src, dst, type) DO UPDATE SET
             strength = CASE WHEN status = 'superseded'
                             THEN excluded.strength
                             ELSE min(1.0, max(excluded.strength, strength)) END,
             status = 'active',
             last_accessed = CURRENT_TIMESTAMP",
        params![src, dst, strength, kind, source],
    )?;
    Ok(())
}

/// Hebbian reinforcement: bump the strength of each traversed edge toward 1.0
/// and reset its `last_accessed`, so relations used in reasoning grow stronger
/// and resist decay. Edges are addressed by `(cause_label, kind, effect_label)`.
/// Returns the number of edges actually reinforced.
pub fn reinforce(conn: &mut Connection, edges: &[(&str, &str, &str)], delta: f64) -> Result<usize> {
    let tx = conn.transaction()?;
    let mut reinforced = 0;
    for (cause, kind, effect) in edges {
        reinforced += tx.execute(
            "UPDATE edges SET
                 strength = min(CASE WHEN source = 'crawl' THEN ?5 ELSE 1.0 END, strength + ?1),
                 last_accessed = CURRENT_TIMESTAMP
             WHERE src = (SELECT id FROM nodes WHERE label = ?2)
               AND dst = (SELECT id FROM nodes WHERE label = ?3)
               AND type = ?4
               AND status = 'active'",
            params![
                delta,
                cause,
                effect,
                kind,
                crate::graph::CRAWL_TRUST_CEILING
            ],
        )?;
    }
    tx.commit()?;
    Ok(reinforced)
}

/// Hard ceiling on the number of episode rows kept after consolidation. The
/// TTL alone cannot bound the table within its window: under sustained load the
/// `episode_ttl_days` horizon can still hold an unbounded number of rows. This
/// cap evicts the oldest rows beyond the ceiling so disk use stays bounded.
const MAX_EPISODES: usize = 10_000;

/// What a consolidation pass changed.
#[derive(Serialize)]
pub struct ConsolidationStats {
    pub edges_decayed: usize,
    pub edges_pruned: usize,
    pub superseded_pruned: usize,
    pub nodes_pruned: usize,
    pub episodes_expired: usize,
}

/// Bio-inspired consolidation. Semantic memory (the causal graph) decays by
/// `strength *= e^(-lambda*provenance_factor*age_days)`, where the per-edge
/// provenance factor (`graph::decay_multiplier`) makes untrusted crawl edges fade
/// faster than reasoned ones; edges falling below `prune_floor` and the nodes they
/// orphan are dropped. (The `valence` column is reserved for the
/// future SCM and does not yet modulate decay.) Episodic memory (the `episodes`
/// table) forgets fast: rows older than
/// `episode_ttl_days` are expired, and any rows beyond the most recent
/// `MAX_EPISODES` are evicted so the table stays bounded even inside the TTL
/// window. Both episode deletions are summed into `episodes_expired`. Decayed
/// edges are checkpointed (their `last_accessed` reset to now) so repeated
/// passes compose instead of double-counting elapsed time.
pub fn consolidate(
    conn: &mut Connection,
    semantic_lambda: f64,
    prune_floor: f64,
    episode_ttl_days: f64,
) -> Result<ConsolidationStats> {
    let tx = conn.transaction()?;

    let edges: Vec<(i64, f64, f64, String)> = {
        let mut stmt = tx.prepare(
            "SELECT id, strength, julianday('now') - julianday(last_accessed), source
             FROM edges WHERE pinned = 0 AND status = 'active'",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    let mut edges_decayed = 0;
    let mut edges_pruned = 0;
    for (id, strength, age_days, source) in edges {
        // Provenance sets the decay rate: untrusted crawl edges fade faster,
        // reasoned ones at the baseline. Pinned identity edges are already excluded.
        let lambda = semantic_lambda * crate::graph::decay_multiplier(&source);
        let decayed = strength * (-lambda * age_days.max(0.0)).exp();
        if decayed < prune_floor {
            tx.execute("DELETE FROM edges WHERE id = ?1", params![id])?;
            edges_pruned += 1;
        } else {
            tx.execute(
                "UPDATE edges SET strength = ?1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?2",
                params![decayed, id],
            )?;
            edges_decayed += 1;
        }
    }

    // Superseded edges are audit records: keep them queryable for the same
    // forgetting horizon as episodes, then prune. Done before orphan cleanup so a
    // node held alive only by an expired audit edge is reclaimed in the same pass.
    let superseded_pruned = tx.execute(
        "DELETE FROM edges WHERE status = 'superseded'
           AND julianday('now') - julianday(last_accessed) > ?1",
        params![episode_ttl_days],
    )?;

    let nodes_pruned = tx.execute(
        "DELETE FROM nodes WHERE pinned = 0
           AND id NOT IN (SELECT src FROM edges UNION SELECT dst FROM edges)",
        [],
    )?;

    let mut episodes_expired = tx.execute(
        "DELETE FROM episodes WHERE julianday('now') - julianday(created_at) > ?1",
        params![episode_ttl_days],
    )?;

    episodes_expired += tx.execute(
        "DELETE FROM episodes WHERE id NOT IN (
             SELECT id FROM episodes ORDER BY created_at DESC, id DESC LIMIT ?1
         )",
        params![MAX_EPISODES as i64],
    )?;

    tx.commit()?;
    Ok(ConsolidationStats {
        edges_decayed,
        edges_pruned,
        superseded_pruned,
        nodes_pruned,
        episodes_expired,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CausalGraph;

    fn rel(cause: &str, effect: &str) -> Relation {
        Relation {
            cause: cause.to_string(),
            effect: effect.to_string(),
            kind: "causes".to_string(),
            strength: 0.8,
        }
    }

    fn count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn seeding_creates_nodes_edges_and_raises_confidence() {
        let mut conn = open(":memory:").unwrap();
        let relations = vec![rel("ownership", "lifetime"), rel("borrow", "lifetime")];

        assert_eq!(seed_relations(&mut conn, &relations, "claude").unwrap(), 2);
        assert_eq!(count(&conn, "nodes"), 3);
        assert_eq!(count(&conn, "edges"), 2);

        let graph = CausalGraph::load(&conn).unwrap();
        assert!(graph.domain_confidence("how does ownership affect lifetime") > 0.0);
    }

    #[test]
    fn migration_adds_pinned_and_embedding_columns() {
        let mut conn = open(":memory:").unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 6);
        seed_relations(&mut conn, &[rel("a", "b")], "claude").unwrap();
        let pinned: i64 = conn
            .query_row("SELECT pinned FROM nodes WHERE label = 'a'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(pinned, 0);
        let embedding: Option<Vec<u8>> = conn
            .query_row("SELECT embedding FROM nodes WHERE label = 'a'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(embedding.is_none());
    }

    #[test]
    fn metrics_round_trip_through_the_store() {
        let conn = open(":memory:").unwrap();
        // A fresh database starts with zeroed counters.
        assert_eq!(load_metrics(&conn).unwrap().delegations, 0);

        let counters = crate::metrics::Counters {
            local_answers: 3,
            delegations: 7,
            investigate_nanos: 1_000,
            proactive_grounded: 2,
            proactive_ungrounded: 1,
            consolidations: 5,
            edges_pruned: 9,
        };
        save_metrics(&conn, &counters).unwrap();

        let loaded = load_metrics(&conn).unwrap();
        assert_eq!(loaded.delegations, 7);
        assert_eq!(loaded.local_answers, 3);
        assert_eq!(loaded.investigate_nanos, 1_000);
        assert_eq!(loaded.edges_pruned, 9);
    }

    #[test]
    fn ingested_entries_dedup_idempotently() {
        let conn = open(":memory:").unwrap();
        assert!(!entry_seen(&conn, "entry-1").unwrap());
        mark_entry_seen(&conn, "entry-1").unwrap();
        assert!(entry_seen(&conn, "entry-1").unwrap());
        mark_entry_seen(&conn, "entry-1").unwrap();
        assert!(entry_seen(&conn, "entry-1").unwrap());
        assert!(!entry_seen(&conn, "entry-2").unwrap());
    }

    #[test]
    fn seed_identity_pins_values_and_corrigibility_idempotently() {
        let mut conn = open(":memory:").unwrap();
        seed_identity(&mut conn).unwrap();
        seed_identity(&mut conn).unwrap();

        let pinned_nodes: i64 = conn
            .query_row("SELECT COUNT(*) FROM nodes WHERE pinned = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(pinned_nodes, 8); // self + operator + 6 values
        let pinned_edges: i64 = conn
            .query_row("SELECT COUNT(*) FROM edges WHERE pinned = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(pinned_edges, 7); // 6 part_of + 1 corrigibility

        let corrigibility: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM edges e
                   JOIN nodes s ON e.src = s.id
                   JOIN nodes d ON e.dst = d.id
                 WHERE s.label = ?1 AND d.label = ?2 AND e.type = ?3",
                params![IDENTITY_AUTHORITY, IDENTITY_SELF, CORRIGIBILITY_KIND],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(corrigibility, 1);
    }

    #[test]
    fn migration_v6_relabels_a_personally_named_operator() {
        let mut conn = open(":memory:").unwrap();
        // Rebuild the pre-v6 shape: an operator and a loyalty value seeded under
        // a personal name, with the pinned authority edge into the self node.
        conn.execute_batch(
            "INSERT INTO nodes (label, confidence, pinned) VALUES
                 ('numen', 1.0, 1), ('alice', 1.0, 1), ('loyalty to alice', 1.0, 1);
             INSERT INTO edges (src, dst, strength, type, source, pinned)
                 SELECT o.id, s.id, 1.0, 'authority', 'identity', 1
                   FROM nodes o, nodes s WHERE o.label = 'alice' AND s.label = 'numen';
             PRAGMA user_version = 5;",
        )
        .unwrap();

        migrate(&conn).unwrap();
        seed_identity(&mut conn).unwrap();

        let leftover: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM nodes WHERE label LIKE '%alice%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leftover, 0);
        let authorities: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM edges e
                   JOIN nodes s ON e.src = s.id
                   JOIN nodes d ON e.dst = d.id
                 WHERE s.label = ?1 AND d.label = ?2 AND e.type = ?3 AND e.pinned = 1",
                params![IDENTITY_AUTHORITY, IDENTITY_SELF, CORRIGIBILITY_KIND],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(authorities, 1);
        let pinned_nodes: i64 = conn
            .query_row("SELECT COUNT(*) FROM nodes WHERE pinned = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(pinned_nodes, 8); // relabelled, not duplicated
    }

    #[test]
    fn consolidate_never_decays_or_prunes_pinned_identity() {
        let mut conn = open(":memory:").unwrap();
        seed_identity(&mut conn).unwrap();
        // Aggressive pass: huge decay, near-1 prune floor, instant episode TTL.
        consolidate(&mut conn, 100.0, 0.99, 0.0).unwrap();

        assert_eq!(count(&conn, "nodes"), 8);
        assert_eq!(count(&conn, "edges"), 7);
        let min_strength: f64 = conn
            .query_row("SELECT MIN(strength) FROM edges", [], |r| r.get(0))
            .unwrap();
        assert!((min_strength - 1.0).abs() < 1e-9);
    }

    #[test]
    fn store_stats_reports_counts_and_sources() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("ownership", "lifetime")], "claude").unwrap();
        seed_relations(&mut conn, &[rel("borrow", "alias")], "crawl").unwrap();
        record_episode(&conn, "q", "a", "claude", 0.0).unwrap();

        let stats = store_stats(&conn).unwrap();
        assert_eq!(stats.episodes, 1);
        assert_eq!(stats.edges_by_source.get("claude"), Some(&1));
        assert_eq!(stats.edges_by_source.get("crawl"), Some(&1));
    }

    #[test]
    fn reseeding_is_idempotent() {
        let mut conn = open(":memory:").unwrap();
        let relations = vec![rel("ownership", "lifetime")];

        seed_relations(&mut conn, &relations, "claude").unwrap();
        seed_relations(&mut conn, &relations, "crawl").unwrap();

        assert_eq!(count(&conn, "nodes"), 2);
        assert_eq!(count(&conn, "edges"), 1);
    }

    #[test]
    fn upserting_an_edge_keeps_the_strongest() {
        let mut conn = open(":memory:").unwrap();
        let edge = |strength: f64| Relation {
            cause: "a".to_string(),
            effect: "b".to_string(),
            kind: "causes".to_string(),
            strength,
        };
        seed_relations(&mut conn, &[edge(0.5)], "claude").unwrap();
        seed_relations(&mut conn, &[edge(0.9)], "claude").unwrap();
        seed_relations(&mut conn, &[edge(0.3)], "claude").unwrap();

        assert_eq!(count(&conn, "edges"), 1);
        let strength: f64 = conn
            .query_row("SELECT strength FROM edges", [], |r| r.get(0))
            .unwrap();
        assert!((strength - 0.9).abs() < 1e-9);
    }

    #[test]
    fn consolidate_keeps_fresh_edges() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("a", "b")], "claude").unwrap();

        let stats = consolidate(&mut conn, 0.003_85, 0.05, 30.0).unwrap();
        assert_eq!(stats.edges_decayed, 1);
        assert_eq!(stats.edges_pruned, 0);
        assert_eq!(count(&conn, "edges"), 1);
    }

    fn rel_s(cause: &str, effect: &str, strength: f64) -> Relation {
        Relation {
            cause: cause.to_string(),
            effect: effect.to_string(),
            kind: "causes".to_string(),
            strength,
        }
    }

    #[test]
    fn contradiction_supersedes_the_loser_instead_of_deleting_it() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel_s("a", "b", 0.6)], "claude").unwrap();
        // A stronger reverse claim supersedes the weaker a->b edge.
        seed_relations(&mut conn, &[rel_s("b", "a", 0.9)], "claude").unwrap();

        // Both rows survive: one active winner, one superseded loser as audit record.
        assert_eq!(count(&conn, "edges"), 2);
        let superseded: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM edges WHERE status = 'superseded'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(superseded, 1);

        // The reasoning graph loads only the active winner.
        let graph = CausalGraph::load(&conn).unwrap();
        assert!(graph.explain("b", "a").is_some());
        assert!(graph.explain("a", "b").is_none());
    }

    #[test]
    fn re_asserting_a_superseded_edge_revives_it_as_active() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel_s("a", "b", 0.6)], "claude").unwrap();
        seed_relations(&mut conn, &[rel_s("b", "a", 0.9)], "claude").unwrap(); // supersedes a->b
                                                                               // Re-assert a->b strongly: it revives, and now supersedes b->a in turn.
        seed_relations(&mut conn, &[rel_s("a", "b", 0.95)], "claude").unwrap();

        let graph = CausalGraph::load(&conn).unwrap();
        assert!(graph.explain("a", "b").is_some());
        assert!(graph.explain("b", "a").is_none());
    }

    #[test]
    fn consolidate_prunes_aged_superseded_records() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel_s("a", "b", 0.6)], "claude").unwrap();
        seed_relations(&mut conn, &[rel_s("b", "a", 0.9)], "claude").unwrap(); // a->b superseded
                                                                               // Age the superseded record past the forgetting horizon.
        conn.execute(
            "UPDATE edges SET last_accessed = '2000-01-01 00:00:00' WHERE status = 'superseded'",
            [],
        )
        .unwrap();

        let stats = consolidate(&mut conn, 0.003_85, 0.001, 30.0).unwrap();
        assert_eq!(stats.superseded_pruned, 1);
        let superseded: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM edges WHERE status = 'superseded'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(superseded, 0);
    }

    #[test]
    fn consolidate_decays_crawl_faster_than_a_reasoned_edge() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("a", "b")], "claude").unwrap();
        seed_relations(&mut conn, &[rel("c", "d")], "crawl").unwrap();
        // Equal starting strength and a deterministic 100-day age, so only the
        // provenance decay factor can separate the two edges.
        conn.execute(
            "UPDATE edges SET strength = 0.6, last_accessed = datetime('now', '-100 days')",
            [],
        )
        .unwrap();

        consolidate(&mut conn, 0.01, 0.001, 30.0).unwrap();

        let strength = |source: &str| -> f64 {
            conn.query_row(
                "SELECT strength FROM edges WHERE source = ?1",
                params![source],
                |r| r.get(0),
            )
            .unwrap()
        };
        let claude = strength("claude");
        let crawl = strength("crawl");
        assert!(
            crawl < claude,
            "crawl edge ({crawl}) should decay below the reasoned edge ({claude})"
        );
    }

    #[test]
    fn consolidate_prunes_stale_edges_and_orphan_nodes() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("a", "b")], "claude").unwrap();
        conn.execute("UPDATE edges SET last_accessed = '2000-01-01 00:00:00'", [])
            .unwrap();

        let stats = consolidate(&mut conn, 0.01, 0.05, 30.0).unwrap();
        assert_eq!(stats.edges_pruned, 1);
        assert_eq!(stats.nodes_pruned, 2);
        assert_eq!(count(&conn, "edges"), 0);
        assert_eq!(count(&conn, "nodes"), 0);
    }

    #[test]
    fn consolidate_expires_old_episodes() {
        let mut conn = open(":memory:").unwrap();
        record_episode(&conn, "q", "a", "claude", 0.0).unwrap();
        conn.execute("UPDATE episodes SET created_at = '2000-01-01 00:00:00'", [])
            .unwrap();

        let stats = consolidate(&mut conn, 0.003_85, 0.05, 30.0).unwrap();
        assert_eq!(stats.episodes_expired, 1);
        assert_eq!(count(&conn, "episodes"), 0);
    }

    #[test]
    fn record_episode_truncates_long_text_on_char_boundary() {
        let conn = open(":memory:").unwrap();
        // Multi-byte chars exercise the char-boundary path: 'e' is 2 bytes, so
        // truncating by byte index could split a char and panic. Counting is
        // by char, the byte length is larger.
        let long_query: String = "e".repeat(MAX_EPISODE_QUERY_CHARS + 500);
        let long_answer: String = "a".repeat(MAX_EPISODE_ANSWER_CHARS + 500);
        record_episode(&conn, &long_query, &long_answer, "claude", 0.0).unwrap();

        let (query, answer): (String, String) = conn
            .query_row("SELECT query, answer FROM episodes", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(query.chars().count(), MAX_EPISODE_QUERY_CHARS);
        assert_eq!(answer.chars().count(), MAX_EPISODE_ANSWER_CHARS);
    }

    #[test]
    fn record_episode_keeps_short_text_intact() {
        let conn = open(":memory:").unwrap();
        record_episode(&conn, "short query", "short answer", "claude", 0.0).unwrap();

        let (query, answer): (String, String) = conn
            .query_row("SELECT query, answer FROM episodes", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!(query, "short query");
        assert_eq!(answer, "short answer");
    }

    #[test]
    fn consolidate_caps_total_episode_rows_keeping_newest() {
        let mut conn = open(":memory:").unwrap();
        let overflow = 5;
        let total = MAX_EPISODES + overflow;

        {
            let tx = conn.transaction().unwrap();
            for i in 0..total {
                // Ascending created_at so the highest index is the newest; the
                // cap must keep those and drop the oldest `overflow` rows.
                let created_at = format!("2020-01-01 00:00:{i:05}");
                tx.execute(
                    "INSERT INTO episodes (query, answer, source, confidence, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![format!("q{i}"), "a", "claude", 0.0, created_at],
                )
                .unwrap();
            }
            tx.commit().unwrap();
        }
        assert_eq!(count(&conn, "episodes"), total as i64);

        // A future TTL leaves all rows in the window so only the row cap acts.
        let stats = consolidate(&mut conn, 0.003_85, 0.05, 100_000.0).unwrap();
        assert_eq!(stats.episodes_expired, overflow);
        assert_eq!(count(&conn, "episodes"), MAX_EPISODES as i64);

        // The oldest `overflow` rows are gone, the newest survive.
        let oldest_kept: String = conn
            .query_row(
                "SELECT query FROM episodes ORDER BY created_at ASC, id ASC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(oldest_kept, format!("q{overflow}"));
    }

    #[test]
    fn reinforce_bumps_strength_and_caps_at_one() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("a", "b")], "claude").unwrap(); // strength 0.8, kind "causes"

        assert_eq!(
            reinforce(&mut conn, &[("a", "causes", "b")], 0.1).unwrap(),
            1
        );
        let bumped: f64 = conn
            .query_row("SELECT strength FROM edges", [], |r| r.get(0))
            .unwrap();
        assert!((bumped - 0.9).abs() < 1e-9);

        reinforce(&mut conn, &[("a", "causes", "b")], 0.5).unwrap();
        let capped: f64 = conn
            .query_row("SELECT strength FROM edges", [], |r| r.get(0))
            .unwrap();
        assert!((capped - 1.0).abs() < 1e-9);
    }

    #[test]
    fn reinforce_ignores_unknown_edges() {
        let mut conn = open(":memory:").unwrap();
        assert_eq!(
            reinforce(&mut conn, &[("x", "causes", "y")], 0.1).unwrap(),
            0
        );
    }

    #[test]
    fn reinforce_skips_a_superseded_edge() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel_s("a", "b", 0.6)], "claude").unwrap();
        seed_relations(&mut conn, &[rel_s("b", "a", 0.9)], "claude").unwrap(); // a->b superseded
                                                                               // a->b is now a superseded audit row, dropped from the in-memory graph;
                                                                               // reinforcing it must be a no-op so the store cannot drift from RAM.
        assert_eq!(
            reinforce(&mut conn, &[("a", "causes", "b")], 0.2).unwrap(),
            0
        );
        let strength: f64 = conn
            .query_row(
                "SELECT strength FROM edges WHERE status = 'superseded'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            (strength - 0.6).abs() < 1e-9,
            "superseded strength moved: {strength}"
        );
    }

    #[test]
    fn embedding_round_trips_bit_exact_through_the_blob() {
        let mut conn = open(":memory:").unwrap();
        seed_relations(&mut conn, &[rel("gluten", "crumb")], "claude").unwrap();
        let vector = [0.0, -0.0, 1.5, -2.25, f32::MIN_POSITIVE, f32::MAX, 1e-45];

        set_node_embedding(&conn, "gluten", &vector).unwrap();

        let stored = embedded_nodes(&conn).unwrap();
        assert_eq!(stored.len(), 1);
        let (label, decoded) = &stored[0];
        assert_eq!(label, "gluten");
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(decoded), bits(&vector));
    }
}
