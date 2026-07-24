//! End-to-end tests for the `TemporalGraph` facade: the bi-temporal invariants
//! (invalidation, as-of, history), deterministic traversal, and hybrid semantic
//! retrieval — driven through the public API exactly as a binding would.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lodedb_graph::{
    AsOf, Direction, EmbedRole, Embedder, GraphConfig, GraphError, Result, TemporalGraph,
};
use serde_json::json;

/// A tiny deterministic embedder (dim 8): bucket bytes into 8 bins, L2-normalize.
/// Similar text → similar vectors, which is all these membership assertions need.
struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn dimension(&self) -> usize {
        8
    }
    fn embed(&self, texts: &[String], _role: EmbedRole) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; 8];
                for b in t.to_lowercase().bytes() {
                    v[(b % 8) as usize] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut v {
                        *x /= norm;
                    }
                } else {
                    v[0] = 1.0;
                }
                v
            })
            .collect())
    }
}

struct FixedQueryEmbedder;

impl Embedder for FixedQueryEmbedder {
    fn dimension(&self) -> usize {
        8
    }

    fn embed(&self, texts: &[String], _role: EmbedRole) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect())
    }
}

struct ToggleEmbedder {
    fail: Arc<AtomicBool>,
}

impl Embedder for ToggleEmbedder {
    fn dimension(&self) -> usize {
        8
    }

    fn embed(&self, texts: &[String], _role: EmbedRole) -> Result<Vec<Vec<f32>>> {
        if self.fail.load(Ordering::SeqCst) {
            return Err(GraphError::Embedding("injected embedding failure".to_string()));
        }
        Ok(texts
            .iter()
            .map(|_| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect())
    }
}

fn graph() -> TemporalGraph {
    let config = GraphConfig {
        vector_dim: 8,
        ..GraphConfig::default()
    };
    TemporalGraph::open_in_memory(config, Some(Box::new(HashEmbedder))).unwrap()
}

/// The headline invariant: a contradicting fact invalidates the prior one instead of
/// deleting it, so "now", "as-of past", and "history" all answer correctly.
#[test]
fn invalidation_preserves_history_and_as_of() {
    let mut g = graph();

    g.upsert_entity("alice", "Person", "Alice, engineer", json!({}), None, None)
        .unwrap();
    g.upsert_entity("acme", "Org", "Acme Corp", json!({}), None, None)
        .unwrap();
    g.upsert_entity("globex", "Org", "Globex Corp", json!({}), None, None)
        .unwrap();

    // Alice worked at Acme from t=1000.
    let f_acme = g
        .add_fact("alice", "works_at", "acme", "Alice works at Acme", json!({}), vec![], Some(1000), &[])
        .unwrap();
    // At t=2000 she moves to Globex — this invalidates the Acme fact.
    let _f_globex = g
        .add_fact(
            "alice",
            "works_at",
            "globex",
            "Alice works at Globex",
            json!({}),
            vec![],
            Some(2000),
            std::slice::from_ref(&f_acme),
        )
        .unwrap();

    // Current view: only Globex.
    let now = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::Now).unwrap();
    assert_eq!(now.len(), 1, "exactly one live works_at");
    assert_eq!(now[0].dst, "globex");

    // As of 1500 (before the move): Acme.
    let then = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::At(1500)).unwrap();
    assert_eq!(then.len(), 1, "one fact valid at t=1500");
    assert_eq!(then[0].dst, "acme");

    // As of 2500 (after the move): Globex.
    let later = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::At(2500)).unwrap();
    assert_eq!(later.len(), 1);
    assert_eq!(later[0].dst, "globex");

    // History: both facts survive; the Acme fact is closed, not deleted.
    let hist = g.history("alice").unwrap();
    assert_eq!(hist.len(), 2, "both assertions preserved in history");
    let acme_fact = hist.iter().find(|f| f.id == f_acme).unwrap();
    assert_eq!(acme_fact.invalid_at, Some(2000), "closed at the new fact's valid_at");
    assert!(acme_fact.expired_at.is_some(), "expired on the transaction axis");
}

/// Regression: superseding with an UNKNOWN new `valid_at` must still close the prior on
/// the event axis, using the new fact's effective (reference) time — otherwise an
/// `AsOf::At` read counts the prior and its replacement as simultaneously valid (a
/// double-count that disagrees with the `AsOf::Now` view).
#[test]
fn supersede_without_valid_at_closes_prior_on_event_axis() {
    let mut g = graph();
    g.upsert_entity("alice", "Person", "Alice", json!({}), None, None).unwrap();
    g.upsert_entity("acme", "Org", "Acme", json!({}), None, None).unwrap();
    g.upsert_entity("globex", "Org", "Globex", json!({}), None, None).unwrap();

    let f_acme = g
        .add_fact("alice", "works_at", "acme", "Alice works at Acme", json!({}), vec![], Some(1000), &[])
        .unwrap();
    // The superseding fact has NO valid_at, but its episode occurred at t=3000, so the
    // prior's event-time end must fall back to that reference time.
    let ep = g.add_episode("note", "Alice moved to Globex", 3000, json!({}), &[]).unwrap();
    let _f_globex = g
        .add_fact("alice", "works_at", "globex", "Alice works at Globex", json!({}),
                  vec![ep], None, std::slice::from_ref(&f_acme))
        .unwrap();

    // Now view: exactly one live works_at (Globex).
    let now = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::Now).unwrap();
    assert_eq!(now.len(), 1, "one live works_at in the Now view");
    assert_eq!(now[0].dst, "globex");

    // The prior is closed on the event axis at the superseding fact's effective time.
    let acme = g.get_fact(&f_acme).unwrap().unwrap();
    assert_eq!(acme.invalid_at, Some(3000), "prior closed at the new fact's effective (reference) time");
    assert!(acme.expired_at.is_some(), "prior expired on the transaction axis");

    // As-of AFTER the move must NOT double-count — only Globex, agreeing with Now.
    let after = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::At(5000)).unwrap();
    assert_eq!(after.len(), 1, "no event-axis double-count of prior + replacement");
    assert_eq!(after[0].dst, "globex");

    // As-of BEFORE the move: only Acme.
    let before = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::At(1500)).unwrap();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].dst, "acme");
}

/// Deterministic k-hop traversal over the topology, time-scoped.
#[test]
fn k_hop_traversal() {
    let mut g = graph();
    for (id, label) in [("a", "node a"), ("b", "node b"), ("c", "node c"), ("d", "node d")] {
        g.upsert_entity(id, "Thing", label, json!({}), None, None).unwrap();
    }
    g.add_fact("a", "rel", "b", "a rel b", json!({}), vec![], Some(1), &[]).unwrap();
    g.add_fact("b", "rel", "c", "b rel c", json!({}), vec![], Some(1), &[]).unwrap();
    g.add_fact("c", "rel", "d", "c rel d", json!({}), vec![], Some(1), &[]).unwrap();

    let one = g.k_hop(&["a".into()], 1, Direction::Out, AsOf::Now).unwrap();
    assert!(one.entities.contains_key("a") && one.entities.contains_key("b"));
    assert!(!one.entities.contains_key("c"), "c is 2 hops out");

    let two = g.k_hop(&["a".into()], 2, Direction::Out, AsOf::Now).unwrap();
    assert!(two.entities.contains_key("c"), "c reached at 2 hops");
    assert!(!two.entities.contains_key("d"), "d is 3 hops out");
}

/// Enumeration by type + hybrid semantic retrieval find the right entities.
#[test]
fn enumerate_and_search() {
    let mut g = graph();
    g.upsert_entity("alice", "Person", "Alice builds robots", json!({}), None, None).unwrap();
    g.upsert_entity("acme", "Org", "Acme robotics company", json!({}), None, None).unwrap();
    g.upsert_entity("nyc", "Place", "New York City", json!({}), None, None).unwrap();

    let people = g.entities(Some("Person"), AsOf::Now).unwrap();
    assert_eq!(people.len(), 1);
    assert_eq!(people[0].id, "alice");

    let all = g.entities(None, AsOf::Now).unwrap();
    assert_eq!(all.len(), 3);

    let hits = g.semantic_entities(Some("robots"), None, 5, None, AsOf::Now).unwrap();
    assert!(!hits.is_empty(), "semantic search returns entities");

    let stats = g.stats().unwrap();
    assert_eq!(stats.entities, 3);
}

/// A reindex rebuilds the semantic index from the topology truth store.
#[test]
fn reindex_rebuilds_from_truth() {
    let mut g = graph();
    g.upsert_entity("x", "Thing", "widget", json!({}), None, None).unwrap();
    g.upsert_entity("y", "Thing", "gadget", json!({}), None, None).unwrap();
    g.add_fact("x", "is", "y", "x is y", json!({}), vec![], Some(1), &[]).unwrap();
    let out = g.reindex().unwrap();
    assert_eq!(out.reindexed_entities, 2);
    assert_eq!(out.reindexed_facts, 1);
}

/// Episode provenance flows onto a fact's reference_time.
#[test]
fn episode_reference_time() {
    let mut g = graph();
    g.upsert_entity("p", "Person", "Pat", json!({}), None, None).unwrap();
    g.upsert_entity("q", "Org", "QCo", json!({}), None, None).unwrap();
    let mut meta = BTreeMap::new();
    meta.insert("k".to_string(), "v".to_string());
    let ep = g.add_episode("note", "Pat joined QCo", 4242, json!({}), &["p".into()]).unwrap();
    let fid = g
        .add_fact("p", "works_at", "q", "Pat works at QCo", json!({}), vec![ep], Some(4242), &[])
        .unwrap();
    let fact = g.get_fact(&fid).unwrap().unwrap();
    assert_eq!(fact.reference_time, Some(4242), "reference_time from the source episode");
}

/// An open-start fact (no `valid_at`) must be included by BOTH the SQL topology and
/// the semantic index under an as-of query — the encoding-consistency invariant
/// (open start → epoch floor, not far future).
#[test]
fn open_start_as_of_consistency() {
    let mut g = graph();
    g.upsert_entity("s", "Thing", "source thing", json!({}), None, None).unwrap();
    g.upsert_entity("t", "Thing", "target thing", json!({}), None, None).unwrap();
    // valid_at = None → "always started".
    g.add_fact("s", "linked", "t", "s linked to t forever", json!({}), vec![], None, &[]).unwrap();

    // Topology path: as-of an arbitrary instant includes the open-start fact.
    let nbrs = g.neighbors("s", Direction::Out, Some("linked"), AsOf::At(500)).unwrap();
    assert_eq!(nbrs.len(), 1, "open-start fact is valid at any T (topology)");
    assert_eq!(nbrs[0].dst, "t");

    // Index path: semantic fact search as-of the same instant also includes it.
    let hits = g.semantic_facts(Some("linked forever"), None, 5, None, AsOf::At(500)).unwrap();
    assert!(!hits.is_empty(), "open-start fact is valid at any T (index) — encoding is consistent");
}

/// Vector-in vectors are retained with the topology, so both explicit reindex and
/// invalidation preserve historical semantic results without an embedder.
#[test]
fn vector_in_reindex_preserves_invalidation_history() {
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    let mut g = TemporalGraph::open_in_memory(config, None).unwrap();
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    g.upsert_entity_vec("x", "Thing", "widget", json!({}), &v, None, None)
        .unwrap();
    g.upsert_entity_vec("y", "Thing", "gadget", json!({}), &v, None, None)
        .unwrap();
    let old = g
        .add_fact_vec(
            "x",
            "state",
            "y",
            "old state",
            json!({}),
            vec![],
            Some(1_000),
            &[],
            &v,
        )
        .unwrap();
    g.add_fact_vec(
        "x",
        "state",
        "y",
        "new state",
        json!({}),
        vec![],
        Some(2_000),
        std::slice::from_ref(&old),
        &v,
    )
    .unwrap();

    let before = g
        .semantic_facts(None, Some(&v), 5, None, AsOf::At(1_500))
        .unwrap();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].1.id, old);

    let out = g.reindex().unwrap();
    assert_eq!(out.reindexed_entities, 2);
    assert_eq!(out.reindexed_facts, 2);
    let rebuilt = g
        .semantic_facts(None, Some(&v), 5, None, AsOf::At(1_500))
        .unwrap();
    assert_eq!(rebuilt.len(), 1);
    assert_eq!(rebuilt[0].1.id, old);
}

/// Caller-vector validation happens before topology mutation, so bad dimensions
/// and non-finite coordinates cannot leave authoritative rows behind.
#[test]
fn invalid_vectors_do_not_mutate_topology() {
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    let mut g = TemporalGraph::open_in_memory(config, None).unwrap();
    assert!(g
        .upsert_entity_vec("bad", "Thing", "bad", json!({}), &[1.0, 0.0], None, None)
        .is_err());
    assert!(g.get_entity("bad").unwrap().is_none());

    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    g.upsert_entity_vec("a", "Thing", "a", json!({}), &v, None, None)
        .unwrap();
    g.upsert_entity_vec("b", "Thing", "b", json!({}), &v, None, None)
        .unwrap();
    let mut non_finite = v;
    non_finite[3] = f32::NAN;
    assert!(g
        .add_fact_vec(
            "a",
            "rel",
            "b",
            "a rel b",
            json!({}),
            vec![],
            None,
            &[],
            &non_finite,
        )
        .is_err());
    assert_eq!(g.stats().unwrap().facts, 0);
}

/// Negative epoch milliseconds remain exactly ordered in both SQLite and semantic
/// metadata filtering.
#[test]
fn negative_timestamps_match_in_topology_and_semantic_index() {
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    let mut g = TemporalGraph::open_in_memory(config, None).unwrap();
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    g.upsert_entity_vec("a", "Thing", "a", json!({}), &v, None, None)
        .unwrap();
    g.upsert_entity_vec("b", "Thing", "b", json!({}), &v, None, None)
        .unwrap();
    let fact_id = g
        .add_fact_vec(
            "a",
            "rel",
            "b",
            "a rel b",
            json!({}),
            vec![],
            Some(-2_000),
            &[],
            &v,
        )
        .unwrap();
    g.invalidate_fact(&fact_id, Some(-500)).unwrap();

    let neighbors = g
        .neighbors("a", Direction::Out, None, AsOf::At(-1_000))
        .unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].id, fact_id);
    let hits = g
        .semantic_facts(None, Some(&v), 5, None, AsOf::At(-1_000))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].1.id, fact_id);
}

/// `invalidates` naming a fact that does not exist (or is already expired) must fail
/// the whole `add_fact` and leave nothing behind: a typo'd id silently leaving its
/// target live would defeat the invalidation semantics.
#[test]
fn invalidates_unknown_fact_fails_atomically() {
    let mut g = graph();
    g.upsert_entity("a", "Thing", "thing a", json!({}), None, None).unwrap();
    g.upsert_entity("b", "Thing", "thing b", json!({}), None, None).unwrap();

    let err = g
        .add_fact("a", "rel", "b", "a rel b", json!({}), vec![], Some(10), &["f-nope".to_string()])
        .unwrap_err();
    assert!(err.to_string().contains("f-nope"), "names the missing prior: {err}");
    assert_eq!(g.stats().unwrap().facts, 0, "the new fact must not have been inserted");

    // Already-expired priors are refused the same way.
    let f1 = g.add_fact("a", "rel", "b", "first", json!({}), vec![], Some(10), &[]).unwrap();
    g.invalidate_fact(&f1, Some(20)).unwrap();
    let err = g
        .add_fact("a", "rel", "b", "second", json!({}), vec![], Some(30), &[f1])
        .unwrap_err();
    assert!(err.to_string().contains("already expired"), "got: {err}");
}

/// Fact endpoints must be existing entities, and provenance must reference existing
/// episodes; dangling references are refused, not silently stored.
#[test]
fn add_fact_validates_endpoints_and_episodes() {
    let mut g = graph();
    g.upsert_entity("a", "Thing", "thing a", json!({}), None, None).unwrap();

    let err = g.add_fact("a", "rel", "ghost", "a rel ghost", json!({}), vec![], None, &[]).unwrap_err();
    assert!(err.to_string().contains("ghost"), "names the missing endpoint: {err}");

    g.upsert_entity("b", "Thing", "thing b", json!({}), None, None).unwrap();
    let err = g
        .add_fact("a", "rel", "b", "a rel b", json!({}), vec!["ep-ghost".to_string()], None, &[])
        .unwrap_err();
    assert!(err.to_string().contains("ep-ghost"), "names the missing episode: {err}");

    let err = g.add_fact("", "rel", "b", "empty src", json!({}), vec![], None, &[]).unwrap_err();
    assert!(err.to_string().contains("src"), "empty src refused: {err}");
}

/// `add_episode` with a bad mention id must fail whole; the rollback (no orphan
/// episode row) is asserted at the store level in `topology.rs`.
#[test]
fn add_episode_with_bad_mention_fails() {
    let mut g = graph();
    let err = g
        .add_episode("note", "text", 100, json!({}), &["ghost".to_string()])
        .unwrap_err();
    assert!(err.to_string().contains("ghost"), "names the missing entity: {err}");
}

/// Fact provenance is written to the `fact_episodes` join table, and `history`
/// round-trips it through the record's `episodes` list.
#[test]
fn fact_episode_provenance_round_trips() {
    let mut g = graph();
    g.upsert_entity("p", "Person", "Pat", json!({}), None, None).unwrap();
    g.upsert_entity("q", "Org", "QCo", json!({}), None, None).unwrap();
    let ep = g.add_episode("note", "Pat joined QCo", 4242, json!({}), &[]).unwrap();
    let fid = g
        .add_fact("p", "works_at", "q", "Pat works at QCo", json!({}), vec![ep.clone()], None, &[])
        .unwrap();
    let fact = g.get_fact(&fid).unwrap().unwrap();
    assert_eq!(fact.episodes, vec![ep]);
}

/// An index callback failure rolls back SQLite. A durable dirty marker left by a
/// simulated crash makes the next open rebuild topology truth before serving reads.
#[test]
fn index_failure_rolls_back_topology_and_reopen_repairs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g");
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    let fail = Arc::new(AtomicBool::new(false));
    {
        let mut g = TemporalGraph::open(
            &path,
            config.clone(),
            Some(Box::new(ToggleEmbedder { fail: fail.clone() })),
        )
        .unwrap();
        g.upsert_entity("a", "Thing", "a", json!({}), None, None)
            .unwrap();
        g.upsert_entity("b", "Thing", "b", json!({}), None, None)
            .unwrap();

        fail.store(true, Ordering::SeqCst);
        let err = g
            .add_fact(
                "a",
                "rel",
                "b",
                "must roll back",
                json!({}),
                vec![],
                None,
                &[],
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("injected embedding failure"),
            "original failure remains visible: {err}"
        );
        assert_eq!(g.stats().unwrap().facts, 0, "topology transaction rolled back");

        fail.store(false, Ordering::SeqCst);
        g.upsert_entity("c", "Thing", "c", json!({}), None, None)
            .unwrap();
    }
    let topology_path = path.join("topology.sqlite3");
    let conn = rusqlite::Connection::open(topology_path).unwrap();
    conn.execute(
        "UPDATE graph_meta SET value = '1' WHERE key = 'index_dirty'",
        [],
    )
    .unwrap();
    drop(conn);

    let g = TemporalGraph::open(
        &path,
        config,
        Some(Box::new(ToggleEmbedder { fail })),
    )
    .unwrap();
    let stats = g.stats().unwrap();
    assert_eq!(stats.facts, 0);
    assert_eq!(stats.indexed_documents, 3, "dirty index rebuilt on reopen");
    let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let hits = g
        .semantic_entities(None, Some(&query), 5, None, AsOf::Now)
        .unwrap();
    assert_eq!(hits.len(), 3, "dirty index rebuilt from topology on reopen");
}

/// A deleted derivative is rebuilt from topology. Configuration is checked before
/// creating its replacement, so one bad open cannot strand a wrong-dimension index.
#[test]
fn missing_index_rebuilds_without_bad_open_side_effects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g");
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    {
        let mut g =
            TemporalGraph::open(&path, config.clone(), Some(Box::new(HashEmbedder))).unwrap();
        g.upsert_entity("a", "Thing", "alpha", json!({}), None, None)
            .unwrap();
        g.persist().unwrap();
    }
    let index_path = path.join("index");
    std::fs::remove_dir_all(&index_path).unwrap();

    let bad = GraphConfig { vector_dim: 16, ..GraphConfig::default() };
    assert!(TemporalGraph::open(&path, bad, None).is_err());
    assert!(
        !index_path.exists(),
        "configuration refusal must happen before index creation"
    );

    let g = TemporalGraph::open(&path, config, Some(Box::new(HashEmbedder))).unwrap();
    assert_eq!(g.stats().unwrap().indexed_documents, 1);
    let hits = g
        .semantic_entities(Some("alpha"), None, 5, None, AsOf::Now)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].1.id, "a");
}

#[test]
fn embedder_dimension_mismatch_is_rejected_before_creating_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g");
    let config = GraphConfig { vector_dim: 16, ..GraphConfig::default() };
    let err = TemporalGraph::open(&path, config, Some(Box::new(HashEmbedder)))
        .err()
        .expect("dimension mismatch must refuse");
    assert!(err.to_string().contains("embedder dimension"));
    assert!(!path.exists(), "invalid open must not claim an on-disk config");
}

/// On-disk lifecycle: open → write → drop → reopen serves the same records, and
/// reopening with any index-shaping configuration change is refused up front.
#[test]
fn on_disk_reopen_and_configuration_guards() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g");
    let config = GraphConfig { vector_dim: 8, ..GraphConfig::default() };
    {
        let mut g = TemporalGraph::open(&path, config.clone(), Some(Box::new(HashEmbedder))).unwrap();
        g.upsert_entity("alice", "Person", "Alice, engineer", json!({}), None, None).unwrap();
        g.upsert_entity("acme", "Org", "Acme Corp", json!({}), None, None).unwrap();
        g.add_fact("alice", "works_at", "acme", "Alice works at Acme", json!({}), vec![], Some(1000), &[])
            .unwrap();
        g.persist().unwrap();
    }
    {
        let g = TemporalGraph::open(&path, config.clone(), Some(Box::new(HashEmbedder))).unwrap();
        let nbrs = g.neighbors("alice", Direction::Out, Some("works_at"), AsOf::Now).unwrap();
        assert_eq!(nbrs.len(), 1, "topology survives reopen");
        let hits = g.semantic_entities(Some("engineer"), None, 5, None, AsOf::Now).unwrap();
        assert!(hits.iter().any(|(_s, e)| e.id == "alice"), "index survives reopen");
    }
    let bad = GraphConfig { vector_dim: 16, ..GraphConfig::default() };
    let err = TemporalGraph::open(&path, bad, None).err().expect("dim mismatch must refuse");
    assert!(err.to_string().contains("vector_dim"), "dim mismatch refused at open: {err}");

    let bad = GraphConfig { vector_dim: 8, index_facts: false, index_text: true };
    let err = TemporalGraph::open(&path, bad, Some(Box::new(HashEmbedder)))
        .err()
        .expect("index_facts mismatch must refuse");
    assert!(
        err.to_string().contains("index_facts"),
        "index_facts mismatch refused at open: {err}"
    );

    let bad = GraphConfig { vector_dim: 8, index_facts: true, index_text: false };
    assert!(
        TemporalGraph::open(&path, bad, Some(Box::new(HashEmbedder))).is_err(),
        "index_text mismatch must refuse"
    );
}
#[test]
fn strict_now_excludes_future_dated_facts_without_breaking_current_view() {
    let mut g = graph();
    g.upsert_entity(
        "future-src",
        "Thing",
        "future source",
        json!({}),
        None,
        None,
    )
    .unwrap();
    g.upsert_entity(
        "future-dst",
        "Thing",
        "future target",
        json!({}),
        None,
        None,
    )
    .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let future = now + 7 * 24 * 60 * 60 * 1_000;
    let fact_id = g
        .add_fact(
            "future-src",
            "activates",
            "future-dst",
            "future source activates future target",
            json!({}),
            vec![],
            Some(future),
            &[],
        )
        .unwrap();

    assert_eq!(
        g.neighbors("future-src", Direction::Out, None, AsOf::Now)
            .unwrap()
            .len(),
        1,
        "the compatibility current view remains Graphiti-like"
    );
    assert!(
        g.neighbors(
            "future-src",
            Direction::Out,
            None,
            AsOf::NowValid(now)
        )
        .unwrap()
        .is_empty(),
        "strict current validity rejects a future start"
    );
    assert!(
        g.semantic_facts(
            Some("activates future target"),
            None,
            5,
            None,
            AsOf::NowValid(now),
        )
        .unwrap()
        .is_empty(),
        "the strict frame is pushed into semantic fact retrieval"
    );
    let subgraph = g
        .search_subgraph(
            Some("future source"),
            None,
            5,
            1,
            Direction::Both,
            None,
            AsOf::NowValid(now),
        )
        .unwrap();
    assert!(
        subgraph.facts.iter().all(|fact| fact.id != fact_id),
        "strict search_subgraph expansion does not leak the future fact"
    );
}

#[test]
fn transaction_time_travel_reconstructs_what_was_known() {
    let mut g = graph();
    for (id, label) in [("person", "Person"), ("old-org", "Old Org"), ("new-org", "New Org")] {
        g.upsert_entity(id, "Thing", label, json!({}), None, None)
            .unwrap();
    }
    let old = g
        .add_fact(
            "person",
            "works_at",
            "old-org",
            "Person works at Old Org",
            json!({}),
            vec![],
            Some(1_000),
            &[],
        )
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let new = g
        .add_fact(
            "person",
            "works_at",
            "new-org",
            "Person works at New Org",
            json!({}),
            vec![],
            Some(2_000),
            std::slice::from_ref(&old),
        )
        .unwrap();
    let old_row = g.get_fact(&old).unwrap().unwrap();
    let new_row = g.get_fact(&new).unwrap().unwrap();
    let learned_replacement_at = old_row.expired_at.unwrap();
    assert!(old_row.created_at < learned_replacement_at);

    let before_learning = g
        .neighbors(
            "person",
            Direction::Out,
            Some("works_at"),
            AsOf::AtKnown {
                valid_at: 2_500,
                known_at: learned_replacement_at - 1,
            },
        )
        .unwrap();
    assert_eq!(before_learning.len(), 1);
    assert_eq!(before_learning[0].id, old);
    let before_learning_semantic = g
        .semantic_facts(
            Some("Person works at"),
            None,
            5,
            Some("works_at"),
            AsOf::AtKnown {
                valid_at: 2_500,
                known_at: learned_replacement_at - 1,
            },
        )
        .unwrap();
    assert_eq!(before_learning_semantic.len(), 1);
    assert_eq!(before_learning_semantic[0].1.id, old);

    let after_learning = g
        .neighbors(
            "person",
            Direction::Out,
            Some("works_at"),
            AsOf::AtKnown {
                valid_at: 2_500,
                known_at: new_row.created_at,
            },
        )
        .unwrap();
    assert_eq!(after_learning.len(), 1);
    assert_eq!(after_learning[0].id, new);
    let after_learning_semantic = g
        .semantic_facts(
            Some("Person works at"),
            None,
            5,
            Some("works_at"),
            AsOf::AtKnown {
                valid_at: 2_500,
                known_at: new_row.created_at,
            },
        )
        .unwrap();
    assert_eq!(after_learning_semantic.len(), 1);
    assert_eq!(after_learning_semantic[0].1.id, new);
}

#[test]
fn caller_stable_ids_make_episode_and_fact_retries_idempotent() {
    let mut g = graph();
    g.upsert_entity("a", "Thing", "A", json!({}), None, None)
        .unwrap();
    g.upsert_entity("b", "Thing", "B", json!({}), None, None)
        .unwrap();
    let episode = g
        .add_episode_with_id(
            Some("episode-stable"),
            "note",
            "A relates to B",
            100,
            json!({"owner": "u1"}),
            &["a".to_string()],
        )
        .unwrap();
    assert_eq!(
        g.add_episode_with_id(
            Some("episode-stable"),
            "note",
            "A relates to B",
            100,
            json!({"owner": "u1"}),
            &["a".to_string()],
        )
        .unwrap(),
        episode
    );
    assert!(
        g.add_episode_with_id(
            Some("episode-stable"),
            "note",
            "different",
            100,
            json!({"owner": "u1"}),
            &[],
        )
        .is_err()
    );

    let fact = g
        .add_fact_with_id(
            Some("fact-stable"),
            "a",
            "rel",
            "b",
            "A relates to B",
            json!({"owner": "u1"}),
            vec![episode.clone()],
            Some(100),
            &[],
        )
        .unwrap();
    assert_eq!(
        g.add_fact_with_id(
            Some("fact-stable"),
            "a",
            "rel",
            "b",
            "A relates to B",
            json!({"owner": "u1"}),
            vec![episode],
            Some(100),
            &[],
        )
        .unwrap(),
        fact
    );
    assert_eq!(g.stats().unwrap().facts, 1);
}

#[test]
fn episode_enumeration_provenance_and_rollback_are_public() {
    let mut g = graph();
    g.upsert_entity("a", "Thing", "A", json!({}), None, None)
        .unwrap();
    g.upsert_entity("b", "Thing", "B", json!({}), None, None)
        .unwrap();
    let ep1 = g
        .add_episode_with_id(Some("ep1"), "note", "one", 1, json!({}), &[])
        .unwrap();
    let ep2 = g
        .add_episode_with_id(Some("ep2"), "note", "two", 2, json!({}), &[])
        .unwrap();
    let retained = g
        .add_fact(
            "a",
            "rel",
            "b",
            "supported by two episodes",
            json!({}),
            vec![ep1.clone(), ep2.clone()],
            Some(1),
            &[],
        )
        .unwrap();
    let removed = g
        .add_fact(
            "a",
            "rel2",
            "b",
            "created by episode two",
            json!({}),
            vec![ep2.clone()],
            Some(2),
            &[],
        )
        .unwrap();

    assert_eq!(g.episodes().unwrap().len(), 2);
    assert_eq!(g.facts_by_episode(&ep2).unwrap().len(), 2);
    assert!(g.remove_episode(&ep2).unwrap());
    assert!(g.get_episode(&ep2).unwrap().is_none());
    assert!(g.get_fact(&removed).unwrap().is_none());
    assert_eq!(
        g.get_fact(&retained).unwrap().unwrap().episodes,
        vec![ep1],
        "non-primary support is detached without deleting the fact"
    );
}

#[test]
fn entity_properties_have_independent_version_lineage() {
    let mut g = graph();
    g.upsert_entity(
        "device",
        "Device",
        "Device",
        json!({"status": "new", "region": "us"}),
        Some(100),
        None,
    )
    .unwrap();
    let episode = g
        .add_episode_with_id(
            Some("status-update"),
            "event",
            "device activated",
            200,
            json!({}),
            &[],
        )
        .unwrap();
    let mut sources = BTreeMap::new();
    sources.insert("status".to_string(), episode.clone());
    g.upsert_entity_with_sources(
        "device",
        "Device",
        "Device",
        json!({"status": "active", "region": "us"}),
        Some(200),
        None,
        &sources,
    )
    .unwrap();

    let status = g
        .entity_property_history("device", Some("status"))
        .unwrap();
    assert_eq!(status.len(), 2);
    assert_eq!(status[0].value, json!("new"));
    assert!(status[0].expired_at.is_some());
    assert_eq!(status[1].value, json!("active"));
    assert_eq!(status[1].episode_id.as_deref(), Some(episode.as_str()));
    assert!(status[1].expired_at.is_none());
    assert_eq!(
        g.entity_property_history("device", Some("region"))
            .unwrap()
            .len(),
        1,
        "an unchanged property does not mint a new version"
    );
}

#[test]
fn authorization_predicate_crowd_out_preserves_allowed_top_k() {
    let config = GraphConfig {
        vector_dim: 8,
        ..GraphConfig::default()
    };
    let mut g =
        TemporalGraph::open_in_memory(config, Some(Box::new(FixedQueryEmbedder))).unwrap();
    let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let allowed = [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let other = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let predicate = json!({"$and": [{"owner": "u1"}, {"allowed": true}]});

    for (id, label, properties, embedding) in [
        (
            "forbidden-top",
            "crowd-out query",
            json!({"owner": "u2", "allowed": false}),
            &query,
        ),
        (
            "allowed-best",
            "allowed fallback",
            json!({"owner": "u1", "allowed": true}),
            &allowed,
        ),
        (
            "allowed-other",
            "other fallback",
            json!({"owner": "u1", "allowed": true}),
            &other,
        ),
    ] {
        g.upsert_entity_vec(
            id,
            "Thing",
            label,
            properties,
            embedding,
            None,
            None,
        )
        .unwrap();
    }

    let forbidden_fact = g
        .add_fact_vec(
            "allowed-best",
            "rel",
            "allowed-other",
            "forbidden top fact",
            json!({"owner": "u2", "allowed": false}),
            vec![],
            None,
            &[],
            &query,
        )
        .unwrap();
    let allowed_fact = g
        .add_fact_vec(
            "allowed-best",
            "rel",
            "allowed-other",
            "allowed fallback fact",
            json!({"owner": "u1", "allowed": true}),
            vec![],
            None,
            &[],
            &allowed,
        )
        .unwrap();

    assert_eq!(
        g.semantic_entities(None, Some(&query), 1, None, AsOf::Now)
            .unwrap()[0]
            .1
            .id,
        "forbidden-top",
        "the forbidden entity must crowd out the allowed candidates without a predicate"
    );
    assert_eq!(
        g.semantic_facts(None, Some(&query), 1, Some("rel"), AsOf::Now)
            .unwrap()[0]
            .1
            .id,
        forbidden_fact,
        "the forbidden fact must rank first without a predicate"
    );
    assert_eq!(
        g.resolve_entity("crowd-out query", 1).unwrap()[0].1.id,
        "forbidden-top",
        "the forbidden resolution candidate must rank first without a predicate"
    );

    let entities = g
        .semantic_entities_filtered(
            None,
            Some(&query),
            1,
            None,
            AsOf::Now,
            Some(&predicate),
        )
        .unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].1.id, "allowed-best");

    let facts = g
        .semantic_facts_filtered(
            None,
            Some(&query),
            1,
            Some("rel"),
            AsOf::Now,
            Some(&predicate),
        )
        .unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].1.id, allowed_fact);

    let resolved = g
        .resolve_entity_filtered("crowd-out query", 1, Some(&predicate))
        .unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].1.id, "allowed-best");

    let unfiltered_subgraph = g
        .search_subgraph_filtered(
            None,
            Some(&query),
            1,
            0,
            Direction::Both,
            None,
            Some("rel"),
            AsOf::Now,
            None,
            "fact",
        )
        .unwrap();
    assert_eq!(unfiltered_subgraph.facts[0].id, forbidden_fact);

    let filtered_subgraph = g
        .search_subgraph_filtered(
            None,
            Some(&query),
            1,
            0,
            Direction::Both,
            None,
            Some("rel"),
            AsOf::Now,
            Some(&predicate),
            "fact",
        )
        .unwrap();
    assert_eq!(filtered_subgraph.facts.len(), 1);
    assert_eq!(filtered_subgraph.facts[0].id, allowed_fact);
}

#[test]
fn authorization_predicate_blocks_forbidden_bridge_expansion() {
    let config = GraphConfig {
        vector_dim: 8,
        ..GraphConfig::default()
    };
    let mut g =
        TemporalGraph::open_in_memory(config, Some(Box::new(FixedQueryEmbedder))).unwrap();
    let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let other = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let allowed_properties = json!({"owner": "u1", "allowed": true});
    let predicate = json!({"$and": [{"owner": "u1"}, {"allowed": true}]});

    for (id, properties, embedding) in [
        ("allowed-a", allowed_properties.clone(), &query),
        (
            "forbidden-m",
            json!({"owner": "u2", "allowed": false}),
            &other,
        ),
        ("allowed-c", allowed_properties.clone(), &other),
    ] {
        g.upsert_entity_vec(
            id,
            "Thing",
            id,
            properties,
            embedding,
            None,
            None,
        )
        .unwrap();
    }
    g.add_fact_vec(
        "allowed-a",
        "rel",
        "forbidden-m",
        "allowed a to forbidden m",
        allowed_properties.clone(),
        vec![],
        None,
        &[],
        &other,
    )
    .unwrap();
    g.add_fact_vec(
        "forbidden-m",
        "rel",
        "allowed-c",
        "forbidden m to allowed c",
        allowed_properties,
        vec![],
        None,
        &[],
        &other,
    )
    .unwrap();

    let seeds = vec!["allowed-a".to_string()];
    let unfiltered = g
        .k_hop(&seeds, 2, Direction::Out, AsOf::Now)
        .unwrap();
    assert!(unfiltered.entities.contains_key("allowed-c"));
    assert_eq!(unfiltered.facts.len(), 2);

    let filtered = g
        .k_hop_filtered(
            &seeds,
            2,
            Direction::Out,
            AsOf::Now,
            Some(&predicate),
        )
        .unwrap();
    assert!(filtered.entities.contains_key("allowed-a"));
    assert!(!filtered.entities.contains_key("forbidden-m"));
    assert!(!filtered.entities.contains_key("allowed-c"));
    assert!(filtered
        .facts
        .iter()
        .all(|fact| fact.src != "forbidden-m" && fact.dst != "forbidden-m"));

    let unfiltered_subgraph = g
        .search_subgraph_filtered(
            None,
            Some(&query),
            1,
            2,
            Direction::Out,
            None,
            Some("rel"),
            AsOf::Now,
            None,
            "entity",
        )
        .unwrap();
    assert!(unfiltered_subgraph.entities.contains_key("allowed-c"));

    let filtered_subgraph = g
        .search_subgraph_filtered(
            None,
            Some(&query),
            1,
            2,
            Direction::Out,
            None,
            Some("rel"),
            AsOf::Now,
            Some(&predicate),
            "entity",
        )
        .unwrap();
    assert!(filtered_subgraph.entities.contains_key("allowed-a"));
    assert!(!filtered_subgraph.entities.contains_key("forbidden-m"));
    assert!(!filtered_subgraph.entities.contains_key("allowed-c"));
    assert!(filtered_subgraph
        .facts
        .iter()
        .all(|fact| fact.src != "forbidden-m" && fact.dst != "forbidden-m"));
}
