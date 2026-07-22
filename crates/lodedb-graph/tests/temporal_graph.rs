//! End-to-end tests for the `TemporalGraph` facade: the bi-temporal invariants
//! (invalidation, as-of, history), deterministic traversal, and hybrid semantic
//! retrieval — driven through the public API exactly as a binding would.

use std::collections::BTreeMap;

use lodedb_graph::{AsOf, Direction, EmbedRole, Embedder, GraphConfig, Result, TemporalGraph};
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
            &[f_acme.clone()],
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
    g.add_fact("x", "is", "y", "x is y", json!({}), vec![], Some(1), &[]).unwrap();
    let out = g.reindex().unwrap();
    assert_eq!(out.reindexed_entities, 1);
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
