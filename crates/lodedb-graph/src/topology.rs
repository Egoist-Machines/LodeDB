//! The SQL topology truth store: episodes, entities, typed facts, provenance, and
//! bi-temporal validity — the authoritative half of the graph.
//!
//! Port target: Graphiti's persistence in `graphiti_core/edges.py` +
//! `graphiti_core/nodes.py` (the `save` / `get_*` / traversal Cypher), and the
//! schema shape from LodeDB's own Python topology store
//! (`src/lodedb/graph/_store.py`), translated Cypher → SQL. rusqlite, bundled.
//!
//! IMPLEMENTATION NOTE (Wave 1a): implement every method below against the schema
//! in `SCHEMA`. Use one transaction per mutation; JSON-encode `properties` and
//! `episodes` to TEXT; use the `crate::temporal::as_of_sql` fragment for temporal
//! filtering; chunk `IN (...)` lists (SQLite bound-param limit) as `_store.py`
//! does. Timestamps are `i64` epoch-ms columns; `NULL` = open interval.

use std::collections::BTreeMap;
use std::path::Path;

use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection, Row};

use crate::error::Result;
use crate::model::{AsOf, Direction, Entity, Episode, Fact, TimeMs};

/// DDL for the topology store. See `docs/temporal-graph-design.html` §05.
pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL DEFAULT '',
    body        TEXT NOT NULL DEFAULT '',
    occurred_at INTEGER NOT NULL,
    created_at  INTEGER NOT NULL,
    properties  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL DEFAULT '',
    label       TEXT NOT NULL DEFAULT '',
    properties  TEXT NOT NULL DEFAULT '{}',
    valid_at    INTEGER,
    invalid_at  INTEGER,
    created_at  INTEGER NOT NULL,
    expired_at  INTEGER
);
CREATE TABLE IF NOT EXISTS facts (
    id             TEXT PRIMARY KEY,
    src            TEXT NOT NULL,
    relation       TEXT NOT NULL,
    dst            TEXT NOT NULL,
    fact           TEXT NOT NULL DEFAULT '',
    properties     TEXT NOT NULL DEFAULT '{}',
    episodes       TEXT NOT NULL DEFAULT '[]',
    valid_at       INTEGER,
    invalid_at     INTEGER,
    created_at     INTEGER NOT NULL,
    expired_at     INTEGER,
    reference_time INTEGER
);
CREATE TABLE IF NOT EXISTS fact_episodes (
    fact_id     TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    episode_id  TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, episode_id)
);
CREATE TABLE IF NOT EXISTS episode_mentions (
    episode_id  TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (episode_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_facts_src ON facts(src, relation);
CREATE INDEX IF NOT EXISTS idx_facts_dst ON facts(dst, relation);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_at, invalid_at);
CREATE INDEX IF NOT EXISTS idx_facts_live ON facts(expired_at);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON episode_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
"#;

/// Embedded SQL store of episodes/entities/facts with bi-temporal validity.
pub struct TopologyStore {
    #[allow(dead_code)]
    conn: Connection,
}

/// Max ids per `IN`-clause chunk; kept well under SQLite's bound-parameter limit
/// (`direction = both` binds each id twice). Mirrors `_store.py`'s `_IN_CHUNK`.
const IN_CHUNK: usize = 400;

/// Canonical column lists, so the row-to-struct helpers can read by name and every
/// `SELECT` feeding them agrees on the projection.
const ENTITY_COLS: &str =
    "id, type, label, properties, valid_at, invalid_at, created_at, expired_at";
const FACT_COLS: &str = "id, src, relation, dst, fact, properties, episodes, \
     valid_at, invalid_at, created_at, expired_at, reference_time";
const EPISODE_COLS: &str = "id, source, body, occurred_at, created_at, properties";

impl TopologyStore {
    /// Open (creating if needed) the topology database at `path`, applying the
    /// schema, `PRAGMA journal_mode=WAL` and `foreign_keys=ON`.
    pub fn open(path: &Path) -> Result<Self> {
        Self::init(Connection::open(path)?)
    }

    /// Open an in-memory topology store (tests).
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    /// Shared connection setup: pragmas + schema.
    fn init(conn: Connection) -> Result<Self> {
        // `journal_mode` reports the resulting mode as a row; read and discard it.
        // (On `:memory:` WAL is unsupported and this yields "memory" — a no-op.)
        conn.query_row("PRAGMA journal_mode=WAL", [], |_row| Ok(()))?;
        // The set-form of `foreign_keys` returns no rows, so `execute` is right.
        conn.execute("PRAGMA foreign_keys=ON", [])?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    // -- episodes -----------------------------------------------------------

    pub fn upsert_episode(&self, episode: &Episode) -> Result<()> {
        let properties = props_to_text(&episode.properties)?;
        self.conn.execute(
            "INSERT INTO episodes (id, source, body, occurred_at, created_at, properties) \
             VALUES (?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
                source=excluded.source, body=excluded.body, occurred_at=excluded.occurred_at, \
                created_at=excluded.created_at, properties=excluded.properties",
            params![
                episode.id,
                episode.source,
                episode.body,
                episode.occurred_at,
                episode.created_at,
                properties,
            ],
        )?;
        Ok(())
    }

    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        let sql = format!("SELECT {} FROM episodes WHERE id = ?", EPISODE_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_episode(row)?)),
            None => Ok(None),
        }
    }

    /// Record which entities an episode mentions (Graphiti `MENTIONS`).
    pub fn link_mentions(&self, episode_id: &str, entity_ids: &[String]) -> Result<()> {
        if entity_ids.is_empty() {
            return Ok(());
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO episode_mentions (episode_id, entity_id) VALUES (?, ?)",
            )?;
            for entity_id in entity_ids {
                stmt.execute(params![episode_id, entity_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // -- entities -----------------------------------------------------------

    pub fn upsert_entity(&self, entity: &Entity) -> Result<()> {
        let properties = props_to_text(&entity.properties)?;
        self.conn.execute(
            "INSERT INTO entities \
                (id, type, label, properties, valid_at, invalid_at, created_at, expired_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
                type=excluded.type, label=excluded.label, properties=excluded.properties, \
                valid_at=excluded.valid_at, invalid_at=excluded.invalid_at, \
                created_at=excluded.created_at, expired_at=excluded.expired_at",
            params![
                entity.id,
                entity.entity_type,
                entity.label,
                properties,
                entity.valid_at,
                entity.invalid_at,
                entity.created_at,
                entity.expired_at,
            ],
        )?;
        Ok(())
    }

    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let sql = format!("SELECT {} FROM entities WHERE id = ?", ENTITY_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_entity(row)?)),
            None => Ok(None),
        }
    }

    /// Batched read (chunked `IN`), missing ids omitted.
    pub fn get_entities(&self, ids: &[String]) -> Result<Vec<Entity>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        for chunk in ids.chunks(IN_CHUNK) {
            let placeholders = placeholders(chunk.len());
            let sql = format!(
                "SELECT {} FROM entities WHERE id IN ({})",
                ENTITY_COLS, placeholders
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(params_from_iter(chunk.iter()))?;
            while let Some(row) = rows.next()? {
                found.push(row_to_entity(row)?);
            }
        }
        Ok(found)
    }

    /// Complete-set enumeration by type (nil = all) in a temporal frame.
    pub fn list_entities(&self, entity_type: Option<&str>, as_of: AsOf) -> Result<Vec<Entity>> {
        let mut where_parts: Vec<String> = Vec::new();
        let mut binds: Vec<SqlValue> = Vec::new();

        if let Some(ty) = entity_type {
            where_parts.push("e.type = ?".to_string());
            binds.push(SqlValue::Text(ty.to_string()));
        }
        // As-of predicate over the entity columns (as_of_sql is fixed to alias `f`).
        match as_of {
            AsOf::Now => {
                where_parts.push("(e.expired_at IS NULL AND e.invalid_at IS NULL)".to_string());
            }
            AsOf::At(t) => {
                where_parts.push(
                    "((e.valid_at IS NULL OR e.valid_at <= ?) \
                      AND (e.invalid_at IS NULL OR e.invalid_at > ?))"
                        .to_string(),
                );
                binds.push(SqlValue::Integer(t));
                binds.push(SqlValue::Integer(t));
            }
            AsOf::All => {}
        }

        let where_sql = if where_parts.is_empty() {
            "1=1".to_string()
        } else {
            where_parts.join(" AND ")
        };
        let sql = format!("SELECT {} FROM entities e WHERE {}", ENTITY_COLS, where_sql);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(binds.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_entity(row)?);
        }
        Ok(out)
    }

    /// Every entity id (a lean store primitive; exercised by tests, not yet wired
    /// into the facade, which uses `iter_entities` for reindex).
    #[allow(dead_code)]
    pub fn all_entity_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT id FROM entities")?;
        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        Ok(ids)
    }

    pub fn iter_entities(&self) -> Result<Vec<Entity>> {
        let sql = format!("SELECT {} FROM entities", ENTITY_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_entity(row)?);
        }
        Ok(out)
    }

    /// Remove an entity and its incident facts; returns `(existed, removed_fact_ids)`.
    pub fn remove_entity(&self, id: &str) -> Result<(bool, Vec<String>)> {
        let tx = self.conn.unchecked_transaction()?;
        let removed: Vec<String> = {
            let mut stmt = tx.prepare("SELECT id FROM facts WHERE src = ? OR dst = ?")?;
            let ids = stmt
                .query_map(params![id, id], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<String>>>()?;
            ids
        };
        tx.execute("DELETE FROM facts WHERE src = ? OR dst = ?", params![id, id])?;
        // episode_mentions rows for this entity cascade via the FK.
        let removed_entity = tx.execute("DELETE FROM entities WHERE id = ?", params![id])?;
        tx.commit()?;
        Ok((removed_entity > 0, removed))
    }

    pub fn entity_count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        Ok(n as usize)
    }

    // -- facts --------------------------------------------------------------

    pub fn upsert_fact(&self, fact: &Fact) -> Result<()> {
        let properties = props_to_text(&fact.properties)?;
        let episodes = serde_json::to_string(&fact.episodes)?;
        self.conn.execute(
            "INSERT INTO facts \
                (id, src, relation, dst, fact, properties, episodes, \
                 valid_at, invalid_at, created_at, expired_at, reference_time) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
                src=excluded.src, relation=excluded.relation, dst=excluded.dst, \
                fact=excluded.fact, properties=excluded.properties, episodes=excluded.episodes, \
                valid_at=excluded.valid_at, invalid_at=excluded.invalid_at, \
                created_at=excluded.created_at, expired_at=excluded.expired_at, \
                reference_time=excluded.reference_time",
            params![
                fact.id,
                fact.src,
                fact.relation,
                fact.dst,
                fact.fact,
                properties,
                episodes,
                fact.valid_at,
                fact.invalid_at,
                fact.created_at,
                fact.expired_at,
                fact.reference_time,
            ],
        )?;
        Ok(())
    }

    pub fn get_fact(&self, id: &str) -> Result<Option<Fact>> {
        let sql = format!("SELECT {} FROM facts WHERE id = ?", FACT_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_fact(row)?)),
            None => Ok(None),
        }
    }

    /// Close a fact's validity (the invalidation write): set `invalid_at` and
    /// `expired_at`. Returns whether the fact existed and was live.
    pub fn close_fact(
        &self,
        id: &str,
        invalid_at: Option<TimeMs>,
        expired_at: TimeMs,
    ) -> Result<bool> {
        // Only close a fact that is still live on the transaction axis; never
        // re-close an already-expired one.
        let n = self.conn.execute(
            "UPDATE facts SET invalid_at = ?, expired_at = ? WHERE id = ? AND expired_at IS NULL",
            params![invalid_at, expired_at, id],
        )?;
        Ok(n > 0)
    }

    /// Atomically close `priors` (the invalidation writes) and insert `new_fact` (the
    /// replacement) in ONE transaction, so a crash can never leave priors closed with no
    /// replacement (an event-time validity gap). Each prior is closed only if still live
    /// on the transaction axis. Returns the ids of the priors actually closed, for the
    /// caller to re-index.
    pub fn supersede_and_insert(
        &self,
        priors: &[(String, Option<TimeMs>, TimeMs)],
        new_fact: &Fact,
    ) -> Result<Vec<String>> {
        let properties = props_to_text(&new_fact.properties)?;
        let episodes = serde_json::to_string(&new_fact.episodes)?;
        let tx = self.conn.unchecked_transaction()?;
        let mut closed_ids = Vec::new();
        {
            let mut upd = tx.prepare(
                "UPDATE facts SET invalid_at = ?, expired_at = ? WHERE id = ? AND expired_at IS NULL",
            )?;
            for (id, invalid_at, expired_at) in priors {
                if upd.execute(params![invalid_at, expired_at, id])? > 0 {
                    closed_ids.push(id.clone());
                }
            }
        }
        tx.execute(
            "INSERT INTO facts \
                (id, src, relation, dst, fact, properties, episodes, \
                 valid_at, invalid_at, created_at, expired_at, reference_time) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(id) DO UPDATE SET \
                src=excluded.src, relation=excluded.relation, dst=excluded.dst, \
                fact=excluded.fact, properties=excluded.properties, episodes=excluded.episodes, \
                valid_at=excluded.valid_at, invalid_at=excluded.invalid_at, \
                created_at=excluded.created_at, expired_at=excluded.expired_at, \
                reference_time=excluded.reference_time",
            params![
                new_fact.id,
                new_fact.src,
                new_fact.relation,
                new_fact.dst,
                new_fact.fact,
                properties,
                episodes,
                new_fact.valid_at,
                new_fact.invalid_at,
                new_fact.created_at,
                new_fact.expired_at,
                new_fact.reference_time,
            ],
        )?;
        tx.commit()?;
        Ok(closed_ids)
    }

    pub fn iter_facts(&self) -> Result<Vec<Fact>> {
        let sql = format!("SELECT {} FROM facts", FACT_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_fact(row)?);
        }
        Ok(out)
    }

    pub fn remove_fact(&self, id: &str) -> Result<bool> {
        let n = self
            .conn
            .execute("DELETE FROM facts WHERE id = ?", params![id])?;
        Ok(n > 0)
    }

    pub fn fact_count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM facts", [], |row| row.get(0))?;
        Ok(n as usize)
    }

    // -- traversal ----------------------------------------------------------

    /// Facts incident to any of `node_ids` in `direction`, optionally restricted to
    /// `relation`, in a temporal frame (`as_of`). The batched frontier expansion
    /// behind k-hop. Dedup by fact id across chunks.
    pub fn facts_for(
        &self,
        node_ids: &[String],
        direction: Direction,
        relation: Option<&str>,
        as_of: AsOf,
    ) -> Result<Vec<Fact>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let (as_of_frag, as_of_params) = crate::temporal::as_of_sql(as_of);

        // Dedup across chunks (a fact can match via `src` in one chunk and `dst`
        // in another when direction = both); ordering is by fact id.
        let mut found: BTreeMap<String, Fact> = BTreeMap::new();

        for chunk in node_ids.chunks(IN_CHUNK) {
            let placeholders = placeholders(chunk.len());
            let mut binds: Vec<SqlValue> = Vec::new();

            let dir_clause = match direction {
                Direction::Out => {
                    for id in chunk {
                        binds.push(SqlValue::Text(id.clone()));
                    }
                    format!("f.src IN ({})", placeholders)
                }
                Direction::In => {
                    for id in chunk {
                        binds.push(SqlValue::Text(id.clone()));
                    }
                    format!("f.dst IN ({})", placeholders)
                }
                Direction::Both => {
                    // Each id binds twice: once for src, once for dst.
                    for id in chunk {
                        binds.push(SqlValue::Text(id.clone()));
                    }
                    for id in chunk {
                        binds.push(SqlValue::Text(id.clone()));
                    }
                    format!("(f.src IN ({0}) OR f.dst IN ({0}))", placeholders)
                }
            };

            let mut where_parts = vec![dir_clause];
            if let Some(rel) = relation {
                where_parts.push("f.relation = ?".to_string());
                binds.push(SqlValue::Text(rel.to_string()));
            }
            where_parts.push(format!("({})", as_of_frag));
            for p in &as_of_params {
                binds.push(SqlValue::Integer(*p));
            }

            let sql = format!(
                "SELECT {} FROM facts f WHERE {}",
                FACT_COLS,
                where_parts.join(" AND ")
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(params_from_iter(binds.iter()))?;
            while let Some(row) = rows.next()? {
                let fact = row_to_fact(row)?;
                found.insert(fact.id.clone(), fact);
            }
        }
        Ok(found.into_values().collect())
    }

    /// Every fact ever touching an entity (both endpoints), all frames — history.
    pub fn history(&self, entity_id: &str) -> Result<Vec<Fact>> {
        let sql = format!(
            "SELECT {} FROM facts WHERE src = ? OR dst = ? ORDER BY created_at",
            FACT_COLS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![entity_id, entity_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_fact(row)?);
        }
        Ok(out)
    }
}

// -- helpers ----------------------------------------------------------------

/// A comma-separated run of `n` positional placeholders (`?, ?, ...`).
fn placeholders(n: usize) -> String {
    std::iter::repeat("?")
        .take(n)
        .collect::<Vec<_>>()
        .join(",")
}

/// Encode a `properties` value for the TEXT column, mapping the "absent" `Null`
/// default to an empty object `{}` (episodes use `serde_json::to_string` directly,
/// which already yields `[]` for an empty list).
fn props_to_text(value: &serde_json::Value) -> Result<String> {
    if value.is_null() {
        Ok("{}".to_string())
    } else {
        Ok(serde_json::to_string(value)?)
    }
}

fn row_to_entity(row: &Row<'_>) -> Result<Entity> {
    let properties: String = row.get("properties")?;
    Ok(Entity {
        id: row.get("id")?,
        entity_type: row.get("type")?,
        label: row.get("label")?,
        properties: serde_json::from_str(&properties)?,
        valid_at: row.get("valid_at")?,
        invalid_at: row.get("invalid_at")?,
        created_at: row.get("created_at")?,
        expired_at: row.get("expired_at")?,
    })
}

fn row_to_fact(row: &Row<'_>) -> Result<Fact> {
    let properties: String = row.get("properties")?;
    let episodes: String = row.get("episodes")?;
    Ok(Fact {
        id: row.get("id")?,
        src: row.get("src")?,
        relation: row.get("relation")?,
        dst: row.get("dst")?,
        fact: row.get("fact")?,
        properties: serde_json::from_str(&properties)?,
        episodes: serde_json::from_str(&episodes)?,
        valid_at: row.get("valid_at")?,
        invalid_at: row.get("invalid_at")?,
        created_at: row.get("created_at")?,
        expired_at: row.get("expired_at")?,
        reference_time: row.get("reference_time")?,
    })
}

fn row_to_episode(row: &Row<'_>) -> Result<Episode> {
    let properties: String = row.get("properties")?;
    Ok(Episode {
        id: row.get("id")?,
        source: row.get("source")?,
        body: row.get("body")?,
        occurred_at: row.get("occurred_at")?,
        created_at: row.get("created_at")?,
        properties: serde_json::from_str(&properties)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entity(id: &str) -> Entity {
        Entity {
            id: id.to_string(),
            entity_type: "person".to_string(),
            label: format!("label-{id}"),
            properties: json!({ "k": id }),
            valid_at: None,
            invalid_at: None,
            created_at: 100,
            expired_at: None,
        }
    }

    fn fact(id: &str, src: &str, relation: &str, dst: &str) -> Fact {
        Fact {
            id: id.to_string(),
            src: src.to_string(),
            relation: relation.to_string(),
            dst: dst.to_string(),
            fact: format!("fact-{id}"),
            properties: json!({ "p": id }),
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: 100,
            expired_at: None,
            reference_time: None,
        }
    }

    fn ids(facts: &[Fact]) -> Vec<String> {
        facts.iter().map(|f| f.id.clone()).collect()
    }

    #[test]
    fn roundtrip_entity_fact_episode() {
        let store = TopologyStore::open_in_memory().unwrap();

        let e = entity("a");
        store.upsert_entity(&e).unwrap();
        assert_eq!(store.get_entity("a").unwrap().unwrap(), e);
        assert_eq!(store.get_entity("nope").unwrap(), None);

        let f = Fact {
            episodes: vec!["ep1".to_string(), "ep2".to_string()],
            valid_at: Some(10),
            invalid_at: Some(20),
            expired_at: Some(30),
            reference_time: Some(5),
            ..fact("f1", "a", "knows", "b")
        };
        store.upsert_fact(&f).unwrap();
        assert_eq!(store.get_fact("f1").unwrap().unwrap(), f);
        assert_eq!(store.get_fact("nope").unwrap(), None);

        let ep = Episode {
            id: "ep1".to_string(),
            source: "message".to_string(),
            body: "hello world".to_string(),
            occurred_at: 50,
            created_at: 60,
            properties: json!({ "x": 1 }),
        };
        store.upsert_episode(&ep).unwrap();
        assert_eq!(store.get_episode("ep1").unwrap().unwrap(), ep);
        assert_eq!(store.get_episode("nope").unwrap(), None);

        assert_eq!(store.entity_count().unwrap(), 1);
        assert_eq!(store.fact_count().unwrap(), 1);

        // Upsert overwrites in place.
        let e2 = Entity {
            label: "renamed".to_string(),
            ..entity("a")
        };
        store.upsert_entity(&e2).unwrap();
        assert_eq!(store.get_entity("a").unwrap().unwrap().label, "renamed");
        assert_eq!(store.entity_count().unwrap(), 1);
    }

    #[test]
    fn empty_properties_default_roundtrips() {
        let store = TopologyStore::open_in_memory().unwrap();
        // The `Null` default persists as `{}` / `[]`; read back as an empty object
        // / empty list.
        let e = Entity {
            properties: serde_json::Value::Null,
            ..entity("a")
        };
        store.upsert_entity(&e).unwrap();
        let got = store.get_entity("a").unwrap().unwrap();
        assert_eq!(got.properties, json!({}));

        let f = Fact {
            properties: serde_json::Value::Null,
            episodes: vec![],
            ..fact("f1", "a", "r", "b")
        };
        store.upsert_fact(&f).unwrap();
        let got = store.get_fact("f1").unwrap().unwrap();
        assert_eq!(got.properties, json!({}));
        assert!(got.episodes.is_empty());
    }

    #[test]
    fn traversal_directions_and_relation() {
        let store = TopologyStore::open_in_memory().unwrap();
        for id in ["a", "b", "c"] {
            store.upsert_entity(&entity(id)).unwrap();
        }
        store.upsert_fact(&fact("ab", "a", "r", "b")).unwrap();
        store.upsert_fact(&fact("bc", "b", "r2", "c")).unwrap();

        // out: src in set
        let out_a = store
            .facts_for(&["a".to_string()], Direction::Out, None, AsOf::All)
            .unwrap();
        assert_eq!(ids(&out_a), vec!["ab"]);

        // in: dst in set
        let in_c = store
            .facts_for(&["c".to_string()], Direction::In, None, AsOf::All)
            .unwrap();
        assert_eq!(ids(&in_c), vec!["bc"]);

        // both: either endpoint (b touches both edges)
        let both_b = store
            .facts_for(&["b".to_string()], Direction::Both, None, AsOf::All)
            .unwrap();
        assert_eq!(ids(&both_b), vec!["ab", "bc"]);

        // relation filter
        let filt = store
            .facts_for(&["b".to_string()], Direction::Out, Some("r2"), AsOf::All)
            .unwrap();
        assert_eq!(ids(&filt), vec!["bc"]);
        let none = store
            .facts_for(&["b".to_string()], Direction::Out, Some("r"), AsOf::All)
            .unwrap();
        assert!(none.is_empty());

        // empty seed set short-circuits
        assert!(store
            .facts_for(&[], Direction::Both, None, AsOf::All)
            .unwrap()
            .is_empty());

        // history: every frame touching an entity, ordered by created_at
        assert_eq!(store.history("b").unwrap().len(), 2);
        assert_eq!(ids(&store.history("a").unwrap()), vec!["ab"]);
    }

    #[test]
    fn temporal_as_of_frames() {
        let store = TopologyStore::open_in_memory().unwrap();
        let seeds = vec!["a".to_string()];

        // f1 becomes true at 1000, still open.
        let f1 = Fact {
            valid_at: Some(1000),
            ..fact("f1", "a", "r", "b")
        };
        store.upsert_fact(&f1).unwrap();
        // f2 becomes true at 2000.
        let f2 = Fact {
            valid_at: Some(2000),
            ..fact("f2", "a", "r", "b")
        };
        store.upsert_fact(&f2).unwrap();

        // Close f1 at 2000 (event + transaction axis).
        assert!(store.close_fact("f1", Some(2000), 2000).unwrap());
        // Re-closing an already-expired fact is a no-op.
        assert!(!store.close_fact("f1", Some(2000), 2000).unwrap());
        // Closing a non-existent fact is a no-op.
        assert!(!store.close_fact("ghost", None, 1).unwrap());

        // Now: only the live fact (f1 is expired).
        let now = store
            .facts_for(&seeds, Direction::Out, None, AsOf::Now)
            .unwrap();
        assert_eq!(ids(&now), vec!["f2"]);

        // At(1500): f1 valid (1000<=1500, invalid 2000>1500); f2 not yet valid.
        let at1500 = store
            .facts_for(&seeds, Direction::Out, None, AsOf::At(1500))
            .unwrap();
        assert_eq!(ids(&at1500), vec!["f1"]);

        // At(2500): f2 valid; f1 already invalid.
        let at2500 = store
            .facts_for(&seeds, Direction::Out, None, AsOf::At(2500))
            .unwrap();
        assert_eq!(ids(&at2500), vec!["f2"]);

        // All: both frames.
        let all = store
            .facts_for(&seeds, Direction::Out, None, AsOf::All)
            .unwrap();
        assert_eq!(ids(&all), vec!["f1", "f2"]);
    }

    #[test]
    fn get_entities_and_list_entities() {
        let store = TopologyStore::open_in_memory().unwrap();
        store.upsert_entity(&entity("a")).unwrap();
        store.upsert_entity(&entity("b")).unwrap();
        store
            .upsert_entity(&Entity {
                entity_type: "org".to_string(),
                ..entity("c")
            })
            .unwrap();

        // batched read, missing ids omitted
        let got = store
            .get_entities(&["a".to_string(), "missing".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(got.len(), 2);
        assert!(store.get_entities(&[]).unwrap().is_empty());

        // list by type
        let people = store.list_entities(Some("person"), AsOf::Now).unwrap();
        assert_eq!(people.len(), 2);
        let orgs = store.list_entities(Some("org"), AsOf::Now).unwrap();
        assert_eq!(orgs.len(), 1);
        assert!(store
            .list_entities(Some("nope"), AsOf::Now)
            .unwrap()
            .is_empty());
        assert_eq!(store.list_entities(None, AsOf::All).unwrap().len(), 3);

        let mut all_ids = store.all_entity_ids().unwrap();
        all_ids.sort();
        assert_eq!(all_ids, vec!["a", "b", "c"]);
        assert_eq!(store.iter_entities().unwrap().len(), 3);
    }

    #[test]
    fn list_entities_temporal() {
        let store = TopologyStore::open_in_memory().unwrap();
        let e = Entity {
            valid_at: Some(1000),
            invalid_at: Some(2000),
            ..entity("a")
        };
        store.upsert_entity(&e).unwrap();

        // Now requires invalid_at IS NULL — this one has ended.
        assert!(store.list_entities(None, AsOf::Now).unwrap().is_empty());
        // At(1500): within [1000, 2000).
        assert_eq!(store.list_entities(None, AsOf::At(1500)).unwrap().len(), 1);
        // At(2500): after it ended.
        assert!(store.list_entities(None, AsOf::At(2500)).unwrap().is_empty());
        // All: no temporal filter.
        assert_eq!(store.list_entities(None, AsOf::All).unwrap().len(), 1);
    }

    #[test]
    fn remove_entity_and_fact() {
        let store = TopologyStore::open_in_memory().unwrap();
        for id in ["a", "b", "c"] {
            store.upsert_entity(&entity(id)).unwrap();
        }
        store.upsert_fact(&fact("ab", "a", "r", "b")).unwrap();
        store.upsert_fact(&fact("bc", "b", "r2", "c")).unwrap();

        // removing b drops both incident facts
        let (existed, mut removed) = store.remove_entity("b").unwrap();
        assert!(existed);
        removed.sort();
        assert_eq!(removed, vec!["ab", "bc"]);
        assert_eq!(store.fact_count().unwrap(), 0);
        assert_eq!(store.entity_count().unwrap(), 2);
        assert_eq!(store.get_entity("b").unwrap(), None);

        // removing again reports non-existence
        let (existed, removed) = store.remove_entity("b").unwrap();
        assert!(!existed);
        assert!(removed.is_empty());

        // single-fact removal
        store.upsert_fact(&fact("ac", "a", "r", "c")).unwrap();
        assert!(store.remove_fact("ac").unwrap());
        assert!(!store.remove_fact("ac").unwrap());
    }

    #[test]
    fn link_mentions_is_idempotent() {
        let store = TopologyStore::open_in_memory().unwrap();
        store.upsert_entity(&entity("a")).unwrap();
        store.upsert_entity(&entity("b")).unwrap();
        let ep = Episode {
            id: "ep1".to_string(),
            source: String::new(),
            body: String::new(),
            occurred_at: 1,
            created_at: 1,
            properties: serde_json::Value::Null,
        };
        store.upsert_episode(&ep).unwrap();

        store
            .link_mentions("ep1", &["a".to_string(), "b".to_string()])
            .unwrap();
        // Re-linking (with overlap) must not error — the join PK dedups.
        store
            .link_mentions("ep1", &["a".to_string()])
            .unwrap();
        // Empty list short-circuits.
        store.link_mentions("ep1", &[]).unwrap();
    }

    #[test]
    fn iter_facts_returns_all_frames() {
        let store = TopologyStore::open_in_memory().unwrap();
        store.upsert_fact(&fact("f1", "a", "r", "b")).unwrap();
        let f2 = Fact {
            expired_at: Some(99),
            invalid_at: Some(99),
            ..fact("f2", "a", "r", "b")
        };
        store.upsert_fact(&f2).unwrap();
        assert_eq!(store.iter_facts().unwrap().len(), 2);
    }
}
