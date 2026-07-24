//! The SQL topology truth store: episodes, entities, typed facts, provenance, and
//! bi-temporal validity — the authoritative half of the graph.
//!
//! Port target: Graphiti's persistence in `graphiti_core/edges.py` +
//! `graphiti_core/nodes.py` (the `save` / `get_*` / traversal Cypher), and the
//! schema shape from LodeDB's own Python topology store
//! (`src/lodedb/graph/_store.py`), translated Cypher → SQL. rusqlite, bundled.
//!
//! Conventions: one transaction per mutation; `properties` and `episodes` are
//! JSON-encoded TEXT; temporal filtering shares `crate::temporal::as_of_sql`;
//! `IN (...)` lists are chunked under SQLite's bound-parameter limit as
//! `_store.py` does. Timestamps are `i64` epoch-ms columns; `NULL` = open
//! interval.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rusqlite::types::Value as SqlValue;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Row};

use crate::error::{GraphError, Result};
use crate::model::{
    AsOf, Direction, Entity, EntityPropertyVersion, Episode, Fact, GraphConfig, TimeMs,
};

/// DDL for the topology store.
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
CREATE TABLE IF NOT EXISTS entity_property_versions (
    version_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    property_key TEXT NOT NULL,
    value       TEXT NOT NULL,
    episode_id  TEXT REFERENCES episodes(id) ON DELETE SET NULL,
    valid_at    INTEGER,
    invalid_at  INTEGER,
    created_at  INTEGER NOT NULL,
    expired_at  INTEGER
);
CREATE TABLE IF NOT EXISTS entity_vectors (
    entity_id   TEXT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_vectors (
    fact_id     TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_src ON facts(src, relation);
CREATE INDEX IF NOT EXISTS idx_facts_dst ON facts(dst, relation);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_at, invalid_at);
CREATE INDEX IF NOT EXISTS idx_facts_live ON facts(expired_at);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON episode_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_fact_episodes_episode ON fact_episodes(episode_id, fact_id);
CREATE INDEX IF NOT EXISTS idx_property_versions_entity
    ON entity_property_versions(entity_id, property_key, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_property_versions_current
    ON entity_property_versions(entity_id, property_key)
    WHERE expired_at IS NULL;
"#;

/// Embedded SQL store of episodes/entities/facts with bi-temporal validity.
pub struct TopologyStore {
    #[allow(dead_code)]
    conn: Connection,
}

pub struct EntityIndexRecord {
    pub entity: Entity,
    pub vector: Option<Vec<f32>>,
}

pub struct FactIndexRecord {
    pub fact: Fact,
    pub vector: Option<Vec<f32>>,
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
const FACT_COLS_F: &str = "f.id AS id, f.src AS src, f.relation AS relation, \
     f.dst AS dst, f.fact AS fact, f.properties AS properties, f.episodes AS episodes, \
     f.valid_at AS valid_at, f.invalid_at AS invalid_at, f.created_at AS created_at, \
     f.expired_at AS expired_at, f.reference_time AS reference_time";
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
        let store = Self { conn };
        store.backfill_property_versions()?;
        Ok(store)
    }

    /// Persist the index-shaping graph configuration on first open and reject any
    /// later open that would reinterpret existing index contents.
    pub fn validate_configuration(&self, config: &GraphConfig) -> Result<()> {
        for (key, value) in expected_configuration(config) {
            let existing: Option<String> = self
                .conn
                .query_row(
                    "SELECT value FROM graph_meta WHERE key = ?",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(existing) = existing {
                if existing != value {
                    return Err(GraphError::InvalidArgument(format!(
                        "graph configuration mismatch for {key}: stored {existing}, requested {value}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn configure(&self, config: &GraphConfig) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        let had_format_version: bool = tx
            .query_row(
                "SELECT 1 FROM graph_meta WHERE key = 'format_version'",
                [],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();
        let had_index_metadata_version: bool = tx
            .query_row(
                "SELECT 1 FROM graph_meta WHERE key = 'index_metadata_version'",
                [],
                |_row| Ok(()),
            )
            .optional()?
            .is_some();
        for (key, value) in expected_configuration(config) {
            let existing: Option<String> = tx
                .query_row(
                    "SELECT value FROM graph_meta WHERE key = ?",
                    params![key],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(existing) = existing {
                if existing != value {
                    return Err(GraphError::InvalidArgument(format!(
                        "graph configuration mismatch for {key}: stored {existing}, requested {value}"
                    )));
                }
            } else {
                tx.execute(
                    "INSERT INTO graph_meta (key, value) VALUES (?, ?)",
                    params![key, value],
                )?;
            }
        }
        if had_index_metadata_version {
            tx.execute(
                "INSERT OR IGNORE INTO graph_meta (key, value) VALUES ('index_dirty', ?)",
                params![if had_format_version { "0" } else { "1" }],
            )?;
        } else {
            tx.execute(
                "INSERT INTO graph_meta (key, value) VALUES ('index_dirty', '1') \
                 ON CONFLICT(key) DO UPDATE SET value='1'",
                [],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn index_dirty(&self) -> Result<bool> {
        let value: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM graph_meta WHERE key = 'index_dirty'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value.is_some_and(|value| value == "1"))
    }

    pub fn mark_index_dirty(&self) -> Result<()> {
        self.set_index_dirty(true)
    }

    pub fn mark_index_clean(&self) -> Result<()> {
        self.set_index_dirty(false)
    }

    fn set_index_dirty(&self, dirty: bool) -> Result<()> {
        self.conn.execute(
            "INSERT INTO graph_meta (key, value) VALUES ('index_dirty', ?) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![bool_text(dirty)],
        )?;
        Ok(())
    }

    /// Seed lineage for stores created before the lineage table existed. This is
    /// idempotent and only fills a missing current version for a property that is
    /// present in the entity snapshot.
    fn backfill_property_versions(&self) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        let entities: Vec<(
            String,
            String,
            Option<TimeMs>,
            Option<TimeMs>,
            TimeMs,
            Option<TimeMs>,
        )> = {
            let mut stmt = tx.prepare(
                "SELECT id, properties, valid_at, invalid_at, created_at, expired_at FROM entities",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        for (entity_id, properties, valid_at, invalid_at, created_at, expired_at) in entities {
            let value: serde_json::Value = serde_json::from_str(&properties)?;
            let Some(object) = value.as_object() else {
                continue;
            };
            for (key, property_value) in object {
                tx.execute(
                    "INSERT INTO entity_property_versions \
                        (entity_id, property_key, value, episode_id, valid_at, invalid_at, \
                         created_at, expired_at) \
                     SELECT ?, ?, ?, NULL, ?, ?, ?, ? \
                     WHERE NOT EXISTS ( \
                         SELECT 1 FROM entity_property_versions \
                         WHERE entity_id = ? AND property_key = ? \
                     )",
                    params![
                        entity_id,
                        key,
                        serde_json::to_string(property_value)?,
                        valid_at,
                        invalid_at,
                        created_at,
                        expired_at,
                        entity_id,
                        key,
                    ],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // -- episodes -----------------------------------------------------------

    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        let sql = format!("SELECT {} FROM episodes WHERE id = ?", EPISODE_COLS);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_episode(row)?)),
            None => Ok(None),
        }
    }

    pub fn list_episodes(&self) -> Result<Vec<Episode>> {
        let sql = format!(
            "SELECT {} FROM episodes ORDER BY created_at, id",
            EPISODE_COLS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_episode(row)?);
        }
        Ok(out)
    }

    pub fn episode_mentions(&self, id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT entity_id FROM episode_mentions WHERE episode_id = ? ORDER BY entity_id",
        )?;
        let mentions = stmt
            .query_map(params![id], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(mentions)
    }

    /// Insert (or replace) an episode and its entity mentions (Graphiti
    /// `MENTIONS`) in ONE transaction,
    /// so a bad mention id can never leave a half-written episode behind. Every
    /// mentioned entity must already exist; missing ids fail the whole call with
    /// `NotFound` and nothing persists.
    pub fn upsert_episode_with_mentions(
        &self,
        episode: &Episode,
        entity_ids: &[String],
    ) -> Result<()> {
        let properties = props_to_text(&episode.properties)?;
        let tx = self.conn.unchecked_transaction()?;
        let missing = missing_ids(&tx, "entities", entity_ids)?;
        if !missing.is_empty() {
            return Err(GraphError::NotFound(format!(
                "mentioned entities do not exist: {}",
                missing.join(", ")
            )));
        }
        tx.execute(
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
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO episode_mentions (episode_id, entity_id) VALUES (?, ?)",
            )?;
            for entity_id in entity_ids {
                stmt.execute(params![episode.id, entity_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // -- entities -----------------------------------------------------------

    #[cfg(test)]
    pub fn upsert_entity(&self, entity: &Entity) -> Result<()> {
        self.upsert_entity_before_commit(entity, None, || Ok(()))
    }

    /// Upsert an entity and its optional authoritative vector, invoking
    /// `before_commit` while the SQL transaction is still rollbackable. The
    /// semantic index uses this seam so an index failure cannot leave a topology
    /// mutation committed even though the public call returned an error.
    #[allow(dead_code)]
    pub fn upsert_entity_before_commit<F>(
        &self,
        entity: &Entity,
        vector: Option<&[f32]>,
        before_commit: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        self.upsert_entity_with_lineage_before_commit(
            entity,
            vector,
            &BTreeMap::new(),
            entity.created_at,
            before_commit,
        )
    }

    /// Upsert an entity snapshot and independently version each changed top-level
    /// property. `property_sources` maps property names to the episode that
    /// supplied that value.
    pub fn upsert_entity_with_lineage_before_commit<F>(
        &self,
        entity: &Entity,
        vector: Option<&[f32]>,
        property_sources: &BTreeMap<String, String>,
        recorded_at: TimeMs,
        before_commit: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        let properties = props_to_text(&entity.properties)?;
        let tx = self.conn.unchecked_transaction()?;
        let existing_properties: Option<String> = tx
            .query_row(
                "SELECT properties FROM entities WHERE id = ?",
                params![entity.id],
                |row| row.get(0),
            )
            .optional()?;
        let source_ids: Vec<String> = property_sources.values().cloned().collect();
        let missing = missing_ids(&tx, "episodes", &source_ids)?;
        if !missing.is_empty() {
            return Err(GraphError::NotFound(format!(
                "property source episodes do not exist: {}",
                missing.join(", ")
            )));
        }
        tx.execute(
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
        sync_property_versions(
            &tx,
            &entity.id,
            existing_properties.as_deref(),
            &entity.properties,
            property_sources,
            entity.valid_at,
            recorded_at,
        )?;
        store_vector(&tx, "entity_vectors", "entity_id", &entity.id, vector)?;
        before_commit()?;
        tx.commit()?;
        Ok(())
    }

    pub fn entity_property_history(
        &self,
        entity_id: &str,
        key: Option<&str>,
    ) -> Result<Vec<EntityPropertyVersion>> {
        let (sql, binds): (String, Vec<SqlValue>) = match key {
            Some(key) => (
                "SELECT entity_id, property_key, value, episode_id, valid_at, invalid_at, \
                        created_at, expired_at \
                 FROM entity_property_versions \
                 WHERE entity_id = ? AND property_key = ? \
                 ORDER BY created_at, version_id"
                    .to_string(),
                vec![
                    SqlValue::Text(entity_id.to_string()),
                    SqlValue::Text(key.to_string()),
                ],
            ),
            None => (
                "SELECT entity_id, property_key, value, episode_id, valid_at, invalid_at, \
                        created_at, expired_at \
                 FROM entity_property_versions \
                 WHERE entity_id = ? ORDER BY property_key, created_at, version_id"
                    .to_string(),
                vec![SqlValue::Text(entity_id.to_string())],
            ),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(binds.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_property_version(row)?);
        }
        Ok(out)
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
            AsOf::NowValid(t) => {
                where_parts.push(
                    "((e.valid_at IS NULL OR e.valid_at <= ?) \
                      AND (e.invalid_at IS NULL OR e.invalid_at > ?) \
                      AND e.created_at <= ? \
                      AND (e.expired_at IS NULL OR e.expired_at > ?))"
                        .to_string(),
                );
                for _ in 0..4 {
                    binds.push(SqlValue::Integer(t));
                }
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
            AsOf::AtKnown { valid_at, known_at } => {
                where_parts.push(
                    "((e.valid_at IS NULL OR e.valid_at <= ?) \
                      AND e.created_at <= ? \
                      AND (e.expired_at IS NULL OR e.expired_at > ?) \
                      AND (e.expired_at > ? OR e.invalid_at IS NULL OR e.invalid_at > ?))"
                        .to_string(),
                );
                binds.push(SqlValue::Integer(valid_at));
                binds.push(SqlValue::Integer(known_at));
                binds.push(SqlValue::Integer(known_at));
                binds.push(SqlValue::Integer(known_at));
                binds.push(SqlValue::Integer(valid_at));
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

    /// Read every entity and retained caller-supplied vector in one scan for
    /// reindexing. Keeping the join here avoids an N+1 SQLite query per record.
    pub fn iter_entity_index_records(&self) -> Result<Vec<EntityIndexRecord>> {
        let sql = format!(
            "SELECT {}, v.embedding AS embedding \
             FROM entities e LEFT JOIN entity_vectors v ON v.entity_id = e.id",
            ENTITY_COLS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let bytes: Option<Vec<u8>> = row.get("embedding")?;
            out.push(EntityIndexRecord {
                entity: row_to_entity(row)?,
                vector: bytes.as_deref().map(vector_from_bytes).transpose()?,
            });
        }
        Ok(out)
    }

    /// Remove an entity and its incident facts; returns `(existed, removed_fact_ids)`.
    #[cfg(test)]
    pub fn remove_entity(&self, id: &str) -> Result<(bool, Vec<String>)> {
        self.remove_entity_before_commit(id, |_removed| Ok(()))
    }

    pub fn remove_entity_before_commit<F>(
        &self,
        id: &str,
        before_commit: F,
    ) -> Result<(bool, Vec<String>)>
    where
        F: FnOnce(&[String]) -> Result<()>,
    {
        let tx = self.conn.unchecked_transaction()?;
        let removed: Vec<String> = {
            let mut stmt = tx.prepare("SELECT id FROM facts WHERE src = ? OR dst = ?")?;
            let ids = stmt
                .query_map(params![id, id], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<String>>>()?;
            ids
        };
        tx.execute(
            "DELETE FROM facts WHERE src = ? OR dst = ?",
            params![id, id],
        )?;
        // episode_mentions rows for this entity cascade via the FK.
        let removed_entity = tx.execute("DELETE FROM entities WHERE id = ?", params![id])?;
        before_commit(&removed)?;
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

    /// Test fixture: raw fact insert/replace that bypasses the facade's boundary
    /// checks (entity/episode existence). Facade writes go through
    /// `supersede_and_insert`.
    #[cfg(test)]
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

    pub fn facts_by_episode(&self, episode_id: &str) -> Result<Vec<Fact>> {
        let sql = format!(
            "SELECT {} FROM facts f \
             INNER JOIN fact_episodes p ON p.fact_id = f.id \
             WHERE p.episode_id = ? ORDER BY f.created_at, f.id",
            FACT_COLS_F
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![episode_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_fact(row)?);
        }
        Ok(out)
    }

    /// Remove an episode and roll back facts it originally created. Facts for
    /// which the episode was only additional support are retained with that
    /// provenance link removed.
    pub fn remove_episode_before_commit<F>(&self, id: &str, before_commit: F) -> Result<bool>
    where
        F: FnOnce(&[String]) -> Result<()>,
    {
        let tx = self.conn.unchecked_transaction()?;
        let existed: bool = tx
            .query_row("SELECT 1 FROM episodes WHERE id = ?", params![id], |_row| {
                Ok(())
            })
            .optional()?
            .is_some();
        if !existed {
            before_commit(&[])?;
            tx.commit()?;
            return Ok(false);
        }

        let linked: Vec<Fact> = {
            let sql = format!(
                "SELECT {} FROM facts f \
                 INNER JOIN fact_episodes p ON p.fact_id = f.id \
                 WHERE p.episode_id = ?",
                FACT_COLS_F
            );
            let mut stmt = tx.prepare(&sql)?;
            let mut rows = stmt.query(params![id])?;
            let mut facts = Vec::new();
            while let Some(row) = rows.next()? {
                facts.push(row_to_fact(row)?);
            }
            facts
        };
        let mut removed = Vec::new();
        for mut fact in linked {
            if fact.episodes.first().is_some_and(|episode| episode == id) {
                tx.execute("DELETE FROM facts WHERE id = ?", params![fact.id])?;
                removed.push(fact.id);
            } else {
                fact.episodes.retain(|episode| episode != id);
                tx.execute(
                    "UPDATE facts SET episodes = ? WHERE id = ?",
                    params![serde_json::to_string(&fact.episodes)?, fact.id],
                )?;
                tx.execute(
                    "DELETE FROM fact_episodes WHERE fact_id = ? AND episode_id = ?",
                    params![fact.id, id],
                )?;
            }
        }
        tx.execute("DELETE FROM episodes WHERE id = ?", params![id])?;
        before_commit(&removed)?;
        tx.commit()?;
        Ok(true)
    }

    /// Batched fact hydration (chunked `IN`), missing ids omitted.
    pub fn get_facts(&self, ids: &[String]) -> Result<Vec<Fact>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        for chunk in ids.chunks(IN_CHUNK) {
            let sql = format!(
                "SELECT {} FROM facts WHERE id IN ({})",
                FACT_COLS,
                placeholders(chunk.len())
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(params_from_iter(chunk.iter()))?;
            while let Some(row) = rows.next()? {
                found.push(row_to_fact(row)?);
            }
        }
        Ok(found)
    }

    /// Close a fact's validity (the invalidation write): set `invalid_at` and
    /// `expired_at`. Returns whether the fact existed and was live.
    #[cfg(test)]
    pub fn close_fact(
        &self,
        id: &str,
        invalid_at: Option<TimeMs>,
        expired_at: TimeMs,
    ) -> Result<bool> {
        self.close_fact_before_commit(id, invalid_at, expired_at, |_closed| Ok(()))
    }

    pub fn close_fact_before_commit<F>(
        &self,
        id: &str,
        invalid_at: Option<TimeMs>,
        expired_at: TimeMs,
        before_commit: F,
    ) -> Result<bool>
    where
        F: FnOnce(Option<&FactIndexRecord>) -> Result<()>,
    {
        let tx = self.conn.unchecked_transaction()?;
        // Only close a fact that is still live on the transaction axis; never
        // re-close an already-expired one.
        let n = tx.execute(
            "UPDATE facts SET invalid_at = ?, expired_at = ? WHERE id = ? AND expired_at IS NULL",
            params![invalid_at, expired_at, id],
        )?;
        let closed = if n > 0 {
            let sql = format!("SELECT {} FROM facts WHERE id = ?", FACT_COLS);
            let fact = {
                let mut stmt = tx.prepare(&sql)?;
                let mut rows = stmt.query(params![id])?;
                match rows.next()? {
                    Some(row) => Some(row_to_fact(row)?),
                    None => None,
                }
            };
            match fact {
                Some(fact) => Some(FactIndexRecord {
                    vector: load_vector(&tx, "fact_vectors", "fact_id", id)?,
                    fact,
                }),
                None => None,
            }
        } else {
            None
        };
        before_commit(closed.as_ref())?;
        tx.commit()?;
        Ok(n > 0)
    }

    /// Atomically close `priors` (the invalidation writes) and insert `new_fact`
    /// (the replacement) in one transaction, invoking `before_commit` while every
    /// topology change is still rollbackable.
    pub fn supersede_and_insert_before_commit<F>(
        &self,
        priors: &[(String, Option<TimeMs>, TimeMs)],
        new_fact: &Fact,
        vector: Option<&[f32]>,
        before_commit: F,
    ) -> Result<Vec<String>>
    where
        F: FnOnce(&[FactIndexRecord]) -> Result<()>,
    {
        let properties = props_to_text(&new_fact.properties)?;
        let episodes = serde_json::to_string(&new_fact.episodes)?;
        let tx = self.conn.unchecked_transaction()?;

        // Boundary checks first, all inside the transaction, so any failure rolls
        // the whole assertion back. Endpoints must be existing entities (this store
        // does not auto-create nodes; resolution is the caller's job), provenance
        // must reference existing episodes, and every prior the caller names must
        // actually close — a typo'd id silently leaving the prior live would defeat
        // the invalidation semantics.
        let endpoints = [new_fact.src.clone(), new_fact.dst.clone()];
        let missing = missing_ids(&tx, "entities", &endpoints)?;
        if !missing.is_empty() {
            return Err(GraphError::NotFound(format!(
                "fact endpoints do not exist as entities: {}",
                missing.join(", ")
            )));
        }
        let missing = missing_ids(&tx, "episodes", &new_fact.episodes)?;
        if !missing.is_empty() {
            return Err(GraphError::NotFound(format!(
                "fact provenance episodes do not exist: {}",
                missing.join(", ")
            )));
        }
        let mut closed_ids = Vec::new();
        {
            let mut upd = tx.prepare(
                "UPDATE facts SET invalid_at = ?, expired_at = ? WHERE id = ? AND expired_at IS NULL",
            )?;
            for (id, invalid_at, expired_at) in priors {
                if upd.execute(params![invalid_at, expired_at, id])? > 0 {
                    closed_ids.push(id.clone());
                } else {
                    return Err(GraphError::NotFound(format!(
                        "fact to invalidate does not exist or is already expired: {id}"
                    )));
                }
            }
        }
        // Plain INSERT: the id is freshly generated, so a collision is a bug and
        // must fail loudly instead of silently overwriting a prior fact's history
        // (`upsert_fact` exists for deliberate replacement).
        tx.execute(
            "INSERT INTO facts \
                (id, src, relation, dst, fact, properties, episodes, \
                 valid_at, invalid_at, created_at, expired_at, reference_time) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO fact_episodes (fact_id, episode_id) VALUES (?, ?)",
            )?;
            for episode_id in &new_fact.episodes {
                stmt.execute(params![new_fact.id, episode_id])?;
            }
        }
        store_vector(&tx, "fact_vectors", "fact_id", &new_fact.id, vector)?;
        let mut closed_facts = Vec::with_capacity(closed_ids.len());
        if !closed_ids.is_empty() {
            let sql = format!("SELECT {} FROM facts WHERE id = ?", FACT_COLS);
            let mut stmt = tx.prepare(&sql)?;
            for id in &closed_ids {
                let fact = {
                    let mut rows = stmt.query(params![id])?;
                    let row = rows.next()?.ok_or_else(|| {
                        GraphError::Internal(format!(
                            "closed fact disappeared in transaction: {id}"
                        ))
                    })?;
                    row_to_fact(row)?
                };
                closed_facts.push(FactIndexRecord {
                    vector: load_vector(&tx, "fact_vectors", "fact_id", id)?,
                    fact,
                });
            }
        }
        before_commit(&closed_facts)?;
        tx.commit()?;
        Ok(closed_ids)
    }

    /// Read every fact and retained caller-supplied vector in one scan for
    /// reindexing. Keeping the join here avoids an N+1 SQLite query per record.
    pub fn iter_fact_index_records(&self) -> Result<Vec<FactIndexRecord>> {
        let sql = format!(
            "SELECT {}, v.embedding AS embedding \
             FROM facts f LEFT JOIN fact_vectors v ON v.fact_id = f.id",
            FACT_COLS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let bytes: Option<Vec<u8>> = row.get("embedding")?;
            out.push(FactIndexRecord {
                fact: row_to_fact(row)?,
                vector: bytes.as_deref().map(vector_from_bytes).transpose()?,
            });
        }
        Ok(out)
    }

    #[cfg(test)]
    pub fn remove_fact(&self, id: &str) -> Result<bool> {
        self.remove_fact_before_commit(id, || Ok(()))
    }

    pub fn remove_fact_before_commit<F>(&self, id: &str, before_commit: F) -> Result<bool>
    where
        F: FnOnce() -> Result<()>,
    {
        let tx = self.conn.unchecked_transaction()?;
        let n = tx.execute("DELETE FROM facts WHERE id = ?", params![id])?;
        before_commit()?;
        tx.commit()?;
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

fn property_object(value: &serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    value.as_object().cloned().unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn sync_property_versions(
    conn: &Connection,
    entity_id: &str,
    existing_properties: Option<&str>,
    new_properties: &serde_json::Value,
    property_sources: &BTreeMap<String, String>,
    new_valid_at: Option<TimeMs>,
    recorded_at: TimeMs,
) -> Result<()> {
    let old_value = existing_properties
        .map(serde_json::from_str)
        .transpose()?
        .unwrap_or(serde_json::Value::Null);
    let old = property_object(&old_value);
    let new = property_object(new_properties);
    for key in property_sources.keys() {
        if !new.contains_key(key) {
            return Err(GraphError::InvalidArgument(format!(
                "property source was supplied for missing property {key:?}"
            )));
        }
    }

    let keys: BTreeSet<String> = old.keys().chain(new.keys()).cloned().collect();
    for key in keys {
        let old_value = old.get(&key);
        let new_value = new.get(&key);
        if old_value == new_value {
            continue;
        }
        conn.execute(
            "UPDATE entity_property_versions \
             SET invalid_at = ?, expired_at = ? \
             WHERE entity_id = ? AND property_key = ? AND expired_at IS NULL",
            params![
                new_valid_at.unwrap_or(recorded_at),
                recorded_at,
                entity_id,
                key,
            ],
        )?;
        if let Some(value) = new_value {
            conn.execute(
                "INSERT INTO entity_property_versions \
                    (entity_id, property_key, value, episode_id, valid_at, invalid_at, \
                     created_at, expired_at) \
                 VALUES (?, ?, ?, ?, ?, NULL, ?, NULL)",
                params![
                    entity_id,
                    key,
                    serde_json::to_string(value)?,
                    property_sources.get(&key),
                    new_valid_at,
                    recorded_at,
                ],
            )?;
        }
    }
    Ok(())
}

fn bool_text(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn expected_configuration(config: &GraphConfig) -> [(&'static str, String); 5] {
    [
        ("format_version", "2".to_string()),
        ("index_metadata_version", "2".to_string()),
        ("vector_dim", config.vector_dim.to_string()),
        ("index_text", bool_text(config.index_text).to_string()),
        ("index_facts", bool_text(config.index_facts).to_string()),
    ]
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector));
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn vector_from_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(GraphError::Internal(
            "stored graph vector has an invalid byte length".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn store_vector(
    conn: &Connection,
    table: &str,
    id_column: &str,
    id: &str,
    vector: Option<&[f32]>,
) -> Result<()> {
    if let Some(vector) = vector {
        let sql = format!(
            "INSERT INTO {table} ({id_column}, embedding) VALUES (?, ?) \
             ON CONFLICT({id_column}) DO UPDATE SET embedding=excluded.embedding"
        );
        conn.execute(&sql, params![id, vector_bytes(vector)])?;
    } else {
        let sql = format!("DELETE FROM {table} WHERE {id_column} = ?");
        conn.execute(&sql, params![id])?;
    }
    Ok(())
}

fn load_vector(
    conn: &Connection,
    table: &str,
    id_column: &str,
    id: &str,
) -> Result<Option<Vec<f32>>> {
    let sql = format!("SELECT embedding FROM {table} WHERE {id_column} = ?");
    let bytes: Option<Vec<u8>> = conn
        .query_row(&sql, params![id], |row| row.get(0))
        .optional()?;
    bytes.as_deref().map(vector_from_bytes).transpose()
}

/// A comma-separated run of `n` positional placeholders (`?, ?, ...`).
/// The subset of `ids` with no row in `table` (deduplicated), chunked under the
/// bound-parameter limit. `table` is always a literal from this module, never
/// caller input.
fn missing_ids(conn: &Connection, table: &str, ids: &[String]) -> Result<Vec<String>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut missing: std::collections::BTreeSet<String> = ids.iter().cloned().collect();
    let unique: Vec<String> = missing.iter().cloned().collect();
    for chunk in unique.chunks(IN_CHUNK) {
        let sql = format!(
            "SELECT id FROM {} WHERE id IN ({})",
            table,
            placeholders(chunk.len())
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(chunk.iter()))?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            missing.remove(&id);
        }
    }
    Ok(missing.into_iter().collect())
}

fn placeholders(n: usize) -> String {
    std::iter::repeat("?").take(n).collect::<Vec<_>>().join(",")
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

fn row_to_property_version(row: &Row<'_>) -> Result<EntityPropertyVersion> {
    let value: String = row.get("value")?;
    Ok(EntityPropertyVersion {
        entity_id: row.get("entity_id")?,
        key: row.get("property_key")?,
        value: serde_json::from_str(&value)?,
        episode_id: row.get("episode_id")?,
        valid_at: row.get("valid_at")?,
        invalid_at: row.get("invalid_at")?,
        created_at: row.get("created_at")?,
        expired_at: row.get("expired_at")?,
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
        assert_eq!(
            store
                .get_facts(&["f1".to_string(), "nope".to_string()])
                .unwrap(),
            vec![f.clone()]
        );
        assert!(store.get_facts(&[]).unwrap().is_empty());

        let ep = Episode {
            id: "ep1".to_string(),
            source: "message".to_string(),
            body: "hello world".to_string(),
            occurred_at: 50,
            created_at: 60,
            properties: json!({ "x": 1 }),
        };
        store.upsert_episode_with_mentions(&ep, &[]).unwrap();
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
        assert_eq!(store.iter_entity_index_records().unwrap().len(), 3);
    }

    #[test]
    fn missing_index_metadata_version_forces_a_rebuild() {
        let store = TopologyStore::open_in_memory().unwrap();
        let config = GraphConfig::default();
        store.configure(&config).unwrap();
        store.mark_index_clean().unwrap();
        store
            .conn
            .execute(
                "DELETE FROM graph_meta WHERE key = 'index_metadata_version'",
                [],
            )
            .unwrap();

        store.configure(&config).unwrap();

        assert!(store.index_dirty().unwrap());
        let version: String = store
            .conn
            .query_row(
                "SELECT value FROM graph_meta WHERE key = 'index_metadata_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, "2");
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
        assert!(store
            .list_entities(None, AsOf::At(2500))
            .unwrap()
            .is_empty());
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
    fn episode_with_bad_mention_rolls_back() {
        let store = TopologyStore::open_in_memory().unwrap();
        store.upsert_entity(&entity("a")).unwrap();
        let ep = Episode {
            id: "ep1".to_string(),
            source: String::new(),
            body: String::new(),
            occurred_at: 1,
            created_at: 1,
            properties: serde_json::Value::Null,
        };
        // "ghost" does not exist: the whole upsert must roll back, leaving no
        // half-written episode row behind.
        let err = store.upsert_episode_with_mentions(&ep, &["a".to_string(), "ghost".to_string()]);
        assert!(err.is_err());
        assert!(
            store.get_episode("ep1").unwrap().is_none(),
            "episode row rolled back"
        );

        store
            .upsert_episode_with_mentions(&ep, &["a".to_string()])
            .unwrap();
        assert!(store.get_episode("ep1").unwrap().is_some());
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
        store.upsert_episode_with_mentions(&ep, &[]).unwrap();

        store
            .upsert_episode_with_mentions(&ep, &["a".to_string(), "b".to_string()])
            .unwrap();
        // Re-linking (with overlap) must not error — the join PK dedups.
        store
            .upsert_episode_with_mentions(&ep, &["a".to_string()])
            .unwrap();
        // Empty list short-circuits.
        store.upsert_episode_with_mentions(&ep, &[]).unwrap();
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
        assert_eq!(store.iter_fact_index_records().unwrap().len(), 2);
    }
}
