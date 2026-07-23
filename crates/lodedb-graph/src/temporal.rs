//! Bi-temporal helpers: as-of predicates, the invalidation state transition, and
//! the sortable-string encoding that lets an as-of query run as a `lodedb-core`
//! metadata filter.
//!
//! Port target: Graphiti's edge-invalidation logic in
//! `graphiti_core/utils/maintenance/edge_operations.py`
//! (`resolve_extracted_edge` / contradiction handling) — the *deterministic state
//! transition* only. The contradiction *detection* is an LLM step and stays with
//! the caller; here we take the set of prior facts to close and compute the new
//! timestamps exactly as Graphiti does: close the prior edge's `invalid_at` at the
//! new fact's `valid_at`, and stamp `expired_at = now`.

use crate::model::{AsOf, TimeMs};

/// Width of the zero-padded epoch-millis string mirrored into index metadata.
/// Wide enough that lexical ordering equals numeric ordering for all realistic
/// (and far-future) timestamps, and for the sentinel below.
pub const TS_WIDTH: usize = 20;

/// Sentinel used for an open `invalid_at`/`expired_at` in the index mirror, so a
/// `$gt`/`$lte` range filter treats "still valid" as "valid arbitrarily far out".
pub const TS_OPEN: TimeMs = 9_223_372_036_854_775_000;

/// Encode a timestamp as a fixed-width, zero-padded string that sorts lexically in
/// the same order as it sorts numerically (LodeDB compares metadata numerically
/// when both sides parse as numbers, and this keeps parity for filters that don't).
///
/// LIMITATION — negative (pre-1970) timestamps are clamped to 0 here, whereas the SQL
/// topology ([`as_of_sql`]) compares the raw signed value. The truth store is therefore
/// authoritative and correct for pre-1970 dates (a 1965 birthday, a historical event);
/// only *time-scoped semantic* reads (`semantic_facts` / `search_subgraph` with an
/// `AsOf::At` instant before 1970) can diverge from the SQL as-of. A full fix — an
/// order-preserving signed encoding that biases the whole i64 range into the padded
/// space (and reworks [`TS_OPEN`] to stay the max without overflowing i64) — changes the
/// on-disk index format and is deferred to a focused change with LodeDB-internal checks.
pub fn encode_ts(ms: TimeMs) -> String {
    // Clamp negatives to 0 (see LIMITATION above); keeps width stable for the common,
    // non-negative range. The SQL truth store remains correct for negative timestamps.
    let v = if ms < 0 { 0 } else { ms };
    format!("{v:0width$}", width = TS_WIDTH)
}

/// Encode an optional END endpoint (`invalid_at` / `expired_at`), mapping "open"
/// (still valid) to [`TS_OPEN`] so an `invalid_at > T` range filter treats it as
/// valid arbitrarily far into the future.
pub fn encode_ts_open(ms: Option<TimeMs>) -> String {
    encode_ts(ms.unwrap_or(TS_OPEN))
}

/// Encode an optional START endpoint (`valid_at`), mapping "open" (unknown start) to
/// the epoch floor so a `valid_at <= T` range filter treats it as having started
/// arbitrarily far in the past. This matches the `f.valid_at IS NULL` "started"
/// semantics in [`as_of_sql`], so the semantic index and the SQL topology agree on
/// as-of for open-start records.
pub fn encode_ts_start(ms: Option<TimeMs>) -> String {
    encode_ts(ms.unwrap_or(0))
}

/// The SQL `WHERE` fragment (over a `facts` alias `f`) for a temporal frame, plus
/// the positional params it binds. Kept here so the topology store and any future
/// query builder share one definition.
///
/// Returns `(sql_fragment, params)` where `params` are bound in order after any
/// earlier params. Implemented by the topology port; see `topology.rs`.
pub fn as_of_sql(as_of: AsOf) -> (String, Vec<TimeMs>) {
    match as_of {
        AsOf::All => ("1=1".to_string(), vec![]),
        AsOf::Now => (
            "(f.expired_at IS NULL AND f.invalid_at IS NULL)".to_string(),
            vec![],
        ),
        AsOf::At(t) => (
            "(f.valid_at IS NULL OR f.valid_at <= ?) AND (f.invalid_at IS NULL OR f.invalid_at > ?)"
                .to_string(),
            vec![t, t],
        ),
    }
}

/// Whether a record's temporal endpoints satisfy a frame: the in-memory twin of
/// [`as_of_sql`], kept adjacent so the two definitions cannot drift. Hydration
/// paths re-check semantic-index hits against the authoritative topology row with
/// this, so a stale index (a crash between the topology commit and the index
/// refresh) cannot leak an expired record into a scoped read.
pub fn frame_matches(
    as_of: AsOf,
    valid_at: Option<TimeMs>,
    invalid_at: Option<TimeMs>,
    expired_at: Option<TimeMs>,
) -> bool {
    match as_of {
        AsOf::All => true,
        AsOf::Now => expired_at.is_none() && invalid_at.is_none(),
        AsOf::At(t) => {
            valid_at.is_none_or(|v| v <= t) && invalid_at.is_none_or(|i| i > t)
        }
    }
}

/// The timestamps to write when a new fact supersedes a prior one — Graphiti's
/// rule. The prior fact's event-time end (`invalid_at`) is the new fact's `valid_at`
/// when known, otherwise the new fact's `effective_time` (its `reference_time`, i.e.
/// the observing episode's event time). It is NEVER left open: were it left `None`, an
/// `AsOf::At(t)` read would count the prior and its replacement as simultaneously valid
/// (a double-count), disagreeing with the `AsOf::Now` view. Its transaction-time end
/// (`expired_at`) is `now`.
///
/// Returns `(invalid_at, expired_at)` to apply to the prior fact.
pub fn supersede_timestamps(
    new_fact_valid_at: Option<TimeMs>,
    effective_time: TimeMs,
    now: TimeMs,
) -> (Option<TimeMs>, TimeMs) {
    (Some(new_fact_valid_at.unwrap_or(effective_time)), now)
}
