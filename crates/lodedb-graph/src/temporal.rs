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

/// Width of the biased unsigned timestamp portion mirrored into index metadata.
/// An additional leading tag orders open-start < every i64 timestamp < open-end,
/// so the complete signed epoch-millisecond domain remains representable.
pub const TS_WIDTH: usize = 20;

// Keep every encoding deliberately non-numeric. The core metadata filter parses
// numeric-looking strings as f64 before ordered comparison; at epoch-sized
// magnitudes that would collapse nearby millisecond values through rounding.
const ACTUAL_TAG: char = 'b';
const OPEN_START: &str = "a00000000000000000000";
const OPEN_END: &str = "c00000000000000000000";

/// Encode any signed epoch-millisecond timestamp as a fixed-width string whose
/// lexical order is its numeric order. Flipping the sign bit maps
/// `i64::MIN..=i64::MAX` monotonically onto `0..=u64::MAX`; the `b` tag leaves room
/// for `a`/`c` open-start/open-end sentinels without sacrificing either endpoint.
/// The alphabetic tags also force the metadata filter to use exact lexical ordering.
pub fn encode_ts(ms: TimeMs) -> String {
    let biased = (ms as u64) ^ (1_u64 << 63);
    format!("{ACTUAL_TAG}{biased:0width$}", width = TS_WIDTH)
}

/// Encode an optional END endpoint (`invalid_at` / `expired_at`), mapping "open"
/// (still valid) above every possible i64 timestamp.
pub fn encode_ts_open(ms: Option<TimeMs>) -> String {
    ms.map(encode_ts).unwrap_or_else(|| OPEN_END.to_string())
}

/// Encode an optional START endpoint (`valid_at`), mapping "open" (unknown start) to
/// below every possible i64 timestamp. This matches the `f.valid_at IS NULL`
/// "started" semantics in [`as_of_sql`].
pub fn encode_ts_start(ms: Option<TimeMs>) -> String {
    ms.map(encode_ts).unwrap_or_else(|| OPEN_START.to_string())
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
        AsOf::NowValid(t) => (
            "(f.valid_at IS NULL OR f.valid_at <= ?) \
             AND (f.invalid_at IS NULL OR f.invalid_at > ?) \
             AND f.created_at <= ? \
             AND (f.expired_at IS NULL OR f.expired_at > ?)"
                .to_string(),
            vec![t, t, t, t],
        ),
        AsOf::At(t) => (
            "(f.valid_at IS NULL OR f.valid_at <= ?) AND (f.invalid_at IS NULL OR f.invalid_at > ?)"
                .to_string(),
            vec![t, t],
        ),
        AsOf::AtKnown { valid_at, known_at } => (
            "(f.valid_at IS NULL OR f.valid_at <= ?) \
             AND f.created_at <= ? \
             AND (f.expired_at IS NULL OR f.expired_at > ?) \
             AND (f.expired_at > ? OR f.invalid_at IS NULL OR f.invalid_at > ?)"
                .to_string(),
            vec![valid_at, known_at, known_at, known_at, valid_at],
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
    created_at: TimeMs,
    expired_at: Option<TimeMs>,
) -> bool {
    match as_of {
        AsOf::All => true,
        AsOf::Now => expired_at.is_none() && invalid_at.is_none(),
        AsOf::NowValid(t) => {
            valid_at.is_none_or(|v| v <= t)
                && invalid_at.is_none_or(|i| i > t)
                && created_at <= t
                && expired_at.is_none_or(|e| e > t)
        }
        AsOf::At(t) => valid_at.is_none_or(|v| v <= t) && invalid_at.is_none_or(|i| i > t),
        AsOf::AtKnown {
            valid_at: v,
            known_at: k,
        } => {
            valid_at.is_none_or(|start| start <= v)
                && created_at <= k
                && expired_at.is_none_or(|end| end > k)
                && (expired_at.is_some_and(|end| end > k) || invalid_at.is_none_or(|end| end > v))
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

#[cfg(test)]
mod tests {
    use super::{encode_ts, encode_ts_open, encode_ts_start};

    #[test]
    fn signed_timestamp_encoding_is_lexically_ordered() {
        let ordered = [
            encode_ts_start(None),
            encode_ts(i64::MIN),
            encode_ts(-2_000),
            encode_ts(-1),
            encode_ts(0),
            encode_ts(1),
            encode_ts(2_000),
            encode_ts(i64::MAX),
            encode_ts_open(None),
        ];
        assert!(ordered.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(ordered.iter().all(|encoded| encoded.len() == 21));
    }
}
