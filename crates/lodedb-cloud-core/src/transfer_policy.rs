//! Which payload-bearing stores a transfer is allowed to ship.
//!
//! The redacted stores (`json`/`tvim`/`tvmv`/`tvann`/`tvvf`) carry no raw text and always ship.
//! They are the metadata and the vector/late-interaction index a restored copy
//! needs to answer searches. Two stores are payload-bearing and opt-in:
//!
//! - `tvtext`: the raw document text (`db.get(id)` content);
//! - `tvlex`: lexical terms, which are tokenised text and so payload-derived.
//!
//! A [`TransferPolicy`] gates those two. `tvmv` (late-interaction patch matrices)
//! is embedding data, not text, so it ships by default like `tvim`, as does
//! `tvvf`, the rescore original-vector sidecar (vectors, never text).
//!
//! Redaction rewrites the *committed body* rather than merely skipping bytes: a
//! redacted push publishes a body whose excluded sub-manifests are null, so the
//! remote generation genuinely has no text and a restore of it cannot resurrect
//! text that was never uploaded.

use serde_json::Value;

/// Whether a transfer ships the payload-bearing text and lexical stores.
///
/// [`Default`] is the redacted posture (both off), so by default only redacted
/// artifacts leave the machine. Every transfer states its
/// policy explicitly at the call site; pass [`TransferPolicy::full`] to ship a
/// generation verbatim (e.g. when restoring a backup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferPolicy {
    /// Ship the raw-text store (`tvtext` base + `.txd` deltas).
    pub include_text: bool,
    /// Ship the lexical-index store (`tvlex` base + `.lxd` deltas).
    pub include_lexical: bool,
}

impl TransferPolicy {
    /// Ships every store, including text and lexical, for a verbatim copy of
    /// the committed generation.
    pub fn full() -> Self {
        Self {
            include_text: true,
            include_lexical: true,
        }
    }

    /// Ships only the redacted stores (no text, no lexical), the default posture.
    pub fn redacted() -> Self {
        Self {
            include_text: false,
            include_lexical: false,
        }
    }

    /// Returns a copy of a committed body with the excluded stores nulled.
    ///
    /// Nulling a top-level store key reproduces exactly what
    /// `build_commit_body` emits for an absent store, and the body checksum is
    /// recomputed when the pointer is written, so the result is a valid,
    /// self-consistent committed body describing a generation that omits those
    /// stores. Cloning-and-nulling (rather than rebuilding via `build_commit_body`)
    /// preserves every other field the engine put in the body, even ones this
    /// crate does not model. A [`full`](Self::full) policy returns an unchanged
    /// clone.
    pub fn redact_body(&self, body: &Value) -> Value {
        let mut body = body.clone();
        if let Some(object) = body.as_object_mut() {
            if !self.include_text {
                object.insert("tvtext".to_string(), Value::Null);
            }
            if !self.include_lexical {
                object.insert("tvlex".to_string(), Value::Null);
            }
        }
        body
    }
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self::redacted()
    }
}
