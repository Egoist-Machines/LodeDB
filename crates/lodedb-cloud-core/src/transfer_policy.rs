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

use crate::error::{ArtifactStoreError, Result};
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

    /// Returns a copy of a committed body with present excluded stores nulled.
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
                if let Some(entry) = object.get_mut("tvtext") {
                    *entry = Value::Null;
                }
            }
            if !self.include_lexical {
                if let Some(entry) = object.get_mut("tvlex") {
                    *entry = Value::Null;
                }
            }
        }
        body
    }

    /// Refuses a body this policy cannot honestly redact.
    ///
    /// A text-bearing LodeGraph topology (`gtopotext`) carries its user text
    /// INSIDE its single SQLite artifact, so nulling the sub-manifest cannot
    /// redact it the way nulling `tvtext` redacts a vector store: the result
    /// would pin no graph store at all, a silent non-backup whose store kind
    /// flips to vector. A text-excluding policy therefore refuses the
    /// transfer outright rather than shipping user text without the opt-in
    /// (the server takes the same stance, gating the artifact on `read:text`
    /// instead of serving a stripped copy). Opting into text ships the body
    /// verbatim. The content-free `gtopo` topology carries no text and never
    /// refuses.
    pub fn refuse_unredactable(&self, body: &Value) -> Result<()> {
        if !self.include_text
            && body
                .get("gtopotext")
                .is_some_and(|value| !value.is_null())
        {
            return Err(ArtifactStoreError::Backend(
                "this generation's LodeGraph topology embeds user text (gtopotext), which \
                 cannot be redacted out of its single SQLite artifact; re-run with the \
                 text opt-in (include_text / --include-text) to transfer it verbatim"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for TransferPolicy {
    fn default() -> Self {
        Self::redacted()
    }
}

#[cfg(test)]
mod tests {
    use super::TransferPolicy;
    use crate::snapshot_identity::snapshot_id;
    use lodedb_core::storage::commit_manifest::render_commit_manifest;
    use serde_json::{json, Value};

    #[test]
    fn redacts_present_text_and_lexical_stores() {
        let body = json!({
            "index_key": "idx",
            "tvtext": {"base": {"file_name": "g1.tvtext"}, "deltas": []},
            "tvlex": {"base": {"file_name": "g1.tvlex"}, "deltas": []},
        });

        let redacted = TransferPolicy::redacted().redact_body(&body);

        assert_eq!(redacted.get("tvtext"), Some(&Value::Null));
        assert_eq!(redacted.get("tvlex"), Some(&Value::Null));
    }

    #[test]
    fn redacting_keyless_graph_body_preserves_canonical_identity_and_serialization() {
        let body = json!({
            "index_key": "graph",
            "generation": 7,
            "base_epoch": 7,
            "gtopo": {
                "base": {
                    "file_name": "topology.sqlite3",
                    "sha256": "f2ba8c4a4a2f1b99ee90f6ec66c3b33f",
                    "file_bytes": 42,
                },
                "deltas": [],
            },
        });

        let redacted = TransferPolicy::default().redact_body(&body);

        assert_eq!(redacted, body);
        assert_eq!(
            redacted.as_object().unwrap().keys().collect::<Vec<_>>(),
            body.as_object().unwrap().keys().collect::<Vec<_>>(),
            "redaction must not add absent payload-store keys"
        );
        assert_eq!(
            render_commit_manifest(&redacted).unwrap(),
            render_commit_manifest(&body).unwrap(),
            "the engine-canonical serialization must remain byte-identical"
        );
        assert_eq!(snapshot_id(&redacted).unwrap(), snapshot_id(&body).unwrap());
    }

    #[test]
    fn redacting_text_only_body_does_not_insert_lexical_store() {
        let body = json!({
            "index_key": "idx",
            "tvtext": {"base": {"file_name": "g1.tvtext"}, "deltas": []},
        });

        let redacted = TransferPolicy::redacted().redact_body(&body);

        assert_eq!(redacted.get("tvtext"), Some(&Value::Null));
        assert!(redacted.get("tvlex").is_none());
    }
}
