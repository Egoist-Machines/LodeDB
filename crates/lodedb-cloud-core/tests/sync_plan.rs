//! Tests for the pure sync classifier: the explicit decision table, plus
//! invariants asserted over an exhaustive enumeration of a small domain.

use lodedb_cloud_core::{classify, SnapRef, SyncClassification};

/// A snapshot of commit `tag` carrying the given payload stores. Snapshots of
/// one commit share a logical id and per-store identities; each distinct
/// store combination is a distinct snapshot id (matching how redaction
/// changes the body bytes).
fn snap(tag: &str, generation: u64, has_text: bool, has_lexical: bool) -> SnapRef {
    let snapshot_id = if has_text || has_lexical {
        format!(
            "snap-{tag}-t{}l{}",
            u8::from(has_text),
            u8::from(has_lexical)
        )
    } else {
        // The fully redacted form is its own logical identity.
        format!("logical-{tag}")
    };
    SnapRef {
        snapshot_id,
        logical_id: format!("logical-{tag}"),
        generation,
        text_id: has_text.then(|| format!("text-{tag}")),
        lexical_id: has_lexical.then(|| format!("lex-{tag}")),
    }
}

/// A snapshot of commit `tag` whose *text store identity* disagrees with
/// [`snap`]'s — same commit, same store set, different payload bytes (a
/// re-encoded or tampered text store).
fn conflicting_text(tag: &str, generation: u64, has_lexical: bool) -> SnapRef {
    let mut reference = snap(tag, generation, true, has_lexical);
    reference.snapshot_id = format!("{}-alttext", reference.snapshot_id);
    reference.text_id = Some(format!("text-{tag}-alt"));
    reference
}

/// A full (text + lexical) snapshot of commit `tag`.
fn full(tag: &str, generation: u64) -> SnapRef {
    snap(tag, generation, true, true)
}

/// The fully redacted form of `full(tag, _)`: snapshot id == logical id.
fn redacted(tag: &str, generation: u64) -> SnapRef {
    snap(tag, generation, false, false)
}

use SyncClassification::*;

/// One decision-table row: (local, base, remote) -> expected classification.
type Case<'a> = (
    Option<&'a SnapRef>,
    Option<&'a SnapRef>,
    Option<&'a SnapRef>,
    SyncClassification,
);

#[test]
fn the_decision_table() {
    let a = full("a", 2);
    let a_red = redacted("a", 2);
    let a_text = snap("a", 2, true, false);
    let a_lex = snap("a", 2, false, true);
    let a_text_alt = conflicting_text("a", 2, false);
    let b = full("b", 3);
    let c = full("c", 3);
    let rollback = full("r", 1);

    let cases: &[Case] = &[
        // Nothing anywhere / one side absent.
        (None, None, None, InSync),
        (None, Some(&a), None, InSync),
        (Some(&a), None, None, LocalAhead),
        (Some(&a), Some(&a), None, LocalAhead),
        (None, None, Some(&a), RemoteAhead),
        (None, Some(&a), Some(&a), RemoteAhead),
        // Same content both sides: in sync (absent or matching base).
        (Some(&a), None, Some(&a), InSync),
        (Some(&a), Some(&a), Some(&a), InSync),
        // Fast-forwards: the side equal to the base did not move.
        (Some(&b), Some(&a), Some(&a), LocalAhead),
        (Some(&b), Some(&a), Some(&a_red), LocalAhead),
        (Some(&a), Some(&a), Some(&b), RemoteAhead),
        (Some(&a_red), Some(&a), Some(&b), RemoteAhead),
        // Both moved past the base: force required.
        (Some(&b), Some(&a), Some(&c), Diverged),
        // No base while the ends differ: no way to pick a direction.
        (Some(&a), None, Some(&b), Unknown),
        // Generation regression against the base: rollback/tamper suspect —
        // even when the two ends agree with each other (a coordinated
        // rollback must be loud, not a silent no-op or republish).
        (Some(&rollback), Some(&a), Some(&a), Unknown),
        (Some(&a), Some(&a), Some(&rollback), Unknown),
        (Some(&a), Some(&b), Some(&a), Unknown),
        (Some(&rollback), Some(&a), Some(&rollback), Unknown),
        // ...including toward an absent side: auto-pushing/pulling a
        // rolled-back generation would silently re-establish it as current.
        (Some(&rollback), Some(&a), None, Unknown),
        (None, Some(&a), Some(&rollback), Unknown),
        (
            Some(&rollback),
            Some(&a),
            Some(&snap("r", 1, false, false)),
            Unknown,
        ),
        // Same engine commit, local carrying strictly more payload stores: a
        // push republishes the pointer without any lineage conflict.
        (Some(&a), Some(&a), Some(&a_red), Republish),
        (Some(&a), None, Some(&a_red), Republish),
        (Some(&a), Some(&a), Some(&a_text), Republish),
        (Some(&a_text), Some(&a), Some(&a_red), Republish),
        // The mirror is NOT a republish: pushing a store-subset local over the
        // remote would drop stores, so it stays a no-op.
        (Some(&a_red), Some(&a), Some(&a), InSync),
        (Some(&a_text), Some(&a), Some(&a), InSync),
        // Incomparable store sets of one commit: publishing either way drops
        // a store the other has — force required.
        (Some(&a_text), Some(&a), Some(&a_lex), Unknown),
        (Some(&a_lex), Some(&a), Some(&a_text), Unknown),
        // Same commit, same store set, but a shared store's payload identity
        // differs (re-encoded or tampered text): never silently "in sync",
        // and never a republish even when local carries more stores.
        (Some(&a_text), Some(&a), Some(&a_text_alt), Unknown),
        (Some(&a), Some(&a), Some(&a_text_alt), Unknown),
        // A side that enriched the base with payload has moved even though
        // its logical id has not: an unpublished `--include-text` upgrade of
        // the last-synced commit must not be discarded by a fast-forward
        // when the other end advanced.
        (Some(&a), Some(&a_red), Some(&b), Diverged),
        (Some(&b), Some(&a_red), Some(&a), Diverged),
        (Some(&a_text_alt), Some(&a_text), Some(&b), Diverged),
        // Whereas a side whose payload merely narrowed (policy view smaller
        // than the recorded base) has nothing to lose: still a fast-forward.
        (Some(&a_red), Some(&a), Some(&b), RemoteAhead),
        (Some(&b), Some(&a), Some(&a_red), LocalAhead),
    ];

    for (index, (local, base, remote, expected)) in cases.iter().enumerate() {
        assert_eq!(
            classify(*local, *base, *remote),
            *expected,
            "case #{index}: local={local:?} base={base:?} remote={remote:?}"
        );
    }
}

/// Every combination of local/base/remote drawn from a small domain (absent,
/// two full commits at different generations, the redacted form of the first,
/// and a low-generation commit) — the classifier must uphold its global
/// invariants everywhere, not just on the table above.
#[test]
fn invariants_hold_over_the_exhaustive_domain() {
    let a = full("a", 2);
    let a_red = redacted("a", 2);
    let a_text = snap("a", 2, true, false);
    let a_lex = snap("a", 2, false, true);
    let a_text_alt = conflicting_text("a", 2, false);
    let b = full("b", 3);
    let low = full("low", 0);
    let domain: Vec<Option<&SnapRef>> = vec![
        None,
        Some(&a),
        Some(&a_red),
        Some(&a_text),
        Some(&a_lex),
        Some(&a_text_alt),
        Some(&b),
        Some(&low),
    ];

    for local in &domain {
        for base in &domain {
            for remote in &domain {
                let classification = classify(*local, *base, *remote);

                // Force is required exactly for Diverged/Unknown; every other
                // outcome is a no-op or a single fast-forward.
                let requires_force = matches!(classification, Diverged | Unknown);

                // Force is never demanded when there is nothing anywhere, and
                // divergence specifically needs both ends (Unknown can arise
                // with one absent end: a rollback below the trusted base).
                if requires_force {
                    assert!(
                        local.is_some() || remote.is_some(),
                        "force demanded with nothing on either end: \
                         base={base:?}"
                    );
                }
                if classification == Diverged {
                    assert!(
                        local.is_some() && remote.is_some(),
                        "divergence needs both ends: local={local:?} \
                         base={base:?} remote={remote:?}"
                    );
                }

                // Identical logical content never classifies as ahead or
                // diverged: a generation regression against the base or a
                // shared-store identity conflict is Unknown; otherwise
                // comparable store sets are InSync/Republish and incomparable
                // ones (each end carrying a store the other lacks) demand
                // force as Unknown.
                if let (Some(local), Some(remote)) = (local, remote) {
                    if local.logical_id == remote.logical_id {
                        let regressed = base.is_some_and(|base| {
                            local.generation < base.generation
                                || remote.generation < base.generation
                        });
                        let conflicting = local.snapshot_id != remote.snapshot_id
                            && ((local.text_id.is_some()
                                && remote.text_id.is_some()
                                && local.text_id != remote.text_id)
                                || (local.lexical_id.is_some()
                                    && remote.lexical_id.is_some()
                                    && local.lexical_id != remote.lexical_id));
                        let comparable = ((local.text_id.is_none() || remote.text_id.is_some())
                            && (local.lexical_id.is_none() || remote.lexical_id.is_some()))
                            || ((remote.text_id.is_none() || local.text_id.is_some())
                                && (remote.lexical_id.is_none() || local.lexical_id.is_some()));
                        if regressed || conflicting {
                            assert_eq!(
                                classification, Unknown,
                                "same-commit rollback/payload-conflict must demand force: \
                                 local={local:?} base={base:?} remote={remote:?}"
                            );
                        } else if comparable {
                            assert!(
                                matches!(classification, InSync | Republish),
                                "same commit with comparable stores must be \
                                 InSync/Republish, got {classification:?} for \
                                 local={local:?} remote={remote:?}"
                            );
                        } else {
                            assert_eq!(
                                classification, Unknown,
                                "incomparable store sets must demand force: \
                                 local={local:?} remote={remote:?}"
                            );
                        }
                    }
                }

                // Push/pull duality: swapping the ends mirrors the direction.
                // Republish mirrors to InSync (a redacted local never
                // "upgrades" a full remote).
                let mirrored = classify(*remote, *base, *local);
                let expected_mirror = match classification {
                    LocalAhead => RemoteAhead,
                    RemoteAhead => LocalAhead,
                    Republish => InSync,
                    InSync => {
                        // InSync may mirror to Republish (the redaction
                        // asymmetry); both directions must still be
                        // conflict-free.
                        assert!(
                            matches!(mirrored, InSync | Republish),
                            "InSync mirrored to {mirrored:?}"
                        );
                        continue;
                    }
                    other => other,
                };
                assert_eq!(
                    mirrored, expected_mirror,
                    "duality broken for local={local:?} base={base:?} remote={remote:?}"
                );
            }
        }
    }
}
