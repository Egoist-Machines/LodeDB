//! Pure three-pointer sync classification, no I/O.
//!
//! Sync compares three snapshots of one index: the **local** committed
//! generation (as the caller's transfer policy would publish it), the **base**
//! recorded by the sidecar (the last state both ends agreed on), and the
//! **remote** committed generation. [`classify`] reduces those to one
//! [`SyncClassification`] that says which transfer, if any, is a
//! fast-forward, and which situations require an explicit force flag.
//!
//! Comparison runs on [`logical_id`](crate::snapshot_identity::logical_id)
//! (redaction-invariant content identity), never on generation numbers alone:
//! two independent lineages can share a generation number. Generations serve
//! only as a monotonicity sanity check, because single-base ancestry is
//! trust-based. The sidecar is a *claim* the local machine makes about
//! history, so a generation that moved backwards relative to the recorded base
//! is treated as rollback/tamper-suspect ([`Unknown`]) rather than
//! fast-forwarded.
//!
//! [`Unknown`]: SyncClassification::Unknown

/// One side's identity: which exact bytes ([`snapshot_id`]), which engine
/// commit regardless of redaction ([`logical_id`]), the committed generation
/// number (monotonicity check only), and a per-store identity for each
/// payload-bearing store the body carries.
///
/// The payload identities (`text_id`/`lexical_id`, `None` when the store is
/// absent) are what let the classifier compare two snapshots of the *same*
/// commit: the two stores are independent redaction choices
/// ([`TransferPolicy`]), so store-set inclusion decides between no-op and
/// republish. And because `logical_id` is computed with both stores nulled,
/// a store present on *both* ends must also match by identity, or the two
/// "same" snapshots disagree about payload content (different encodings, or
/// tampering) and no untransfer-free reconciliation exists.
///
/// [`snapshot_id`]: crate::snapshot_identity::snapshot_id
/// [`logical_id`]: crate::snapshot_identity::logical_id
/// [`TransferPolicy`]: crate::TransferPolicy
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapRef {
    pub snapshot_id: String,
    pub logical_id: String,
    pub generation: u64,
    /// Identity of the raw-text store's sub-manifest; `None` when absent.
    pub text_id: Option<String>,
    /// Identity of the lexical store's sub-manifest; `None` when absent.
    pub lexical_id: Option<String>,
}

impl SnapRef {
    /// Whether a store present on both snapshots differs by identity, the
    /// "same commit, conflicting payload" case that must never be reconciled
    /// silently.
    fn shared_store_conflict(&self, other: &Self) -> bool {
        let conflicts = |mine: &Option<String>, theirs: &Option<String>| matches!((mine, theirs), (Some(a), Some(b)) if a != b);
        conflicts(&self.text_id, &other.text_id) || conflicts(&self.lexical_id, &other.lexical_id)
    }

    /// Whether this snapshot's payload stores are a strict superset of
    /// `other`'s: it carries at least one store `other` lacks, and lacks none
    /// `other` carries.
    fn carries_more_than(&self, other: &Self) -> bool {
        let gains = (self.text_id.is_some() && other.text_id.is_none())
            || (self.lexical_id.is_some() && other.lexical_id.is_none());
        gains && other.carries_no_more_than(self)
    }

    /// Whether this snapshot's payload stores are a (non-strict) subset of
    /// `other`'s.
    fn carries_no_more_than(&self, other: &Self) -> bool {
        (self.text_id.is_none() || other.text_id.is_some())
            && (self.lexical_id.is_none() || other.lexical_id.is_some())
    }

    /// Whether discarding this snapshot in favor of something derived from
    /// `base` loses nothing: the same commit, no payload store `base` lacks,
    /// and no shared store whose content disagrees with `base`'s.
    ///
    /// This, not bare logical equality, is what makes a side "unchanged"
    /// for fast-forward purposes: a side that enriched the base with payload
    /// (an unpublished `--include-text` upgrade) has moved, even though its
    /// logical id has not.
    fn adds_nothing_over(&self, base: &Self) -> bool {
        self.logical_id == base.logical_id
            && !self.shared_store_conflict(base)
            && self.carries_no_more_than(base)
    }
}

/// What a sync of one index would have to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncClassification {
    /// Local and remote already hold the same content; nothing to transfer.
    InSync,
    /// Remote still equals the recorded base and local moved forward: a push
    /// is a fast-forward.
    LocalAhead,
    /// Local still equals the recorded base and remote moved forward: a pull
    /// is a fast-forward.
    RemoteAhead,
    /// Both sides moved past the base independently; a transfer in either
    /// direction discards the other side's commit, so force is required.
    Diverged,
    /// Local and remote are the same engine commit but the local copy carries
    /// payload stores the remote lacks (and none the other way): a push
    /// republishes the pointer (upgrading the remote) without any lineage
    /// conflict.
    Republish,
    /// History cannot be trusted or the transfer cannot be lossless: no
    /// recorded base while both sides hold different content, a generation
    /// moved backwards relative to the base, or the two ends carry
    /// incomparable payload-store sets of one commit. Force is required.
    Unknown,
}

impl SyncClassification {
    /// The stable lowercase name reports and error messages use.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InSync => "in_sync",
            Self::LocalAhead => "local_ahead",
            Self::RemoteAhead => "remote_ahead",
            Self::Diverged => "diverged",
            Self::Republish => "republish",
            Self::Unknown => "unknown",
        }
    }
}

/// Classifies one index's sync state from its three pointers.
///
/// `local` is the generation the caller *would publish* (i.e. already redacted
/// by its transfer policy), `base` is the sidecar's recorded last-synced state
/// (`None` when the sidecar is absent or corrupt), and `remote` is the remote's
/// committed generation. All lineage equality runs on `logical_id`.
///
/// Only [`Diverged`](SyncClassification::Diverged) and
/// [`Unknown`](SyncClassification::Unknown) require force; every other
/// classification maps to at most one fast-forward transfer.
pub fn classify(
    local: Option<&SnapRef>,
    base: Option<&SnapRef>,
    remote: Option<&SnapRef>,
) -> SyncClassification {
    // Nothing anywhere (a recorded base with both ends gone included): there
    // is no transfer to make.
    if local.is_none() && remote.is_none() {
        return SyncClassification::InSync;
    }

    // Generation regression against the recorded base on any present side is
    // rollback/tamper-suspect: never proceed silently over it, not for ends
    // that agree with each other (a coordinated rollback must be loud, and a
    // republish would publish payload onto a suspect remote), and not toward
    // an absent side either (auto-pushing a rolled-back generation to a fresh
    // remote would silently re-establish it as current). Checked before every
    // other comparison.
    if let Some(base) = base {
        let regressed =
            |side: Option<&SnapRef>| side.is_some_and(|side| side.generation < base.generation);
        if regressed(local) || regressed(remote) {
            return SyncClassification::Unknown;
        }
    }

    let (local, remote) = match (local, remote) {
        // One side absent (and no regression above): the transfer direction
        // is unambiguous, since there is nothing on the other end to discard.
        (Some(_), None) => return SyncClassification::LocalAhead,
        (None, Some(_)) => return SyncClassification::RemoteAhead,
        (Some(local), Some(remote)) => (local, remote),
        (None, None) => unreachable!("handled above"),
    };

    if local.logical_id == remote.logical_id {
        // Same engine commit; the ends should only differ by which payload
        // stores they carry (the logical id nulls exactly those stores).
        // First, a store present on BOTH ends must match by identity; a
        // mismatch means the two "same" snapshots disagree about payload
        // content, which no direction of transfer reconciles losslessly.
        // Then compare the store sets by inclusion:
        // - local carries stores the remote lacks (and nothing less): a push
        //   republishes the pointer, upgrading the remote without discarding
        //   anything;
        // - local carries no more than the remote: nothing to publish, no-op;
        // - incomparable (each carries a store the other lacks, text-only vs
        //   lexical-only): publishing either way drops a store the other end
        //   has, so force is required.
        return if local.snapshot_id == remote.snapshot_id {
            SyncClassification::InSync
        } else if local.shared_store_conflict(remote) {
            SyncClassification::Unknown
        } else if local.carries_more_than(remote) {
            SyncClassification::Republish
        } else if local.carries_no_more_than(remote) {
            SyncClassification::InSync
        } else {
            SyncClassification::Unknown
        };
    }

    // Local and remote differ. Without a base there is no way to tell which
    // side moved: fail toward force rather than guess a direction.
    let Some(base) = base else {
        return SyncClassification::Unknown;
    };

    // A fast-forward discards the "unchanged" side, so that side must add
    // nothing over the base: same commit AND no payload the base lacks. A
    // side that merely enriched the base with payload stores (an unpublished
    // `--include-text` upgrade of the last-synced commit) has moved too:
    // discarding it would silently drop that payload, so both-moved is
    // Diverged even when one side's logical id still equals the base's.
    match (
        local.adds_nothing_over(base),
        remote.adds_nothing_over(base),
    ) {
        (false, true) => SyncClassification::LocalAhead,
        (true, false) => SyncClassification::RemoteAhead,
        (false, false) => SyncClassification::Diverged,
        // Both ends equal the base while differing from each other is
        // impossible (the same-logical case returned above); keep the
        // conservative answer for completeness.
        (true, true) => SyncClassification::InSync,
    }
}
