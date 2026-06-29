"""Runtime selection policies and defaults for the LodeDB engine."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum


class TvimDeltaPersistencePolicy(StrEnum):
    """Names direct TurboVec `.tvim` delta persistence policies."""

    OFF = "off"
    AUTO = "auto"


class CommitMode(StrEnum):
    """Names the engine's per-mutation commit strategy.

    ``wal`` (the default) appends one framed record to ``<key>.wal`` per mutation
    and checkpoints into a generation periodically; it trades the lock-free reader
    snapshot (unnecessary for single-process writers) for a much cheaper per-write
    commit, and replays the WAL crash-atomically on open. ``generation`` is the
    classic path that publishes a new crash-atomic, MVCC-readable generation on
    every mutation via the root-manifest swap (pick it when many out-of-process
    readers must see every write the instant it commits, with no checkpoint lag).

    Both modes load and recover the same way: the durable on-disk base is always a
    committed generation, and any WAL tail is folded into a fresh generation on
    open, so a handle always opens onto a consistent generation regardless of mode.
    """

    GENERATION = "generation"
    WAL = "wal"


def tvim_delta_persistence_policy_from_env(
    env: Mapping[str, str] | None = None,
) -> TvimDeltaPersistencePolicy:
    """Returns the direct TurboVec delta persistence policy from the environment.

    Defaults to ``auto`` (delta appends persist O(changed rows) `.tvd` + `.jsd`
    segments with byte-exact restart equality and no query/load
    regressions); ``off`` keeps the legacy full `.tvim`/JSON rewrites.
    ``auto`` falls back visibly to full base rewrites for cold builds,
    missing manifests, unavailable patched vendored APIs, or compaction.
    """

    source = os.environ if env is None else env
    return _parse_tvim_delta_persistence_policy(
        source.get("LODEDB_TVIM_DELTA_PERSISTENCE", "auto")
    )


def commit_mode_from_env(env: Mapping[str, str] | None = None) -> CommitMode:
    """Returns the per-mutation commit mode from the environment.

    Defaults to ``wal``: a single-writer write-ahead-log commit path (one framed
    append per mutation, periodic checkpoint into a generation, crash-atomic WAL
    replay on open) that makes durable single adds roughly an order of magnitude
    cheaper than the per-mutation generation publish. ``LODEDB_COMMIT_MODE=generation``
    selects the classic path that publishes a crash-atomic, MVCC-readable
    generation on every mutation. Used only when no explicit ``commit_mode=`` is
    passed.
    """

    source = os.environ if env is None else env
    return parse_commit_mode(source.get("LODEDB_COMMIT_MODE", "wal"))


def parse_commit_mode(value: str | None) -> CommitMode:
    """Parses and validates a commit-mode string (shared by the env + SDK paths)."""

    try:
        return CommitMode(str(value or "wal").strip().lower())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in CommitMode)
        raise ValueError(f"commit_mode must be one of: {allowed}") from exc


def _parse_tvim_delta_persistence_policy(value: str | None) -> TvimDeltaPersistencePolicy:
    """Parses and validates a `.tvim` delta persistence policy string."""

    try:
        return TvimDeltaPersistencePolicy(str(value or "auto"))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in TvimDeltaPersistencePolicy)
        raise ValueError(
            f"LODEDB_TVIM_DELTA_PERSISTENCE must be one of: {allowed}"
        ) from exc
