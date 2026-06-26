from __future__ import annotations

import pytest

from lodedb.engine.runtime_policy import (
    NativeCoreMode,
    native_core_mode_from_env,
    native_core_strict_parity_from_env,
    native_core_write_mode_from_env,
)


def test_native_core_flags_default_off() -> None:
    env: dict[str, str] = {}
    assert native_core_mode_from_env(env) == NativeCoreMode.OFF
    assert native_core_write_mode_from_env(env) == NativeCoreMode.OFF
    assert native_core_strict_parity_from_env(env) is False


def test_native_core_flags_parse_rollout_modes() -> None:
    env = {
        "LODEDB_NATIVE_CORE": "shadow",
        "LODEDB_NATIVE_CORE_WRITE": "on",
        "LODEDB_NATIVE_CORE_STRICT_PARITY": "yes",
    }
    assert native_core_mode_from_env(env) == NativeCoreMode.SHADOW
    assert native_core_write_mode_from_env(env) == NativeCoreMode.ON
    assert native_core_strict_parity_from_env(env) is True


def test_native_core_flags_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="LODEDB_NATIVE_CORE"):
        native_core_mode_from_env({"LODEDB_NATIVE_CORE": "maybe"})
    with pytest.raises(ValueError, match="LODEDB_NATIVE_CORE_WRITE"):
        native_core_write_mode_from_env({"LODEDB_NATIVE_CORE_WRITE": "maybe"})
    with pytest.raises(ValueError, match="STRICT_PARITY"):
        native_core_strict_parity_from_env({"LODEDB_NATIVE_CORE_STRICT_PARITY": "maybe"})
