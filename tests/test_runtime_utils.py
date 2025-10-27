import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    CAPABILITY_FACTORIES,
    Capability,
    CapabilityUseResult,
    create_default_environment,
    ensure_capability,
    extract_capability,
    move_env_value,
    read_env_value,
    resolve_value,
    store_capability,
    use_file_read,
    use_file_write,
)


@pytest.fixture
def fresh_env():
    return create_default_environment()


def test_capability_evolve_consumes_previous_generation():
    cap = Capability("Demo", state={"count": 0})
    next_cap = cap.evolve("increment", detail=1, state_updates={"count": 1})

    assert not cap.is_active
    assert next_cap.generation == cap.generation + 1
    assert next_cap.state["count"] == 1

    with pytest.raises(RuntimeError):
        cap.evolve("increment", detail=2)


def test_read_env_value_handles_defaults_and_moved_entries():
    env = {}

    assert read_env_value(env, None, default="sentinel") == "sentinel"
    assert read_env_value(env, "missing", default="fallback") == "fallback"

    with pytest.raises(KeyError):
        read_env_value(env, "missing")

    env["life"] = "payload"
    assert read_env_value(env, "life") == "payload"

    move_env_value(env, "life")

    with pytest.raises(RuntimeError):
        read_env_value(env, "life")


def test_move_env_value_enforces_single_move_and_unknown_lifetime():
    env = {"life": "payload"}

    moved = move_env_value(env, "life")
    assert moved == "payload"
    assert repr(env["life"]).startswith("<moved:")

    with pytest.raises(RuntimeError):
        move_env_value(env, "life")

    with pytest.raises(KeyError):
        move_env_value(env, "missing")

    with pytest.raises(RuntimeError):
        move_env_value(env, None)


def test_ensure_and_store_capability_roundtrip(fresh_env):
    caps = fresh_env["__capabilities__"]
    read_cap = caps.pop("FileRead")

    regenerated = ensure_capability(fresh_env, "FileRead")
    assert regenerated.kind == "FileRead"
    assert regenerated is not read_cap

    replacement = Capability("FileRead", state={"index": 42})
    store_capability(fresh_env, "FileRead", replacement)

    assert ensure_capability(fresh_env, "FileRead") is replacement


def test_resolve_and_extract_capability_from_use_result():
    cap = CAPABILITY_FACTORIES["FileWrite"]()
    result = use_file_write(cap, payload=7)

    assert isinstance(result, CapabilityUseResult)
    assert resolve_value(result) is True

    successor = extract_capability(result)
    assert successor.kind == "FileWrite"
    assert successor.generation == cap.generation + 1
    assert successor.history[-1] == {"action": "write", "detail": 7}

    assert extract_capability(successor) is successor
    assert extract_capability("not a capability") is None


def test_use_file_read_advances_contents(fresh_env):
    cap = fresh_env["__capabilities__"]["FileRead"]

    first = use_file_read(cap)
    second = use_file_read(first.capability)

    assert resolve_value(first) == "input_data"
    assert resolve_value(second) is None
    assert first.capability.generation == cap.generation + 1
    assert second.capability.state["index"] == 2
    assert not cap.is_active
