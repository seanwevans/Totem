"""Tests for :mod:`totem.runtime.capabilities` covering all branches."""

import pytest

from totem.runtime import capabilities as capabilities_module
from totem.runtime.capabilities import (
    CAPABILITY_FACTORIES,
    Capability,
    CapabilityUseResult,
    create_default_environment,
    ensure_capability,
    extract_capability,
    resolve_value,
    store_capability,
    use_file_read,
    use_file_write,
    use_net_send,
)


def test_capability_evolve_updates_state_and_history():
    cap = Capability(
        "Test", state={"count": 0}, history=[{"action": "start", "detail": None}]
    )

    new_cap = cap.evolve("tick", detail=1, state_updates={"count": 1})

    assert cap.history == [{"action": "start", "detail": None}]
    assert new_cap.history[-1] == {"action": "tick", "detail": 1}
    assert new_cap.state["count"] == 1
    assert not cap.is_active  # original capability consumed
    assert new_cap.is_active


def test_capability_evolve_rejects_when_inactive():
    cap = Capability("Inactive")
    cap.evolve("consume")

    with pytest.raises(RuntimeError, match="already consumed"):
        cap.evolve("again")


def test_capability_evolve_validates_permission_subset():
    read_cap = Capability("Reader", permissions={"read"})

    with pytest.raises(ValueError, match="outside of its scope"):
        read_cap.evolve("restrict", permission_subset={"write"})

    unrestricted = Capability("Grantor")
    narrowed = unrestricted.evolve("narrow", permission_subset={"alpha", "beta"})

    assert narrowed.permissions == frozenset({"alpha", "beta"})
    assert narrowed.is_active

    existing = Capability("Existing", permissions={"read", "write"})
    reduced = existing.evolve("reduce", permission_subset={"read"})
    assert reduced.permissions == frozenset({"read"})


def test_capability_evolve_revocation_paths():
    cap = Capability("Revokable", permissions={"send"})

    revoked = cap.evolve("close", revoke=True)
    assert revoked.permissions == frozenset()
    assert revoked.revoked
    assert not revoked.is_active

    with pytest.raises(ValueError, match="Cannot restrict permissions"):
        Capability("Bad").evolve("oops", revoke=True, permission_subset={"x"})


def test_capability_has_permission_handles_revocation_and_default():
    unrestricted = Capability("Any")
    assert unrestricted.has_permission("whatever")

    revoked = unrestricted.evolve("revoke", revoke=True)
    assert not revoked.has_permission("anything")


def test_capability_use_result_helpers():
    cap = Capability("Helper")
    result = CapabilityUseResult(cap, value="payload")

    assert resolve_value(result) == "payload"
    assert resolve_value("direct") == "direct"
    assert extract_capability(result) is cap
    assert extract_capability(cap) is cap
    assert extract_capability(123) is None


def test_use_file_read_covers_end_of_contents():
    env_cap = CAPABILITY_FACTORIES["FileRead"]()
    first = use_file_read(env_cap)
    second = use_file_read(first.capability)

    assert first.value == "input_data"
    assert second.value is None  # exhausted contents
    assert second.capability.history[-1]["action"] == "read"


def test_use_file_write_and_net_send_mutate_state():
    write_cap = CAPABILITY_FACTORIES["FileWrite"]()
    net_cap = CAPABILITY_FACTORIES["NetSend"]()

    write_result = use_file_write(write_cap, payload={"hello": "world"})
    assert write_result.value is True
    assert write_result.capability.state["writes"] == [{"hello": "world"}]

    send_result = use_net_send(net_cap, payload="data")
    assert send_result.value == "sent:data"
    assert send_result.capability.state["transmissions"] == ["data"]


def test_use_functions_respect_permissions():
    no_read = Capability("NoRead", permissions={"write"})
    with pytest.raises(RuntimeError, match="does not permit read"):
        use_file_read(no_read)

    no_write = Capability("NoWrite", permissions={"read"})
    with pytest.raises(RuntimeError, match="does not permit write"):
        use_file_write(no_write, payload=None)

    no_send = Capability("NoSend", permissions=set())
    with pytest.raises(RuntimeError, match="does not permit network send"):
        use_net_send(no_send, payload=None)


def test_environment_helpers_manage_capabilities():
    env = create_default_environment()

    for kind in CAPABILITY_FACTORIES:
        assert kind in env["__capabilities__"]

    custom_cap = Capability("Custom")
    store_capability(env, "Custom", custom_cap)
    assert env["__capabilities__"]["Custom"] is custom_cap

    ensured = ensure_capability(env, "FileRead")
    assert ensured is env["__capabilities__"]["FileRead"]

    fresh_env = {"__capabilities__": {}}
    ensured_new = ensure_capability(fresh_env, "FileWrite")
    assert ensured_new is fresh_env["__capabilities__"]["FileWrite"]


def test_clone_list_handles_none_and_existing_lists():
    assert capabilities_module._clone_list(None) == []

    sample = [1, 2]
    cloned = capabilities_module._clone_list(sample)
    assert cloned == sample and cloned is not sample
