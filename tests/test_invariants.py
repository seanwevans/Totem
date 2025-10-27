import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    EFFECT_GRADES,
    CapabilityUseResult,
    create_default_environment,
    evaluate_scope,
    structural_decompress,
)


def collect_scopes(scope):
    scopes = [scope]
    for child in scope.children:
        scopes.extend(collect_scopes(child))
    return scopes


def compute_expected_grade(scope):
    idx = 0
    for node in scope.nodes:
        idx = max(idx, EFFECT_GRADES.index(node.grade))
    for child in scope.children:
        idx = max(idx, compute_expected_grade(child))
    return idx


@pytest.fixture
def sample_root():
    return structural_decompress("{a{bc}de{fg}}")


def test_each_drop_belongs_to_its_scope(sample_root):
    for scope in collect_scopes(sample_root):
        for lifetime in scope.drops:
            assert lifetime.end_scope is scope
            assert lifetime.owner_scope is scope


def test_borrows_target_known_lifetime(sample_root):
    scopes = collect_scopes(sample_root)
    known_lifetimes = {
        node.owned_life.id: node.owned_life for scope in scopes for node in scope.nodes
    }

    for scope in scopes:
        for node in scope.nodes:
            for borrow in node.borrows:
                assert (
                    borrow.target.id in known_lifetimes
                ), f"Unknown lifetime target: {borrow.target}"


def test_root_grade_matches_max_child(sample_root):
    result = evaluate_scope(sample_root)
    expected_idx = compute_expected_grade(sample_root)
    assert result.grade == EFFECT_GRADES[expected_idx]


def test_file_read_capability_linear_progression():
    env = create_default_environment()
    initial_cap = env["__capabilities__"]["FileRead"]
    root = structural_decompress("cc")
    result = evaluate_scope(root, env)

    read_cap = env["__capabilities__"]["FileRead"]
    assert read_cap.generation == initial_cap.generation + 2
    assert len(read_cap.history) == 2
    assert not initial_cap.is_active
    assert result.grade == "io"


def test_file_write_capability_records_payload():
    env = create_default_environment()
    root = structural_decompress("{fg}")
    evaluate_scope(root, env)

    write_cap = env["__capabilities__"]["FileWrite"]
    assert write_cap.history[-1]["action"] == "write"
    assert write_cap.history[-1]["detail"] == 5
    assert write_cap.state["writes"] == [5]

    g_node = root.children[0].nodes[-1]
    result_value = env[g_node.owned_life.id]
    assert isinstance(result_value, CapabilityUseResult)
    assert result_value.value is True


def test_net_send_capability_updates_and_returns_result():
    env = create_default_environment()
    root = structural_decompress("as")
    result = evaluate_scope(root, env)

    assert result.grade == "sys"

    net_cap = env["__capabilities__"]["NetSend"]
    assert net_cap.history[-1]["action"] == "send"
    assert net_cap.history[-1]["detail"] == 1

    s_node = root.nodes[-1]
    stored = env[s_node.owned_life.id]
    assert isinstance(stored, CapabilityUseResult)
    assert stored.value == "sent:1"
