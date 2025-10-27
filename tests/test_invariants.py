import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    EFFECT_GRADES,
    build_bitcode_document,
    evaluate_scope,
    structural_decompress,
    verify_bitcode_document,
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


def find_first_node_dict(scope_dict):
    nodes = scope_dict.get("nodes", [])
    if nodes:
        return nodes[0]
    for child in scope_dict.get("children", []):
        candidate = find_first_node_dict(child)
        if candidate:
            return candidate
    return None


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


def test_bitcode_includes_certificates(sample_root):
    result = evaluate_scope(sample_root)
    doc = build_bitcode_document(sample_root, result)

    assert "certificates" in doc
    alias_cert = doc["certificates"]["aliasing"]
    grade_cert = doc["certificates"]["grades"]

    assert alias_cert["ok"]
    assert grade_cert["ok"]

    # Machine-checkable verification passes for untampered bitcode.
    assert verify_bitcode_document(doc) is True


def test_certificate_verification_detects_tampering(sample_root):
    result = evaluate_scope(sample_root)
    doc = build_bitcode_document(sample_root, result)

    # Mutate a node grade without updating certificates.
    first_node = find_first_node_dict(doc["root_scope"])
    assert first_node is not None
    first_node["grade"] = EFFECT_GRADES[-1]

    with pytest.raises(ValueError):
        verify_bitcode_document(doc)
