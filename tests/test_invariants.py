import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    EFFECT_GRADES,
    build_tir,
    compute_tir_distance,
    continuous_semantics_profile,
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


def build_tir_from_src(src):
    return build_tir(structural_decompress(src))


def test_tir_distance_identical_programs():
    tir_a = build_tir_from_src("ab")
    tir_b = build_tir_from_src("ab")
    dist = compute_tir_distance(tir_a, tir_b)
    assert dist == {
        "node_edits": 0,
        "grade_delta": 0,
        "borrow_rewires": 0,
        "total": 0,
    }


def test_tir_distance_grade_and_borrow_changes():
    tir_a = build_tir_from_src("ab")
    tir_b = build_tir_from_src("ac")
    dist = compute_tir_distance(tir_a, tir_b)
    assert dist["node_edits"] == 0
    assert dist["grade_delta"] == 1
    assert dist["borrow_rewires"] == 1
    assert dist["total"] == 2


def test_tir_distance_detects_node_additions():
    tir_a = build_tir_from_src("ab")
    tir_b = build_tir_from_src("abc")
    dist = compute_tir_distance(tir_a, tir_b)
    assert dist["node_edits"] == 1
    assert dist["total"] >= 1


def test_continuous_semantics_profile_reports_entries():
    src = "{ab}"
    profile = continuous_semantics_profile(src)
    assert len(profile) == len(src)
    for entry in profile:
        assert "distance" in entry
        assert entry["distance"]["total"] >= 0
