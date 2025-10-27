import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    EFFECT_GRADES,
    assemble_bytecode,
    build_tir,
    evaluate_scope,
    run_bytecode,
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


def test_pure_fence_rejects_impure_ops():
    with pytest.raises(ValueError):
        structural_decompress("(c)")


def test_state_fence_rejects_io_ops():
    # 'b' (state) is allowed, but 'c' (io) is not.
    structural_decompress("[ab]")
    with pytest.raises(ValueError):
        structural_decompress("[ac]")


def test_nested_scope_inherits_parent_cap():
    with pytest.raises(ValueError):
        structural_decompress("(a{b})")
def test_bytecode_vm_matches_scope_evaluation(sample_root):
    tir = build_tir(sample_root)
    bytecode = assemble_bytecode(tir)
    vm_result = run_bytecode(bytecode)
    scope_result = evaluate_scope(sample_root)

    assert vm_result.grade == scope_result.grade
    assert vm_result.log == scope_result.log
