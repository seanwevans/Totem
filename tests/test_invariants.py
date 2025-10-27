import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    EFFECT_GRADES,
    Borrow,
    Node,
    Scope,
    build_tir,
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


def test_inferred_node_types_follow_arity(sample_root):
    for scope in collect_scopes(sample_root):
        for node in scope.nodes:
            assert node.typ == f"ADT<{len(node.borrows)}>"
            assert node.arity == len(node.borrows)


def test_pattern_match_lowers_to_switch():
    root_scope = Scope("root")

    ctor = Node("A", "int32", root_scope)
    root_scope.nodes.append(ctor)
    ctor.update_type()

    match_node = Node("P", "match", root_scope)
    match_node.meta["match_cases"] = [
        {"constructor": ("A", ctor.arity), "result": "arm_a"},
        {"constructor": ("B", 2), "result": "arm_b"},
    ]
    match_node.meta["default_case"] = "fallthrough"
    root_scope.nodes.append(match_node)

    borrow = Borrow("shared", ctor.owned_life, root_scope)
    match_node.borrows.append(borrow)
    ctor.owned_life.borrows.append(borrow)
    match_node.update_type()

    program = build_tir(root_scope)
    switch_instrs = [instr for instr in program.instructions if instr.op == "SWITCH"]
    assert switch_instrs, "MATCH should lower to SWITCH"

    switch = switch_instrs[0]
    assert switch.metadata.get("default") == "fallthrough"
    constructor_tags = {tuple(case["constructor"]): case["tag"] for case in switch.metadata["cases"]}
    assert ("A", ctor.arity) in constructor_tags
    assert switch.args == [ctor.owned_life.id]
