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
    build_tir,
    compute_tir_distance,
    continuous_semantics_profile,
    evaluate_scope,
    ActorSystem,
    OwnedMessage,
    evaluate_scope,
    CapabilityUseResult,
    create_default_environment,
    evaluate_scope,
    Borrow,
    Node,
    Scope,
    evaluate_scope,
    assemble_bytecode,
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


def build_tir_from_src(src):
    return build_tir(structural_decompress(src))


def test_tir_distance_identical_programs():
    tir_a = build_tir_from_src("ab")
    tir_b = build_tir_from_src("ab")
    dist = compute_tir_distance(tir_a, tir_b)
    assert dist == {
        "node_edits": 0,
        "grade_delta": 0,
        "op_changes": 0,
        "type_changes": 0,
        "borrow_rewires": 0,
        "total": 0,
    }


def test_tir_distance_grade_and_borrow_changes():
    tir_a = build_tir_from_src("ab")
    tir_b = build_tir_from_src("ac")
    dist = compute_tir_distance(tir_a, tir_b)
    assert dist["node_edits"] == 0
    assert dist["grade_delta"] == 1
    assert dist["op_changes"] == 1
    assert dist["type_changes"] == 0
    assert dist["borrow_rewires"] == 1
    assert dist["total"] == 3


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


def test_actor_messages_are_move_only():
    system = ActorSystem()
    capability = system.spawn()
    message = OwnedMessage({"payload": "ping"}, capability, system.next_message_id())

    send_result = capability.send(message)
    assert send_result.log == [f"send:{capability.actor_id}:msg{message.message_id}"]

    with pytest.raises(RuntimeError):
        capability.send(message)


def test_actor_pipeline_integration():
    program = structural_decompress("{hjklp}")
    result = evaluate_scope(program)
    assert result.grade == "sys"
    assert any(entry.startswith("actor_0:") for entry in result.log)


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
    constructor_tags = {
        tuple(case["constructor"]): case["tag"] for case in switch.metadata["cases"]
    }
    assert ("A", ctor.arity) in constructor_tags
    assert switch.args == [ctor.owned_life.id]


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
