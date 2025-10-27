import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import totem.runtime as runtime_mod

from totem.constants import LOGBOOK_FILE
from totem.ffi import FFIDeclaration, clear_ffi_registry, register_ffi_declarations
from totem.runtime import (
    ActorCapability,
    IRNode,
    Borrow,
    BytecodeInstruction,
    BytecodeProgram,
    BytecodeVM,
    MetaObject,
    Node,
    OwnedMessage,
    Scope,
    TIRInstruction,
    TIRProgram,
    _collect_aliasing_payload,
    _collect_grade_payload,
    _grade_certificate,
    _instruction_identity,
    _mutate_byte,
    assemble_bytecode,
    build_bitcode_document,
    build_tir,
    canonicalize_bitcode,
    check_aliasing,
    check_lifetimes,
    compute_scope_grades,
    compute_tir_distance,
    continuous_semantics_profile,
    create_default_environment,
    diff_bitcodes,
    evaluate_node,
    evaluate_scope,
    evaluate_pure_regions,
    emit_mlir_module,
    emit_llvm_ir,
    export_wasm_module,
    explain_borrow,
    explain_grade,
    fold_constants,
    common_subexpression_elimination,
    hash_bitcode_document,
    inline_pure_regions,
    inline_trivial_io,
    list_meta_ops,
    meta_emit,
    record_run,
    reorder_pure_ops,
    dead_code_elimination,
    scope_to_dict,
    schedule_effects,
    show_logbook,
    structural_decompress,
    tir_to_wat,
    print_scopes,
    reconstruct_scope,
    verify_ffi_calls,
    verify_bitcode_document,
    write_bitcode_document,
    load_totem_bitcode,
    run_bytecode,
    reflect,
)


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_scope_with_borrow():
    root = Scope("root")
    first = Node("A", "int32", root)
    root.nodes.append(first)
    first.update_type()
    second = Node("B", "int32", root)
    root.nodes.append(second)
    second.update_type()
    borrow = Borrow("mut", first.owned_life, root)
    second.borrows.append(borrow)
    first.owned_life.borrows.append(borrow)
    second.update_type()
    for node in (first, second):
        node.owned_life.end_scope = root
        if node.owned_life not in root.lifetimes:
            root.lifetimes.append(node.owned_life)
            root.drops.append(node.owned_life)
    return root, first, second, borrow


def test_repr_helpers_and_meta_objects():
    root, first, _, borrow = make_scope_with_borrow()

    capability = ActorCapability(None, "actor_1")
    message = OwnedMessage({"payload": 7}, capability, 42)
    before = repr(message)
    assert "actor_1" in before and "moved=False" in before
    message.move_payload()
    assert "moved=True" in repr(message)
    assert repr(capability) == "<Capability actor_1>"
    assert "mut" in repr(borrow)

    instr = TIRInstruction(
        "v0",
        "E",
        "int32",
        "pure",
        [{"kind": "borrow", "target": first.owned_life.id}],
        "root",
    )
    assert "borrow" in repr(instr)

    assert "MetaNode" in repr(MetaObject("Node", first))
    assert MetaObject("TIR", TIRProgram()).to_dict() == []


def test_structural_decompress_error_paths():
    with pytest.raises(ValueError):
        structural_decompress("}")

    with pytest.raises(ValueError):
        structural_decompress("(c")

    with pytest.raises(ValueError):
        structural_decompress("[c]")


def test_alias_lifetime_and_ffi_checks():
    root, first, second, borrow = make_scope_with_borrow()

    # Shared borrow alongside mutable borrow triggers aliasing error
    shared = Borrow("shared", first.owned_life, root)
    second.borrows.append(shared)
    first.owned_life.borrows.append(shared)
    alias_errors = []
    check_aliasing(root, alias_errors)
    assert any("Aliasing violation" in msg for msg in alias_errors)

    # Borrow from a deeper scope to trigger lifetime warning
    first.owned_life.end_scope = root
    child = Scope("child", root)
    reader = Node("A", "int32", child)
    child.nodes.append(reader)
    reader.update_type()
    deep_borrow = Borrow("shared", first.owned_life, child)
    reader.borrows.append(deep_borrow)
    first.owned_life.borrows.append(deep_borrow)
    lifetime_errors = []
    check_lifetimes(root, lifetime_errors)
    assert any("outlives" in msg for msg in lifetime_errors)

    # FFI metadata mismatches
    decl = FFIDeclaration("FFI_ADD", "io", ["int32"], "int32")
    register_ffi_declarations([decl])
    try:
        ffi_node = Node("FFI_ADD", "int32", root)
        root.nodes.append(ffi_node)
        ffi_node.update_type()
        # Borrow with wrong target count and type forcing error collection
        bogus = Node("A", "int32", root)
        root.nodes.append(bogus)
        bogus.update_type()
        first_borrow = Borrow("shared", bogus.owned_life, root)
        second_borrow = Borrow("shared", bogus.owned_life, root)
        ffi_node.borrows.extend([first_borrow, second_borrow])
        bogus.owned_life.borrows.extend([first_borrow, second_borrow])
        bogus.meta["fixed_type"] = "float"
        ffi_node.typ = "wrong"
        ffi_node.grade = "state"
        errors = []
        verify_ffi_calls(root, errors)
        assert any("arity" in msg for msg in errors)
        assert any("return type" in msg for msg in errors)
        assert any("grade mismatch" in msg for msg in errors)
    finally:
        clear_ffi_registry()


def test_grade_and_borrow_explanations():
    root, first, second, _ = make_scope_with_borrow()
    first.owned_life.end_scope = root
    second.owned_life.end_scope = root
    compute_scope_grades(root)

    summary = explain_grade(root, "state")
    assert summary["achieved"] and summary["nodes"]

    with pytest.raises(ValueError):
        explain_grade(root, "invalid")

    report = explain_borrow(root, first.owned_life.id)
    assert report["found"] and report["lines"]
    missing = explain_borrow(root, "not-there")
    assert not missing["found"]


def test_payload_helpers_cover_multiple_branches():
    root, first, second, _ = make_scope_with_borrow()
    first.owned_life.end_scope = root
    second.owned_life.end_scope = root

    alias_payload = _collect_aliasing_payload(root)
    assert "lifetimes" in alias_payload["summary"]

    grade_payload = _collect_grade_payload(root)
    assert grade_payload["computed_grade"] in {"pure", "state", "io", "sys", "meta"}

    certificate = _grade_certificate(root, grade_payload["computed_grade"])
    assert certificate["ok"]


def test_bitcode_round_trip_and_logging(temp_dir, capsys):
    tree = structural_decompress("a")
    result = evaluate_scope(tree)
    doc = build_bitcode_document(tree, result)
    bitcode_path = temp_dir / "program.json"
    write_bitcode_document(doc, bitcode_path)

    loaded = load_totem_bitcode(bitcode_path)
    assert verify_bitcode_document(loaded)
    assert canonicalize_bitcode(doc)["totem_version"] == doc["totem_version"]
    assert len(hash_bitcode_document(doc)) == 64

    modified = json.loads(json.dumps(doc))
    modified["evaluation"]["log"] = doc["evaluation"]["log"] + ["extra"]
    alt_path = temp_dir / "modified.json"
    write_bitcode_document(modified, alt_path)
    diff_bitcodes(str(bitcode_path), str(alt_path))
    out = capsys.readouterr().out
    assert "Bitcodes differ" in out

    with pytest.raises(RuntimeError):
        record_run(str(bitcode_path), result)
    entry = {
        "timestamp": "2024-01-01T00:00:00Z",
        "filename": str(bitcode_path),
        "hash": hash_bitcode_document(doc),
        "signature": None,
        "final_grade": result.grade,
        "log_length": len(result.log),
        "first_log": result.log[0] if result.log else None,
        "last_log": result.log[-1] if result.log else None,
    }
    Path(LOGBOOK_FILE).write_text(json.dumps(entry) + "\n")
    show_logbook(limit=1)
    log_output = capsys.readouterr().out
    assert "Totem Logbook" in log_output


def test_continuous_semantics_profile_and_mutator():
    base_tree = structural_decompress("ab")
    base_tir = build_tir(base_tree)

    profile = continuous_semantics_profile("a]", base_tir=base_tir, mutate_fn=_mutate_byte)
    assert profile and profile[0]["distance"]["total"] >= 0

    # Mutator returning original char skips entries
    profile_skip = continuous_semantics_profile("abc", base_tir=base_tir, mutate_fn=lambda ch: ch)
    assert profile_skip == []


def test_tir_to_wat_errors_and_export(temp_dir):
    tir = TIRProgram()
    pure_instr = TIRInstruction("v0", "E", "int32", "pure", [], "root")
    tir.instructions.append(pure_instr)
    with pytest.raises(ValueError):
        tir_to_wat(tir)

    pure_instr.args = ["unknown"]
    with pytest.raises(ValueError):
        tir_to_wat(tir)

    pure_instr.args = [{"target": "missing"}]
    with pytest.raises(ValueError):
        tir_to_wat(tir)

    pure_instr.op = "Z"
    with pytest.raises(NotImplementedError):
        tir_to_wat(tir)

    io_instr = TIRInstruction("v1", "G", "int32", "io", [{"target": "v0"}], "root")
    tir.instructions = [TIRInstruction("v0", "A", "int32", "pure", [], "root"), io_instr]
    with pytest.raises(PermissionError):
        tir_to_wat(tir)

    io_instr.args = ["no-producer"]
    with pytest.raises(ValueError):
        tir_to_wat(tir, capabilities={"io.write"})

    io_instr.args = []
    metadata = export_wasm_module(tir, temp_dir / "module.wat", capabilities={"io.write"}, metadata_path=temp_dir / "meta.json")
    assert metadata["io_instructions"] == 1


def test_meta_emit_and_optimizer_pipeline():
    program = TIRProgram()
    meta_obj = meta_emit(program, "A")
    assert meta_obj.kind == "TIR_Instruction"
    assert "reflect" in list_meta_ops()

    tir = TIRProgram()
    v0 = tir.emit("A", "int32", "pure", [], "root")
    v1 = tir.emit("D", "int32", "pure", [], "root")
    add = TIRInstruction("v2", "ADD", "int32", "pure", [v0, v1], "root")
    tir.instructions.append(add)
    fold_constants(tir)
    evaluate_pure_regions(tir)
    inline_trivial_io(tir)
    common_subexpression_elimination(tir)
    dead_code_elimination(tir)
    inline_pure_regions(tir)
    reorder_pure_ops(tir)
    schedule_effects(tir)


def test_bytecode_vm_custom_logging():
    vm = BytecodeVM()
    instr = BytecodeInstruction("v0", "A", "pure")
    vm._apply_operation = lambda _: (0, "single")  # type: ignore[attr-defined]
    vm._step(instr)
    assert vm.log == ["single"]
    assert vm._grade_index("unknown") == 0

    program = BytecodeProgram([BytecodeInstruction("v1", "Z", "pure")])
    result = vm.execute(program)
    assert result.log

    bytecode = assemble_bytecode(TIRProgram())
    assert isinstance(run_bytecode(bytecode), type(result))


def test_scope_dict_identity_and_distance():
    root, first, second, _ = make_scope_with_borrow()
    first.owned_life.end_scope = root
    second.owned_life.end_scope = root
    assert scope_to_dict(root)["nodes"]

    tir = build_tir(root)
    identity = _instruction_identity(tir.instructions[0], {})
    assert identity[0] in {"life", "scope"}

    other = TIRProgram()
    other.instructions.append(TIRInstruction("v0", "A", "int32", "pure", [], "root"))
    distance = compute_tir_distance(tir, other)
    assert distance["total"] >= 0


def test_evaluate_node_meta_and_effects():
    root = Scope("root")
    node_m = Node("M", "int32", root)
    root.nodes.append(node_m)
    node_n = Node("N", "int32", root)
    root.nodes.append(node_n)
    node_o = Node("O", "int32", root)
    root.nodes.append(node_o)
    node_h = Node("H", "int32", root)
    root.nodes.append(node_h)
    node_j = Node("J", "int32", root)
    root.nodes.append(node_j)
    node_k = Node("K", "int32", root)
    root.nodes.append(node_k)
    node_l = Node("L", "int32", root)
    root.nodes.append(node_l)

    env = create_default_environment()
    assert evaluate_node(node_m, env).grade == "meta"
    assert evaluate_node(node_n, env).grade == "meta"
    optimized = evaluate_node(node_o, env)
    assert optimized.grade == "meta"

    actor_system_effect = evaluate_node(node_h, env)
    assert actor_system_effect.grade == "sys"
    env[node_h.owned_life.id] = actor_system_effect.value

    node_j.borrows.append(Borrow("shared", node_h.owned_life, root))
    node_h.owned_life.borrows.append(node_j.borrows[0])
    cap_effect = evaluate_node(node_j, env)
    env[node_j.owned_life.id] = cap_effect.value

    node_k.borrows.append(Borrow("shared", node_j.owned_life, root))
    node_j.owned_life.borrows.append(node_k.borrows[0])
    msg_effect = evaluate_node(node_k, env)
    env[node_k.owned_life.id] = msg_effect.value

    node_l.borrows.append(Borrow("shared", node_k.owned_life, root))
    node_k.owned_life.borrows.append(node_l.borrows[0])
    send_effect = evaluate_node(node_l, env)
    assert send_effect.grade == "sys"



def test_irnode_and_emitters_cover_branches():
    ir = IRNode("i0", "OP", "int32", "pure", ["x"])
    assert ir.op == "OP"

    instr = TIRInstruction("v_repr", "E", "int32", "pure", [{"foo": "bar"}], "root")
    assert "foo" in repr(instr)

    tir = TIRProgram()
    skip_arg = {"kind": "borrow", "target": None}
    pure_instr = TIRInstruction("v0", "ADD", "int32", "pure", [skip_arg], "root")
    tir.instructions.append(pure_instr)
    mlir = emit_mlir_module(tir)
    assert "totem.add" in mlir

    llvm = emit_llvm_ir(tir)
    assert "totem_add" in llvm

    wasm_tir = TIRProgram()
    const_instr = TIRInstruction("v_const", "CONST", "int32", "pure", [], "root")
    const_instr.value = 7
    wasm_tir.instructions.append(const_instr)
    io_instr = TIRInstruction("v_io", "G", "int32", "io", [], "root")
    wasm_tir.instructions.append(io_instr)
    wat, metadata = tir_to_wat(wasm_tir, capabilities={"io.write"})
    assert "(i32.const 7)" in wat
    assert metadata["io_instructions"] == 1


def test_structural_decompress_additional_errors():
    with pytest.raises(ValueError):
        structural_decompress("{)")

    root, first, _, _ = make_scope_with_borrow()
    another = Node("C", "int32", root)
    root.nodes.append(another)
    another.update_type()
    mut_borrow = Borrow("mut", first.owned_life, root)
    another.borrows.append(mut_borrow)
    first.owned_life.borrows.append(mut_borrow)
    errors = []
    check_aliasing(root, errors)
    assert any("Multiple mutable" in msg for msg in errors)


def test_explain_grade_failure_and_cycle_detection():
    root, first, _, _ = make_scope_with_borrow()
    result = explain_grade(root, "meta")
    assert not result["achieved"]

    cyclic = Borrow("shared", first.owned_life, root)
    first.borrows.append(cyclic)
    first.owned_life.borrows.append(cyclic)
    cycle_report = explain_borrow(root, first.owned_life.id)
    assert any("cycle" in line for line in cycle_report["lines"])


def test_print_scopes_emits_details(capsys):
    root, _, _, _ = make_scope_with_borrow()
    print_scopes(root)
    out = capsys.readouterr().out
    assert "borrow" in out


def test_evaluate_node_error_branches():
    root = Scope("root")
    env = create_default_environment()
    node_j = Node("J", "int32", root)
    root.nodes.append(node_j)
    with pytest.raises(RuntimeError):
        evaluate_node(node_j, env)

    node_k = Node("K", "int32", root)
    root.nodes.append(node_k)
    node_k.borrows.append(Borrow("shared", node_j.owned_life, root))
    node_j.owned_life.borrows.append(node_k.borrows[0])
    env[node_j.owned_life.id] = "not-capability"
    with pytest.raises(RuntimeError):
        evaluate_node(node_k, env)

    node_l = Node("L", "int32", root)
    root.nodes.append(node_l)
    node_l.borrows.append(Borrow("shared", node_k.owned_life, root))
    node_k.owned_life.borrows.append(node_l.borrows[0])
    env[node_k.owned_life.id] = "bad"
    with pytest.raises(RuntimeError):
        evaluate_node(node_l, env)


def test_node_to_dict_and_reconstruct_scope_paths():
    root, first, _, _ = make_scope_with_borrow()
    first.ffi_capabilities = ["io.write"]
    entry = runtime_mod._node_to_dict(first)
    assert entry.get("ffi_capabilities") == ["io.write"]

    doc = scope_to_dict(root)
    recon = reconstruct_scope(doc)
    assert isinstance(recon, Scope)


def test_certificate_validation_errors():
    with pytest.raises(ValueError):
        runtime_mod._validate_certificate("alias", None, {"ok": True, "payload_digest": "0", "summary": {}})

    with pytest.raises(ValueError):
        runtime_mod._validate_certificate(
            "grades",
            {"ok": True, "payload_digest": "1", "summary": {}},
            {"ok": True, "payload_digest": "0", "summary": {}},
        )


def test_diff_bitcodes_node_count_branch(tmp_path, capsys):
    root_a, _, _, _ = make_scope_with_borrow()
    result_a = evaluate_scope(root_a)
    doc_a = build_bitcode_document(root_a, result_a)

    root_b, first_b, second_b, _ = make_scope_with_borrow()
    extra = Node("C", "int32", root_b)
    root_b.nodes.append(extra)
    extra.update_type()
    extra.owned_life.end_scope = root_b
    root_b.lifetimes.append(extra.owned_life)
    root_b.drops.append(extra.owned_life)
    result_b = evaluate_scope(root_b)
    doc_b = build_bitcode_document(root_b, result_b)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    write_bitcode_document(doc_a, left_path)
    write_bitcode_document(doc_b, right_path)
    diff_bitcodes(str(left_path), str(right_path))
    assert "Node count differs" in capsys.readouterr().out


def test_build_tir_match_cases_branch():
    root_scope = Scope("root")
    ctor = Node("A", "int32", root_scope)
    root_scope.nodes.append(ctor)
    ctor.update_type()
    match_node = Node("P", "match", root_scope)
    match_node.meta["match_cases"] = [(("A", 0), "arm")]
    root_scope.nodes.append(match_node)
    borrow = Borrow("shared", ctor.owned_life, root_scope)
    match_node.borrows.append(borrow)
    ctor.owned_life.borrows.append(borrow)
    program = build_tir(root_scope)
    assert any(
        instr.metadata.get("cases") for instr in program.instructions if instr.op == "SWITCH"
    )


def test_compute_tir_distance_variants():
    base = TIRProgram()
    base.instructions.append(TIRInstruction("v0", "A", "int32", "pure", [], "root"))
    other = TIRProgram()
    other.instructions.append(TIRInstruction("v0", "D", "int32", "pure", [], "root"))
    other.instructions.append(TIRInstruction("v1", "A", "int32", "pure", ["v0"], "root"))
    distance = compute_tir_distance(base, other)
    assert distance["total"] >= 1


def test_export_wasm_nested_paths(tmp_path):
    tir = TIRProgram()
    io_instr = TIRInstruction("v1", "G", "int32", "io", [], "root")
    tir.instructions.append(io_instr)
    export_wasm_module(
        tir,
        tmp_path / "nested" / "module.wat",
        capabilities={"io.write"},
        metadata_path=tmp_path / "meta" / "info.json",
    )
    assert (tmp_path / "nested" / "module.wat").exists()


def test_reflect_variants():
    root, first, _, _ = make_scope_with_borrow()
    assert reflect(root).kind == "Scope"
    assert reflect(first).kind == "Node"
    assert reflect(TIRProgram()).kind == "TIR"
    assert reflect(123).kind == "Value"


def test_optimizer_alias_and_inlining_paths():
    tir = TIRProgram()
    v0 = tir.emit("A", "int32", "pure", [], "root.child")
    dup = TIRInstruction("v1", "ADD", "int32", "pure", [v0], "root.child")
    dup.value = 1
    tir.instructions.append(dup)
    dup2 = TIRInstruction("v2", "ADD", "int32", "pure", [v0], "root.child")
    dup2.value = 1
    tir.instructions.append(dup2)
    common_subexpression_elimination(tir)
    inline_pure_regions(tir)
    assert tir.instructions


def test_reorder_pure_ops_dependency_guards():
    tir = TIRProgram()
    a = TIRInstruction("v0", "A", "int32", "pure", [], "root")
    b = TIRInstruction("v1", "D", "int32", "pure", ["v0"], "root.fence")
    tir.instructions.extend([a, b])
    result = reorder_pure_ops(tir)
    assert result.instructions

def test_metaobject_repr_and_dict():
    root, first, _, _ = make_scope_with_borrow()
    meta_scope = MetaObject("Scope", root)
    meta_node = MetaObject("Node", first)
    assert "MetaScope" in repr(meta_scope)
    assert "MetaNode" in repr(meta_node)
    assert isinstance(meta_scope.to_dict(), dict)
    assert meta_node.to_dict()["op"] == first.op


def test_structural_decompress_nested_caps():
    structural_decompress("{[a]}")


def test_evaluate_node_type_error_path():
    root = Scope("root")
    source = Node("A", "int32", root)
    root.nodes.append(source)
    target = Node("E", "int32", root)
    root.nodes.append(target)
    borrow = Borrow("shared", source.owned_life, root)
    target.borrows.append(borrow)
    source.owned_life.borrows.append(borrow)
    env = create_default_environment()
    env[source.owned_life.id] = "text"
    effect = evaluate_node(target, env)
    assert effect.value == 3


def test_evaluate_scope_existing_env():
    root, _, _, _ = make_scope_with_borrow()
    env = {"__capabilities__": {"FileRead": runtime_mod.CAPABILITY_FACTORIES["FileRead"]()}}
    evaluate_scope(root, env)
    for kind in runtime_mod.CAPABILITY_FACTORIES:
        assert kind in env["__capabilities__"]


def test_reconstruct_scope_with_ffi_metadata():
    root, first, _, _ = make_scope_with_borrow()
    doc = scope_to_dict(root)
    node_doc = doc["nodes"][0]
    decl = FFIDeclaration("FFI_ADD", "io", ["int32"], "int32")
    node_doc["ffi"] = decl.to_dict()
    node_doc["ffi_capabilities"] = ["io.write"]
    recon = reconstruct_scope(doc)
    assert recon.nodes[0].ffi.name == "FFI_ADD"


def test_compute_tir_distance_borrow_rewire():
    prog_a = TIRProgram()
    instr_a = TIRInstruction("v0", "A", "int32", "pure", [], "root")
    prog_a.instructions.append(instr_a)
    prog_b = TIRProgram()
    instr_b = TIRInstruction("v0", "A", "int32", "pure", ["missing"], "root")
    prog_b.instructions.append(instr_b)
    distance = compute_tir_distance(prog_a, prog_b)
    assert distance["borrow_rewires"] >= 1
