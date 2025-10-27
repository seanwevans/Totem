import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import totem.runtime as runtime_module
from totem.runtime import (
    CAPABILITY_FACTORIES,
    ActorCapability,
    ActorSystem,
    BytecodeInstruction,
    BytecodeProgram,
    BytecodeVM,
    CapabilityUseResult,
    Lifetime,
    Node,
    Scope,
    OwnedMessage,
    TIRInstruction,
    TIRProgram,
    Effect,
    _mutate_byte,
    _scope_full_path,
    _scope_path,
    _stable_id,
    _arity_type_name,
    assemble_bytecode,
    build_bitcode_document,
    build_bitcode_certificates,
    build_tir,
    canonicalize_bitcode,
    compute_tir_distance,
    continuous_semantics_profile,
    create_default_environment,
    extract_capability,
    hash_bitcode,
    hash_bitcode_document,
    emit_llvm_ir,
    emit_mlir_module,
    ensure_capability,
    load_totem_bitcode,
    move_env_value,
    reexecute_bitcode,
    read_env_value,
    resolve_value,
    run_bytecode,
    show_logbook,
    store_capability,
    explain_grade,
    explain_borrow,
    parse_inline_ffi,
    verify_bitcode_document,
    write_bitcode_document,
    diff_bitcodes,
    reconstruct_scope,
    structural_decompress,
    use_file_read,
    use_file_write,
    use_net_send,
    compile_and_evaluate,
    visualize_graph,
    export_graphviz,
)


def test_capabilities_read_write_and_net_flow():
    env = create_default_environment()
    file_read = env["__capabilities__"]["FileRead"]

    first_result = use_file_read(file_read)
    assert not file_read.is_active
    assert isinstance(first_result, CapabilityUseResult)
    assert first_result.capability.generation == 1
    assert first_result.value == "input_data"

    with pytest.raises(RuntimeError):
        use_file_read(file_read)

    second_result = use_file_read(first_result.capability)
    assert second_result.value is None

    file_write = ensure_capability(env, "FileWrite")
    write_result = use_file_write(file_write, "payload")
    assert resolve_value(write_result) is True
    assert extract_capability(write_result) == write_result.capability
    store_capability(env, "FileWrite", write_result.capability)

    net_cap = ensure_capability(env, "NetSend")
    net_result = use_net_send(net_cap, "ping")
    assert "sent:ping" == resolve_value(net_result)
    store_capability(env, "NetSend", net_result.capability)

    raw_cap = CAPABILITY_FACTORIES["FileWrite"]()
    assert resolve_value(raw_cap) is raw_cap
    assert extract_capability(raw_cap) is raw_cap
    assert extract_capability("not a capability") is None


def test_scope_helpers_and_effect_and_repr():
    root = Scope("root")
    child = Scope("child", root)
    node = Node("A", "int32", root)
    helper_path = _scope_path(child)
    assert helper_path.endswith("child")
    assert "root" in _scope_full_path(child)
    stable = _stable_id(helper_path, 0)
    assert len(stable) == 12
    assert _arity_type_name(2) == "ADT<2>"
    effect = Effect("pure", 2, ["start"])

    def to_state(val):
        return Effect("state", val + 1, ["inc"])

    bound = effect.bind(to_state)
    assert bound.grade == "state"
    assert bound.log[-1] == "inc"
    repr_text = repr(node)
    assert node.op in repr_text

    actor_system = ActorSystem()
    cap = actor_system.spawn()
    actor_system.run_until_idle()
    assert isinstance(actor_system.last_public_log, list)


def test_read_env_value_and_move_semantics():
    env = {"life": 42}
    assert read_env_value(env, "life") == 42
    assert read_env_value(env, None, default="sentinel") == "sentinel"
    assert read_env_value(env, "missing", default=0) == 0

    moved = move_env_value(env, "life")
    assert moved == 42
    with pytest.raises(RuntimeError):
        read_env_value(env, "life")
    with pytest.raises(RuntimeError):
        move_env_value(env, "life")
    with pytest.raises(RuntimeError):
        move_env_value({}, None)
    with pytest.raises(KeyError):
        read_env_value({}, "unknown")
    with pytest.raises(KeyError):
        move_env_value({}, "unknown")


def test_actor_system_message_flow_and_errors():
    system = ActorSystem()
    cap = system.spawn()
    msg = OwnedMessage("hello", cap, system.next_message_id())

    effect = cap.send(msg)
    assert effect.grade == "sys"
    assert effect.log == [f"send:{cap.actor_id}:msg{msg.message_id}"]

    with pytest.raises(RuntimeError):
        cap.send(msg)

    other_cap = system.spawn()
    wrong_msg = OwnedMessage("oops", cap, system.next_message_id())
    with pytest.raises(RuntimeError):
        other_cap.send(wrong_msg)

    ghost_cap = ActorCapability(system, "ghost")
    ghost_msg = OwnedMessage("boo", ghost_cap, system.next_message_id())
    with pytest.raises(RuntimeError):
        system.send(ghost_cap, ghost_msg)

    run_effect = system.run_until_idle()
    assert run_effect.grade == "sys"
    assert run_effect.value is system
    assert run_effect.log[0] == "run:delivered=1"
    assert system.last_public_log == run_effect.log[1:]


def _make_sample_tir():
    program = TIRProgram()
    program.instructions = [
        TIRInstruction(
            "v0",
            "MATCH",
            "int32",
            "pure",
            [{"target": "v_in"}],
            "root.fn",
            metadata={
                "cases": [
                    {"constructor": ("A", 0), "tag": 0, "result": "then"},
                ],
                "default": "else",
            },
        ),
        TIRInstruction("v1", "A", "int32", "pure", [], "root.fn", produces="life"),
        TIRInstruction("v2", "F", "int32", "pure", ["v1"], "root.fn"),
        TIRInstruction(
            "v3", "G", "int32", "io", [{"kind": "borrow", "target": "v1"}], "root.fn"
        ),
    ]
    return program


def test_tir_transforms_and_distance_metrics():
    tir = _make_sample_tir()
    lowered = tir.desugar_pattern_matches()
    assert lowered.instructions[0].op == "SWITCH"
    assert lowered.instructions[0].metadata["cases"][0]["constructor"] == ("A", 0)

    other = _make_sample_tir()
    other.instructions[1] = TIRInstruction(
        "u1", "B", "int32", "state", [], "root.fn", produces="life"
    )

    dist = compute_tir_distance(tir, other)
    assert dist["total"] >= 1
    assert dist["op_changes"] >= 1 or dist["grade_delta"] >= 1

    collision = TIRProgram()
    collision.instructions = [
        TIRInstruction("c0", "A", "int32", "pure", [], "root.fn", produces="shared"),
        TIRInstruction("c1", "D", "int32", "pure", [], "root.fn", produces="shared"),
    ]
    zero_dist = compute_tir_distance(collision, collision)
    assert zero_dist["total"] == 0

    mlir = emit_mlir_module(tir)
    assert "module attributes" in mlir
    assert "totem.switch" in mlir or "totem.a" in mlir

    llvm = emit_llvm_ir(tir)
    assert "define void @totem_main" in llvm

    assert emit_llvm_ir(TIRProgram()) == "; Totem program has no pure segment to lower"


def test_bytecode_vm_execution_and_assembly():
    tir = _make_sample_tir()
    program = assemble_bytecode(tir)
    vm_result = run_bytecode(program)
    assert vm_result.grade == "io"
    assert vm_result.log
    assert vm_result.env

    manual_program = BytecodeProgram(
        [
            BytecodeInstruction("x0", "A", "pure"),
            BytecodeInstruction("x1", "B", "state"),
            BytecodeInstruction("x2", "E", "pure", [("borrow", "x0")]),
            BytecodeInstruction("x3", "G", "io", [("consume", "x2")]),
            BytecodeInstruction("x4", "Z", "unknown"),
        ]
    )

    vm = BytecodeVM()
    result = vm.execute(manual_program)
    assert result.grade == "io"
    assert any(entry.startswith("Z:") for entry in result.log)
    assert result.env["x1"] == 1
    assert result.env["x2"] == 4


def test_continuous_semantics_profile_handles_mutations():
    src = "{ad}"
    tree = structural_decompress(src)
    base_tir = build_tir(tree)
    profile = continuous_semantics_profile(
        src, base_tir=base_tir, mutate_fn=lambda c: chr(ord(c) + 1)
    )
    assert all(entry["index"] >= 0 for entry in profile)
    assert any("distance" in entry for entry in profile)

    error_profile = continuous_semantics_profile(
        src,
        base_tir=base_tir,
        mutate_fn=lambda _ch: "x",
    )
    assert any(entry.get("error") for entry in error_profile)


def test_bitcode_roundtrip_and_diff(tmp_path, capsys):
    tree, errors, result = compile_and_evaluate("{ad}")
    assert not errors
    doc = build_bitcode_document(tree, result)

    file_a = tmp_path / "a.totem.json"
    write_bitcode_document(doc, file_a)

    loaded = load_totem_bitcode(file_a)
    assert verify_bitcode_document(loaded)

    reconstructed = reconstruct_scope(loaded["root_scope"])
    assert reconstructed.name == "root"

    canon = canonicalize_bitcode(loaded)
    assert canon["evaluation"]["final_grade"] == result.grade

    digest = hash_bitcode_document(loaded)
    assert len(digest) == 64
    assert hash_bitcode(str(file_a)) == digest

    tree_b, errors_b, result_b = compile_and_evaluate("{ac}")
    assert not errors_b
    doc_b = build_bitcode_document(tree_b, result_b)
    file_b = tmp_path / "b.totem.json"
    write_bitcode_document(doc_b, file_b)

    diff_bitcodes(str(file_a), str(file_b))
    diff_output = capsys.readouterr().out
    assert "Bitcodes" in diff_output

    diff_bitcodes(str(file_a), str(file_a))
    identical_output = capsys.readouterr().out
    assert "identical" in identical_output

    tampered = json.loads(json.dumps(loaded))
    tampered["certificates"]["grades"]["payload_digest"] = "deadbeef"
    with pytest.raises(ValueError):
        verify_bitcode_document(tampered)

    rerun = reexecute_bitcode(str(file_a))
    assert rerun.grade == result.grade


def test_logbook_display(monkeypatch, tmp_path, capsys):
    temp_log = tmp_path / "totem.logbook.jsonl"
    monkeypatch.setattr(runtime_module, "LOGBOOK_FILE", str(temp_log))

    show_logbook()
    first_output = capsys.readouterr().out
    assert "No logbook yet." in first_output

    entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "filename": "a.totem.json",
            "hash": "a" * 64,
            "final_grade": "pure",
            "first_log": "start",
            "last_log": "end",
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "filename": "b.totem.json",
            "hash": "b" * 64,
            "final_grade": "state",
            "first_log": "hello",
            "last_log": "bye",
        },
    ]
    with open(temp_log, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    show_logbook(limit=5)
    second_output = capsys.readouterr().out
    assert "Totem Logbook" in second_output
    assert "a.totem.json" in second_output


def test_visualization_guards():
    tree, errors, _ = compile_and_evaluate("{ad}")
    assert not errors
    with pytest.raises(RuntimeError):
        visualize_graph(tree)
    with pytest.raises(ModuleNotFoundError):
        export_graphviz(tree, "out.svg")


def test_mutate_byte_wrapping():
    assert _mutate_byte(" ") == "!"
    assert _mutate_byte("~") == " "
    assert _mutate_byte("\u0000") == "\u0001"


def test_build_bitcode_certificates_failures():
    scope = Scope("root")
    life_a = Lifetime(scope, "life")
    duplicate = Lifetime(scope, "life")
    scope.lifetimes.extend([life_a, duplicate])

    with pytest.raises(ValueError):
        build_bitcode_certificates(scope, "pure")

    mismatch_scope = Scope("root")
    Node("B", "int32", mismatch_scope)
    with pytest.raises(ValueError):
        build_bitcode_certificates(mismatch_scope, "meta")


def test_structural_decompress_error_paths():
    with pytest.raises(ValueError):
        structural_decompress("}")
    with pytest.raises(ValueError):
        structural_decompress("{a")


def test_explain_grade_and_borrow():
    tree, errors, result = compile_and_evaluate("{ag}")
    assert not errors
    grade_info = explain_grade(tree, result.grade)
    assert grade_info["achieved"]
    assert grade_info["nodes"]

    borrow_info = explain_borrow(tree, grade_info["nodes"][0].owned_life.id)
    assert borrow_info["found"]


def test_aliasing_and_node_helpers():
    tree, errors, result = compile_and_evaluate("{ad}")
    assert not errors
    alias_payload = runtime_module._collect_aliasing_payload(tree)
    assert alias_payload["ok"]
    grade_cert = runtime_module._grade_certificate(tree, result.grade)
    assert grade_cert["ok"]
    scopes = list(runtime_module.iter_scopes(tree))
    first_node = next(node for scope in scopes for node in scope.nodes)
    node_dict = runtime_module._node_to_dict(first_node)
    assert node_dict["id"] == first_node.id
    scope_dict = runtime_module.scope_to_dict(tree)
    assert scope_dict["name"] == "root"


def test_node_to_dict_with_ffi_metadata():
    ffi_spec = parse_inline_ffi("H:io(int32)->int32")
    tree, errors, result = compile_and_evaluate("{ah}", ffi_decls=ffi_spec)
    assert not errors
    nodes = [node for scope in runtime_module.iter_scopes(tree) for node in scope.nodes]
    ffi_node = next(node for node in nodes if node.ffi)
    entry = runtime_module._node_to_dict(ffi_node)
    assert "ffi" in entry
    assert entry["ffi"]["name"] == "H"


def test_module_entrypoint(monkeypatch, tmp_path, capsys):
    temp_log = tmp_path / "log.jsonl"
    monkeypatch.setattr(runtime_module, "LOGBOOK_FILE", str(temp_log))
    argv = sys.argv[:]
    sys.argv = ["totem", "--logbook"]
    try:
        runpy.run_module("totem.__main__", run_name="__main__")
    finally:
        sys.argv = argv
    output = capsys.readouterr().out
    assert "Totem" in output or "No logbook" in output
