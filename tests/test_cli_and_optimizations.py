import json
import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    TIRProgram,
    build_bitcode_document,
    evaluate_scope,
    list_meta_ops,
    list_optimizers,
    explain_borrow,
    meta_emit,
    optimize_tir,
    parse_args,
    record_run,
    reflect,
    reexecute_bitcode,
    show_logbook,
    structural_decompress,
    write_bitcode_document,
)
from totem import runtime


def _build_sample_doc(src="{ad}"):
    tree = structural_decompress(src)
    result = evaluate_scope(tree)
    return tree, result, build_bitcode_document(tree, result)


def test_optimize_tir_and_meta_operations():
    program = TIRProgram()
    const_a = program.emit("A", "int32", "pure", [], "root")
    const_d = program.emit("D", "int32", "pure", [], "root")
    partial_add = program.emit("ADD", "int32", "pure", [const_a, "external"], "root.child")
    program.emit("ADD", "int32", "pure", [const_a, const_d], "root.child")
    program.emit("C", "int32", "io", [partial_add], "root")

    optimize_tir(program)

    assert program.instructions[0].op == "ADD"
    assert program.instructions[0].partial_constant == 1
    assert program.instructions[1].op == "CONST_IO"

    second = TIRProgram()
    const_a = second.emit("A", "int32", "pure", [], "root")
    const_d = second.emit("D", "int32", "pure", [], "root")
    folded_add = second.emit("ADD", "int32", "pure", [const_a, const_d], "root.child")
    second.emit("C", "int32", "io", [folded_add], "root")

    optimize_tir(second)

    assert second.instructions[0].op == "CONST"
    assert getattr(second.instructions[0], "value", None) == 3

    meta_program = TIRProgram()
    meta_program.emit("A", "int32", "pure", [], "root")
    meta_info = list_meta_ops()
    assert "reflect" in meta_info and "meta_emit" in meta_info
    meta_obj = reflect(meta_program)
    assert meta_obj.kind == "TIR"
    emitted = meta_emit(meta_program, "Z", args=["v0"], scope_path="root.meta")
    assert emitted.kind == "TIR_Instruction"
    assert meta_program.instructions[-1].scope_path == "root.meta"
    assert "inline_trivial_io" in list_optimizers()


def test_bitcode_hash_diff_logbook_and_reexecution(tmp_path, capsys, monkeypatch):
    tree, result, doc = _build_sample_doc("{ad}")
    bitcode_path = tmp_path / "program.totem.json"
    write_bitcode_document(doc, bitcode_path)

    file_hash = runtime.hash_bitcode_document(doc)
    runtime.hash_bitcode(str(bitcode_path))
    captured = capsys.readouterr().out
    assert file_hash in captured

    _, _, mutated_doc = _build_sample_doc("{ac}")
    mutated_path = tmp_path / "mutated.totem.json"
    write_bitcode_document(mutated_doc, mutated_path)

    runtime.diff_bitcodes(str(bitcode_path), str(mutated_path))
    diff_output = capsys.readouterr().out
    assert "Bitcodes differ" in diff_output

    monkeypatch.setattr(runtime, "LOGBOOK_FILE", str(tmp_path / "logbook.jsonl"), raising=False)
    monkeypatch.setattr(runtime, "sign_hash", lambda sha: f"sig:{sha}")

    entry = record_run(str(bitcode_path), result)
    record_output = capsys.readouterr().out
    assert entry["hash"] == file_hash
    assert "Recorded and signed run" in record_output

    show_logbook(limit=5)
    log_output = capsys.readouterr().out
    assert "Totem Logbook" in log_output

    rerun = reexecute_bitcode(str(bitcode_path))
    reexec_output = capsys.readouterr().out
    assert rerun.grade == result.grade
    assert "Loaded Totem Bitcode" in reexec_output


def test_crypto_helpers_require_dependencies():
    with pytest.raises(RuntimeError):
        runtime.ensure_keypair()

    with pytest.raises(RuntimeError):
        runtime.sign_hash("deadbeef")

    with pytest.raises(RuntimeError):
        runtime.verify_signature("deadbeef", "cafebabe")


def test_main_argument_branches(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(runtime, "diff_bitcodes", lambda a, b: calls.append(("diff", a, b)))
    runtime.main(["--diff", "fileA", "fileB"])
    assert calls == [("diff", "fileA", "fileB")]

    calls.clear()
    monkeypatch.setattr(runtime, "hash_bitcode", lambda path: calls.append(("hash", path)))
    runtime.main(["--hash", "program.totem.json"])
    assert calls == [("hash", "program.totem.json")]

    calls.clear()
    monkeypatch.setattr(runtime, "reexecute_bitcode", lambda path: calls.append(("load", path)))
    runtime.main(["--load", "program.totem.json"])
    assert calls == [("load", "program.totem.json")]

    calls.clear()
    monkeypatch.setattr(runtime, "show_logbook", lambda limit=10: calls.append(("logbook", limit)))
    runtime.main(["--logbook"])
    assert calls == [("logbook", 10)]

    calls.clear()
    monkeypatch.setattr(runtime, "run_repl", lambda : calls.append(("repl", None)))
    runtime.main(["--repl"])
    assert calls == [("repl", None)]

    calls.clear()
    monkeypatch.setattr(runtime, "verify_signature", lambda hash_value, sig: calls.append((hash_value, sig)) or True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "feedface")
    runtime.main(["--verify", "abc123"])
    verify_output = capsys.readouterr().out
    assert ("abc123", "feedface") in calls
    assert "Signature valid" in verify_output

    monkeypatch.setattr(runtime, "record_run", lambda filename, result: None)
    runtime.main(["--src", "a"])
    default_output = capsys.readouterr().out
    assert "Runtime evaluation:" in default_output

    params = parse_args([
        "--src",
        "{ab}",
        "--capability",
        "io.read",
        "--capability",
        "io.write",
        "--why-grade",
        "io",
        "--why-borrow",
        "life-1",
    ])
    assert params.capabilities == ["io.read", "io.write"]
    assert params.why_grade == "io"
    assert params.why_borrow == "life-1"


def test_main_with_analysis_overrides(monkeypatch, capsys):
    sample_tree = structural_decompress("{ab}")
    fake_result = type("Result", (), {"grade": "state", "log": ["log-entry"]})()

    monkeypatch.setattr(runtime, "compile_and_evaluate", lambda src: (sample_tree, ["error"], fake_result))
    monkeypatch.setattr(runtime, "print_scopes", lambda tree: None)
    monkeypatch.setattr(runtime, "explain_grade", lambda tree, grade: {
        "achieved": True,
        "final_grade": "state",
        "nodes": [],
    })
    monkeypatch.setattr(runtime, "explain_borrow", lambda tree, ident: {
        "identifier": ident,
        "origin": "root",
        "chain": [],
        "found": True,
        "lines": ["Borrow chain"],
    })
    monkeypatch.setattr(runtime, "export_totem_bitcode", lambda scope, effect, filename="program.totem.json": None)
    monkeypatch.setattr(runtime, "record_run", lambda filename, result: None)

    runtime.main([
        "--src",
        "{ab}",
        "--why-grade",
        "state",
        "--why-borrow",
        "life-1",
    ])

    output = capsys.readouterr().out
    assert "Grade not reached" not in output
    assert "Borrow analysis" in output
    assert "Borrow chain" in output


def test_run_repl_handles_commands(monkeypatch, capsys, tmp_path):
    commands = iter(
        [
            "{a}",
            ":hash",
            ":bitcode",
            f":save 1 {tmp_path / 'program.json'}",
            ":viz",
            ":diff 1 1",
            ":quit",
        ]
    )

    def fake_input(prompt=""):
        try:
            return next(commands)
        except StopIteration:
            raise EOFError

    saved = []
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(runtime, "visualize_graph", lambda tree: saved.append("viz"))
    monkeypatch.setattr(runtime, "write_bitcode_document", lambda doc, filename: saved.append(str(filename)))
    monkeypatch.setattr(runtime, "record_run", lambda filename, result: None)
    monkeypatch.setattr(runtime, "export_totem_bitcode", lambda scope, effect, filename="program.totem.json": None)

    runtime.run_repl(history_limit=2)

    output = capsys.readouterr().out
    assert "Totem REPL" in output
    assert "SHA256" in output
    assert any(str(tmp_path / "program.json") in item for item in saved)


def test_module_main_invokes_runtime_main(monkeypatch):
    captured_args = []
    monkeypatch.setattr(runtime, "main", lambda args: captured_args.append(args))
    monkeypatch.setattr(sys, "argv", ["python", "--src", "demo"])

    from importlib import reload

    mod = reload(__import__("totem.__main__", fromlist=["_run"]))
    mod._run()

    assert captured_args == [["--src", "demo"]]


def test_explain_borrow_describes_chain():
    scope = structural_decompress("ab")
    first_life = scope.nodes[0].owned_life.id
    info = explain_borrow(scope, first_life)

    assert info["found"] is True
    assert any("Lifetime" in line for line in info["lines"])

    missing = explain_borrow(scope, "unknown")
    assert missing["found"] is False


def test_export_graphviz_with_stub(monkeypatch, tmp_path):
    import types

    class FakeNode:
        def __init__(self, name, **kwargs):
            self.name = name

    class FakeCluster:
        def __init__(self, name, **kwargs):
            self.name = name
            self.nodes = []
            self.subgraphs = []

        def add_node(self, node):
            self.nodes.append(node)

        def add_subgraph(self, sub):
            self.subgraphs.append(sub)

    class FakeEdge:
        def __init__(self, src, dst, **kwargs):
            self.src = src
            self.dst = dst

    class FakeDot:
        def __init__(self, *args, **kwargs):
            self.subgraphs = []
            self.edges = []

        def add_subgraph(self, sub):
            self.subgraphs.append(sub)

        def add_edge(self, edge):
            self.edges.append(edge)

        def write_svg(self, path):
            Path(path).write_text("<svg/>", encoding="utf-8")

    fake_pydot = types.ModuleType("pydot")
    fake_pydot.Dot = FakeDot
    fake_pydot.Cluster = FakeCluster
    fake_pydot.Node = FakeNode
    fake_pydot.Edge = FakeEdge

    monkeypatch.setitem(sys.modules, "pydot", fake_pydot)
    monkeypatch.setattr(runtime, "pydot", fake_pydot, raising=False)

    scope = structural_decompress("{ab}")
    output = tmp_path / "graph.svg"
    runtime.export_graphviz(scope, output)

    assert output.exists()
    assert output.read_text(encoding="utf-8") == "<svg/>"


def test_visualize_graph_requires_optional_dependencies():
    with pytest.raises(RuntimeError):
        runtime.visualize_graph(structural_decompress("{a}"))
