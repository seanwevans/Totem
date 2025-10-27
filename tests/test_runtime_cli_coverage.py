"""Additional coverage-focused tests for ``totem.runtime.cli``."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from totem.runtime import cli as runtime_cli
from totem import runtime


def test_runtime_callable_prefers_runtime_module(monkeypatch):
    """Ensure ``_runtime_callable`` resolves attributes correctly."""

    sentinel = object()

    module_with_attr = SimpleNamespace(custom=lambda: "runtime-value")
    monkeypatch.setitem(sys.modules, "totem.runtime", module_with_attr)
    resolved = runtime_cli._runtime_callable("custom", lambda: "fallback")
    assert resolved() == "runtime-value"

    module_without_attr = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "totem.runtime", module_without_attr)
    fallback = lambda: sentinel  # noqa: E731 - simple inline callable for the test
    assert runtime_cli._runtime_callable("custom", fallback) is fallback

    monkeypatch.delitem(sys.modules, "totem.runtime", raising=False)
    assert runtime_cli._runtime_callable("custom", fallback)() is sentinel


def test_parse_args_covers_optional_flags():
    """Verify optional CLI arguments populate the expected defaults."""

    params = runtime_cli.parse_args(
        [
            "--visualize",
            "--viz",
            "output.svg",
            "--wasm",
            "module.wat",
            "--wasm-metadata",
            "meta.json",
            "--capability",
            "io.read",
        ]
    )

    assert params.visualize == "purity"
    assert params.viz == "output.svg"
    assert params.wasm == "module.wat"
    assert params.wasm_metadata == "meta.json"
    assert params.capabilities == ["io.read"]


def _stub_main_environment(monkeypatch, *, errors=None, profile=None):
    """Install minimal stubs so ``runtime_cli.main`` executes quickly."""

    tree = SimpleNamespace(name="tree")
    result = SimpleNamespace(grade="io", log=["log-entry"])

    monkeypatch.setattr(
        runtime_cli, "compile_and_evaluate", lambda src: (tree, errors or [], result)
    )
    monkeypatch.setattr(runtime_cli, "print_scopes", lambda tree: None)

    monkeypatch.setattr(
        runtime, "export_totem_bitcode", lambda tree, res, filename: None
    )
    monkeypatch.setattr(runtime, "record_run", lambda filename, res: None)
    monkeypatch.setattr(runtime, "build_tir", lambda tree: "fake-tir")
    monkeypatch.setattr(
        runtime,
        "continuous_semantics_profile",
        lambda src, base_tir: profile if profile is not None else [],
    )
    monkeypatch.setattr(runtime, "emit_mlir_module", lambda tir: "mlir-module")
    monkeypatch.setattr(runtime, "emit_llvm_ir", lambda tir: "llvm-ir")
    monkeypatch.setattr(runtime_cli, "export_graphviz", lambda tree, output: None)
    monkeypatch.setattr(runtime_cli, "visualize_graph", lambda tree, script=None: None)

    return tree, result


def test_main_reports_unachieved_grade_and_missing_borrow(monkeypatch, capsys):
    tree, _ = _stub_main_environment(monkeypatch)

    def explain_grade_stub(scope, grade):
        return {"achieved": False, "final_grade": "pure", "nodes": []}

    def explain_borrow_stub(scope, ident):
        return {"found": False, "lines": []}

    def export_wasm_stub(*args, **kwargs):
        raise NotImplementedError("lowering unavailable")

    monkeypatch.setattr(runtime, "explain_grade", explain_grade_stub)
    monkeypatch.setattr(runtime, "explain_borrow", explain_borrow_stub)
    monkeypatch.setattr(runtime, "export_wasm_module", export_wasm_stub)

    runtime_cli.main(
        [
            "--src",
            "{abc}",
            "--why-grade",
            "sys",
            "--why-borrow",
            "life-1",
            "--wasm",
            "module.wat",
        ]
    )

    output = capsys.readouterr().out
    assert "Grade not reached" in output
    assert "Identifier not found" in output
    assert "✗ lowering unavailable" in output


def test_main_describes_grade_nodes_and_profile(monkeypatch, capsys):
    calls: list[tuple[str, object]] = []
    tree, result = _stub_main_environment(
        monkeypatch,
        profile=[
            {
                "index": 0,
                "original": "a",
                "mutated": "b",
                "distance": {
                    "total": 2,
                    "node_edits": 1,
                    "grade_delta": 1,
                    "borrow_rewires": 0,
                },
            }
        ],
    )

    class Scope:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent

    parent_scope = Scope("root")
    child_scope = Scope("child", parent=parent_scope)

    fake_node = SimpleNamespace(
        op="ADD",
        grade="io",
        id=42,
        scope=child_scope,
        borrows=[SimpleNamespace(kind="ref", target=SimpleNamespace(id="life-9"))],
    )

    def explain_grade_stub(scope, grade):
        return {"achieved": True, "final_grade": result.grade, "nodes": [fake_node]}

    def explain_borrow_stub(scope, ident):
        return {"found": True, "lines": ["Borrow chain detail"]}

    def export_wasm_stub(tir, output, *, capabilities=None, metadata_path=None):
        calls.append(("wasm", output, tuple(capabilities or ()), metadata_path))
        raise PermissionError("capability required")

    def export_graphviz_stub(tree, output):
        calls.append(("viz", output))

    def visualize_stub(tree, script=None):
        calls.append(("visualize", script))

    monkeypatch.setattr(runtime, "explain_grade", explain_grade_stub)
    monkeypatch.setattr(runtime, "explain_borrow", explain_borrow_stub)
    monkeypatch.setattr(runtime, "export_wasm_module", export_wasm_stub)
    monkeypatch.setattr(runtime_cli, "export_graphviz", export_graphviz_stub)
    monkeypatch.setattr(runtime_cli, "visualize_graph", visualize_stub)

    runtime_cli.main(
        [
            "--src",
            "{abc}",
            "--why-grade",
            "io",
            "--why-borrow",
            "life-9",
            "--viz",
            "graph.svg",
            "--visualize",
            "custom-script",
            "--wasm",
            "output.wat",
            "--wasm-metadata",
            "meta.json",
            "--capability",
            "io.read",
            "--capability",
            "io.write",
        ]
    )

    output = capsys.readouterr().out
    assert "Minimal cut responsible" in output
    assert "borrows: ref->life-9" in output
    assert "Borrow chain detail" in output
    assert "idx 0" in output
    assert "capability required" in output

    assert ("wasm", "output.wat", ("io.read", "io.write"), "meta.json") in calls
    assert ("viz", "graph.svg") in calls
    assert ("visualize", "custom-script") in calls
