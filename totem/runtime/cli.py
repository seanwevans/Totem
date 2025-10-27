"""Command-line interface for the Totem runtime."""
from __future__ import annotations

import argparse
import difflib
import json
import sys

from ..constants import REPL_HISTORY_LIMIT
from .analysis import (
    explain_borrow,
    explain_grade,
    print_scopes,
    visualize_graph,
    export_graphviz,
)
from .bitcode import (
    build_bitcode_document,
    canonicalize_bitcode,
    diff_bitcodes,
    export_totem_bitcode,
    hash_bitcode,
    hash_bitcode_document,
    record_run,
    reexecute_bitcode,
    show_logbook,
    write_bitcode_document,
)
from .compiler import (
    build_tir,
    compile_and_evaluate,
    continuous_semantics_profile,
    export_wasm_module,
)
from .core import _scope_full_path
from .crypto import verify_signature
from .tir import emit_llvm_ir, emit_mlir_module

def _runtime_callable(name, fallback):
    runtime_mod = sys.modules.get('totem.runtime')
    if runtime_mod and hasattr(runtime_mod, name):
        return getattr(runtime_mod, name)
    return fallback

def run_repl(history_limit=REPL_HISTORY_LIMIT):  # pragma: no cover
    """Interactive Totem shell."""

    print("Totem REPL — enter program bytes or commands (:help for help)")
    history = []
    counter = 0

    def resolve_entry(token=None):
        if not history:
            print("No cached programs yet.")
            return None
        if token is None:
            return history[-1]
        try:
            target = int(token)
        except ValueError:
            print("Program index must be an integer.")
            return None
        for entry in reversed(history):
            if entry["index"] == target:
                return entry
        print(f"No cached program #{target}.")
        return None

    while True:
        try:
            line = input("totem> ")
        except EOFError:
            print()
            break

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(":"):
            parts = stripped.split()
            cmd = parts[0]

            if cmd in (":quit", ":exit"):
                break
            if cmd == ":help":
                print(
                    "Commands: :help, :quit, :viz [n], :save [n] [file], :hash [n], :bitcode [n], :diff n m"
                )
                print(f"History: last {history_limit} programs cached.")
                continue
            if cmd == ":viz":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if entry:
                    _runtime_callable('visualize_graph', visualize_graph)(entry["tree"])
                continue
            if cmd == ":save":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if not entry:
                    continue
                if len(parts) > 2:
                    filename = parts[2]
                else:
                    filename = f"program_{entry['index']}.totem.json"
                _runtime_callable('write_bitcode_document', write_bitcode_document)(entry["bitcode_doc"], filename)
                continue
            if cmd == ":hash":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if entry:
                    h = _runtime_callable('hash_bitcode_document', hash_bitcode_document)(entry["bitcode_doc"])
                    print(f"SHA256(program_{entry['index']}) = {h}")
                continue
            if cmd == ":bitcode":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if entry:
                    canon = _runtime_callable('canonicalize_bitcode', canonicalize_bitcode)(entry["bitcode_doc"])
                    print(json.dumps(canon, indent=2))
                continue
            if cmd == ":diff":
                if len(parts) != 3:
                    print("Usage: :diff <a> <b>")
                    continue
                entry_a = resolve_entry(parts[1])
                entry_b = resolve_entry(parts[2])
                if not entry_a or not entry_b:
                    continue
                canon_a = _runtime_callable('canonicalize_bitcode', canonicalize_bitcode)(entry_a["bitcode_doc"])
                canon_b = _runtime_callable('canonicalize_bitcode', canonicalize_bitcode)(entry_b["bitcode_doc"])
                text_a = json.dumps(canon_a, indent=2, sort_keys=True).splitlines()
                text_b = json.dumps(canon_b, indent=2, sort_keys=True).splitlines()
                diff = list(
                    difflib.unified_diff(
                        text_a,
                        text_b,
                        fromfile=f"program_{entry_a['index']}",
                        tofile=f"program_{entry_b['index']}",
                        lineterm="",
                    )
                )
                if diff:
                    for line in diff:
                        print(line)
                else:
                    print("Programs are identical.")
                continue

            print(f"Unknown command: {cmd}")
            continue

        tree, errors, result = _runtime_callable('compile_and_evaluate', compile_and_evaluate)(line)
        counter += 1
        entry = {
            "index": counter,
            "src": line,
            "tree": tree,
            "errors": errors,
            "result": result,
            "bitcode_doc": _runtime_callable('build_bitcode_document', build_bitcode_document)(tree, result),
        }
        history.append(entry)
        if len(history) > history_limit:
            history.pop(0)

        print(f"[#%d] grade: %s" % (entry["index"], result.grade))
        if errors:
            print("  analysis:")
            for e in errors:
                print("   ", f"✗ {e}")
        else:
            print("  analysis: ✓ All lifetime and borrow checks passed")
        print("  log:")
        if result.log:
            for item in result.log:
                print("   ", item)
        else:
            print("    (no log entries)")


def parse_args(args):
    argp = argparse.ArgumentParser(description="Totem Language Runtime")

    argp.add_argument(
        "--diff",
        nargs=2,
        metavar=("A", "B"),
        help="Compare two .totem.json Bitcode files",
    )
    argp.add_argument("--hash", help="Compute hash of a .totem.json Bitcode file")
    argp.add_argument("--load", help="Load a .totem.json Bitcode file instead")
    argp.add_argument(
        "--logbook", action="store_true", help="Show Totem provenance logbook"
    )
    argp.add_argument("--repl", action="store_true", help="Start an interactive REPL")
    argp.add_argument("--src", help="Inline Totem source", default="{a{bc}de{fg}}")
    argp.add_argument("--verify", help="Verify signature for a logbook entry hash")
    argp.add_argument(
        "--visualize",
        nargs="?",
        const="purity",
        metavar="SCRIPT",
        help=(
            "Render program graph; optionally provide a visualization DSL script "
            "(e.g. 'purity', 'fence', 'purity+fence', 'animate:lifetime')"
        ),
    )
    argp.add_argument(
        "--viz",
        metavar="OUTPUT",
        help="Export a Graphviz scope visualization to an SVG file",
    )
    argp.add_argument(
        "--wasm",
        metavar="OUTPUT",
        help="Export the pure TIR as a WebAssembly text module",
    )
    argp.add_argument(
        "--wasm-metadata",
        metavar="OUTPUT",
        help="Write lowering metadata alongside the WebAssembly module",
    )
    argp.add_argument(
        "--capability",
        action="append",
        dest="capabilities",
        help="Grant a capability (e.g. io.read) when lowering to WebAssembly",
    )
    argp.add_argument(
        "--why-grade",
        metavar="GRADE",
        help="Explain which nodes raised the program to a given effect grade",
    )
    argp.add_argument(
        "--why-borrow",
        metavar="ID",
        help="Explain the borrow chain for a lifetime or node identifier",
    )

    return argp.parse_args(args)


def main(args):  # pragma: no cover
    params = parse_args(args)

    if params.diff:
        _runtime_callable('diff_bitcodes', diff_bitcodes)(params.diff[0], params.diff[1])
        return
    if params.hash:
        _runtime_callable('hash_bitcode', hash_bitcode)(params.hash)
        return
    if params.load:
        _runtime_callable('reexecute_bitcode', reexecute_bitcode)(params.load)
        return
    if params.logbook:
        _runtime_callable('show_logbook', show_logbook)()
        return
    if params.repl:
        _runtime_callable('run_repl', run_repl)()
        return
    if params.verify:
        ok = _runtime_callable('verify_signature', verify_signature)(
            params.verify,
            input("Signature hex: ").strip(),
        )
        print("✓ Signature valid" if ok else "✗ Invalid signature")
        return

    print("Source:", params.src)
    tree, errors, result = compile_and_evaluate(params.src)
    print_scopes(tree)

    print("\nCompile-time analysis:")
    if not errors:
        print("  ✓ All lifetime and borrow checks passed")
    else:
        for e in errors:
            print("  ✗", e)

    print("\nRuntime evaluation:")
    print(f"  → final grade: {result.grade}")
    print("  → execution log:")
    for entry in result.log:
        print("   ", entry)

    if params.why_grade:
        print(f"\nWhy grade '{params.why_grade}':")
        try:
            info = _runtime_callable('explain_grade', explain_grade)(tree, params.why_grade)
        except ValueError as exc:
            print(f"  ✗ {exc}")
        else:
            if not info["achieved"]:
                print(
                    "  "
                    + "Grade not reached. Final grade: "
                    + info["final_grade"]
                )
            elif not info["nodes"]:
                print("  No nodes with that grade were found.")
            else:
                print("  Minimal cut responsible for the requested grade:")
                for node in info["nodes"]:
                    scope_path = _scope_full_path(node.scope)
                    print(
                        f"    • {node.op} [{node.grade}] id={node.id} @ {scope_path}"
                    )
                    if node.borrows:
                        borrow_desc = ", ".join(
                            f"{b.kind}->{b.target.id}" for b in node.borrows
                        )
                        print(f"        borrows: {borrow_desc}")

    if params.why_borrow:
        print(f"\nBorrow analysis for '{params.why_borrow}':")
        info = _runtime_callable('explain_borrow', explain_borrow)(tree, params.why_borrow)
        if not info["found"]:
            print("  ✗ Identifier not found in this program.")
        else:
            for line in info["lines"]:
                print("  " + line)

    _runtime_callable('export_totem_bitcode', export_totem_bitcode)(tree, result, "program.totem.json")
    _runtime_callable('record_run', record_run)("program.totem.json", result)
    tir = _runtime_callable('build_tir', build_tir)(tree)
    print("\nTIR:")
    print(tir)

    profile = _runtime_callable('continuous_semantics_profile', continuous_semantics_profile)(params.src, base_tir=tir)
    print("\nContinuous semantics (Δ per byte):")
    if not profile:
        print("  (no bytes to mutate)")
    else:
        for entry in profile:
            dist = entry["distance"]
            print(
                "  idx {idx}: {orig!r}->{mut!r} Δtotal={total} "
                "(nodes={nodes}, grades={grades}, borrows={borrows})".format(
                    idx=entry["index"],
                    orig=entry["original"],
                    mut=entry["mutated"],
                    total=dist["total"],
                    nodes=dist["node_edits"],
                    grades=dist["grade_delta"],
                    borrows=dist["borrow_rewires"],
                )
            )
    if params.wasm:
        try:
            _runtime_callable('export_wasm_module', export_wasm_module)(
                tir,
                params.wasm,
                capabilities=params.capabilities,
                metadata_path=params.wasm_metadata,
            )
        except PermissionError as exc:
            print(f"  ✗ {exc}")
        except NotImplementedError as exc:
            print(f"  ✗ {exc}")
    mlir_module = _runtime_callable('emit_mlir_module', emit_mlir_module)(tir)
    print("\nMLIR dialect:")
    print(mlir_module)

    llvm_ir = _runtime_callable('emit_llvm_ir', emit_llvm_ir)(tir)
    print("\nLLVM IR (pure segment):")
    print(llvm_ir)

    if params.viz:
        export_graphviz(tree, params.viz)
    if params.visualize:
        visualize_graph(tree, params.visualize)



__all__ = [
    "main",
    "parse_args",
    "run_repl",
]


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
