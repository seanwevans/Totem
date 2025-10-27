"""Tests for the incremental verifier/linter kernel."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import totem.runtime as runtime_mod


def test_incremental_verifier_tracks_grade_updates():
    verifier = runtime_mod.IncrementalVerifier("ab")

    first = verifier.lint()
    assert first.ok
    assert first.computed_grade == "state"

    cached = verifier.lint()
    assert cached is first

    updated = verifier.apply_edit(len("ab"), len("ab"), "c")
    assert updated is not first
    assert updated.computed_grade == "io"
    assert updated.ok


def test_incremental_verifier_rejects_invalid_edits():
    verifier = runtime_mod.IncrementalVerifier("abc")

    with pytest.raises(ValueError):
        verifier.apply_edit(-1, 0, "z")

    with pytest.raises(ValueError):
        verifier.apply_edit(2, 1, "z")

    with pytest.raises(ValueError):
        verifier.apply_edit(0, 10, "z")


def test_incremental_verifier_reports_alias_conflicts(monkeypatch):
    def fake_decompress(_src):
        root = runtime_mod.Scope("root")
        first = runtime_mod.Node("A", "int32", root)
        root.nodes.append(first)
        first.update_type()
        second = runtime_mod.Node("B", "int32", root)
        root.nodes.append(second)
        second.update_type()

        for node in (first, second):
            node.owned_life.end_scope = root
            if node.owned_life not in root.lifetimes:
                root.lifetimes.append(node.owned_life)
                root.drops.append(node.owned_life)

        mut_borrow = runtime_mod.Borrow("mut", first.owned_life, root)
        shared_borrow = runtime_mod.Borrow("shared", first.owned_life, root)
        second.borrows.extend([mut_borrow, shared_borrow])
        first.owned_life.borrows.extend([mut_borrow, shared_borrow])
        second.update_type()

        return root

    monkeypatch.setattr(runtime_mod, "structural_decompress", fake_decompress)

    verifier = runtime_mod.IncrementalVerifier("xy")
    result = verifier.lint()

    assert not result.aliasing["ok"]
    assert any("Aliasing violation" in msg for msg in result.diagnostics)


def test_incremental_verifier_respects_inline_ffi():
    decl = runtime_mod.FFIDeclaration("X", "io", [], "int32")

    verifier = runtime_mod.IncrementalVerifier("x", ffi_decls=[decl])
    result = verifier.lint()

    assert result.computed_grade == "io"
    assert result.ok
    assert "X" not in runtime_mod.FFI_REGISTRY
