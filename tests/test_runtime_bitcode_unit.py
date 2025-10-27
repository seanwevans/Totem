from datetime import datetime, timezone

import pytest

from totem.runtime.bitcode import (
    _validate_certificate,
    reconstruct_scope,
    verify_bitcode_document,
)


def _base_cert():
    """Helper providing a canonical certificate payload for tests."""

    return {"payload_digest": "digest", "summary": "ok", "ok": True}


def test_reconstruct_scope_restores_complex_tree():
    root_scope_dict = {
        "name": "root",
        "effect_cap": "io",
        "fence": "strict",
        "nodes": [
            {
                "op": "CUSTOM_FFI",
                "type": "void",
                "id": "node-ffi",
                "grade": "io",
                "lifetime_id": "life-ffi",
                "meta": {"fixed_type": "u64"},
                "ffi": {
                    "name": "CUSTOM_FFI",
                    "grade": "io",
                    "arg_types": ["u32"],
                    "return_type": "u64",
                    "capabilities": ["net"],
                },
                "borrows": [
                    {"kind": "shared", "target": "shared-life"},
                ],
            },
            {
                "op": "NODE",
                "type": "ADT<0>",
                "id": "node-plain",
                "grade": "pure",
                "lifetime_id": "life-plain",
                "ffi_capabilities": ["disk"],
                "borrows": [
                    {"kind": "unique", "target": "missing-life"},
                ],
            },
        ],
        "lifetimes": [
            {"id": "shared-life"},
            {"id": "unused-life"},
        ],
        "drops": ["shared-life", "nonexistent-life"],
        "children": [
            {
                "name": "child",
                "nodes": [
                    {
                        "op": "CHILD",
                        "type": "void",
                        "id": "child-node",
                        "grade": "pure",
                        "lifetime_id": "child-owned",
                        "meta": {"fixed_type": "child-type"},
                        "borrows": [
                            {"kind": "shared", "target": "child-external"},
                        ],
                    }
                ],
                "lifetimes": [
                    {"id": "child-external", "end_scope": "root"},
                ],
                "drops": ["child-external"],
                "children": [],
            }
        ],
    }

    scope = reconstruct_scope(root_scope_dict)

    assert scope.effect_cap == "io"
    assert scope.fence == "strict"
    assert len(scope.nodes) == 2

    ffi_node, plain_node = scope.nodes
    assert ffi_node.ffi is not None
    assert ffi_node.ffi.name == "CUSTOM_FFI"
    assert ffi_node.typ == "u64"
    assert ffi_node.ffi_capabilities == ["net"]
    assert [b.target.id for b in ffi_node.borrows] == ["shared-life"]

    # Borrow targeting an unknown lifetime is ignored.
    assert plain_node.ffi is None
    assert plain_node.ffi_capabilities == ["disk"]
    assert plain_node.borrows == []

    shared_life, unused_life = scope.lifetimes
    assert shared_life.id == "shared-life"
    assert shared_life.borrows[0].borrower_scope is scope
    assert unused_life.id == "unused-life"
    assert scope.drops == [shared_life]

    child = scope.children[0]
    assert child.parent is scope
    child_life = child.lifetimes[0]
    assert child_life.id == "child-external"
    assert child_life.end_scope is scope
    assert child.nodes[0].borrows[0].target is child_life
    assert child.nodes[0].typ == "child-type"


@pytest.mark.parametrize(
    "stored,expected,message",
    [
        (None, _base_cert(), "missing aliasing certificate"),
        (
            {"payload_digest": "other", "summary": "ok", "ok": True},
            _base_cert(),
            "aliasing certificate digest mismatch",
        ),
        (
            {"payload_digest": "digest", "summary": "different", "ok": True},
            _base_cert(),
            "aliasing certificate summary mismatch",
        ),
        (
            {"payload_digest": "digest", "summary": "failure", "ok": False},
            {"payload_digest": "digest", "summary": "failure", "ok": True},
            "aliasing certificate indicates failure: failure",
        ),
        (
            {"payload_digest": "digest", "summary": "expected", "ok": True},
            {"payload_digest": "digest", "summary": "expected", "ok": False},
            "aliasing certificate recomputation failed: expected",
        ),
    ],
)
def test_validate_certificate_error_paths(stored, expected, message):
    with pytest.raises(ValueError, match=message):
        _validate_certificate("aliasing", stored, expected)


def test_verify_bitcode_document_requires_certificates():
    doc = {
        "totem_version": "0.5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root_scope": {"name": "root", "nodes": [], "children": []},
        "evaluation": {"final_grade": "pure", "log": []},
    }

    with pytest.raises(ValueError, match="Totem bitcode missing proof certificates"):
        verify_bitcode_document(doc)
