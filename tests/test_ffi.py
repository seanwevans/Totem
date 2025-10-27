import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (  # noqa: E402
    FFIDeclaration,
    clear_ffi_registry,
    compile_and_evaluate,
    get_registered_ffi_declarations,
    parse_inline_ffi,
    register_ffi_declarations,
)


def test_parse_inline_ffi_schema():
    schema = "H:io(int32)->bytes|requires console,fs"
    decls = parse_inline_ffi(schema)
    assert len(decls) == 1
    decl = decls[0]
    assert decl.name == "H"
    assert decl.grade == "io"
    assert decl.arg_types == ["int32"]
    assert decl.return_type == "bytes"
    assert decl.arity == 1
    assert decl.capabilities == ["console", "fs"]


def test_compile_with_inline_ffi_validates_arity():
    valid_spec = parse_inline_ffi("H:io(int32)->int32")
    tree, errors, result = compile_and_evaluate("{ah}", ffi_decls=valid_spec)
    assert not errors
    scope = tree.children[0]
    ffi_node = scope.nodes[1]
    assert ffi_node.ffi is not None
    assert ffi_node.typ == "int32"
    assert any(entry.startswith("FFI:H") for entry in result.log)

    invalid_spec = parse_inline_ffi("H:io(int32,int32)->int32")
    _, bad_errors, _ = compile_and_evaluate("{ah}", ffi_decls=invalid_spec)
    assert any("arity mismatch" in msg for msg in bad_errors)


def test_json_ffi_spec_support():
    json_spec = [
        {
            "name": "J",
            "grade": "state",
            "arg_types": ["int32"],
            "return_type": "string",
            "capabilities": ["fs"],
        }
    ]
    tree, errors, result = compile_and_evaluate("{aj}", ffi_decls=json_spec)
    assert not errors
    node = tree.children[0].nodes[1]
    assert node.ffi is not None
    assert node.grade == "state"
    assert node.typ == "string"
    assert node.ffi_capabilities == ["fs"]
    assert any(entry.startswith("FFI:J") for entry in result.log)


def test_register_ffi_declarations_supports_various_spec_shapes():
    clear_ffi_registry()

    inline_spec = """
    # comment only line should be ignored
    X:io(int32)->bytes|requires console
    """

    legacy_dict = {"declarations": [{"name": "Y", "grade": "state", "arg_types": []}]}
    ffi_list_wrapper = {"ffi": [{"name": "Z", "grade": "pure", "arg_types": ["int32"]}]}
    object_decl = FFIDeclaration("W", "sys", ["int32", "bytes"], "void", ["net"])
    json_blob = json.dumps([
        {
            "name": "Q",
            "grade": "pure",
            "arg_types": [],
            "return_type": "void",
        }
    ])

    register_ffi_declarations(
        [inline_spec, legacy_dict, ffi_list_wrapper, object_decl, json_blob],
        reset=True,
    )

    registry = get_registered_ffi_declarations()
    assert set(registry) == {"X", "Y", "Z", "W", "Q"}
    assert registry["X"].capabilities == ["console"]
    assert registry["Z"].arity == 1
    assert registry["W"].capabilities == ["net"]


def test_register_ffi_declarations_rejects_invalid_specs():
    clear_ffi_registry()

    with pytest.raises(TypeError):
        register_ffi_declarations(42, reset=True)

    register_ffi_declarations({"name": "OK", "grade": "pure"}, reset=True)
    with pytest.raises(ValueError):
        register_ffi_declarations({"name": "OK", "grade": "pure"})


def test_ffi_declaration_from_dict_supports_alias_keys():
    decl = FFIDeclaration.from_dict(
        {
            "name": "Alias",
            "grade": "io",
            "args": ["int32"],
            "returns": "bytes",
            "requires": ["fs"],
        }
    )

    assert decl.name == "ALIAS"
    assert decl.grade == "io"
    assert decl.arg_types == ["int32"]
    assert decl.return_type == "bytes"
    assert decl.capabilities == ["fs"]
    assert decl.arity == 1
