import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    compile_and_evaluate,
    parse_inline_ffi,
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
