"""Tests for the reflective helpers in ``totem.runtime.meta``."""

from __future__ import annotations

import pytest

from totem.runtime.core import Node, Scope
from totem.runtime.meta import (
    MetaObject,
    _unwrap_meta_tir,
    list_meta_ops,
    meta_emit,
    reflect,
)
from totem.runtime.tir import TIRProgram


def test_meta_object_to_dict_for_tir_program():
    program = TIRProgram()
    program.emit("CONST", "int32", "pure", [1], "root")

    meta = MetaObject("TIR", program)
    as_dict = meta.to_dict()

    assert isinstance(as_dict, list)
    assert as_dict[0]["op"] == "CONST"
    assert as_dict[0]["args"] == [1]


def test_meta_object_to_dict_for_runtime_structures():
    scope = Scope("root")
    node = Node("ADD", "int32", scope)
    scope.nodes.append(node)

    scope_meta = MetaObject("Scope", scope)
    node_meta = MetaObject("Node", node)

    assert scope_meta.to_dict()["name"] == "root"
    assert node_meta.to_dict()["op"] == "ADD"


def test_meta_object_to_dict_for_other_values():
    meta = MetaObject("Custom", {"ignored": True})

    assert meta.to_dict() == {"kind": "Custom"}


def test_reflect_identifies_runtime_types():
    scope = Scope("root")
    node = Node("ADD", "int32", scope)
    tir = TIRProgram()

    assert reflect(scope).kind == "Scope"
    assert reflect(node).kind == "Node"
    assert reflect(tir).kind == "TIR"
    assert reflect(123).kind == "Value"


def test_meta_emit_appends_instruction_and_wraps_result():
    program = TIRProgram()

    meta_instr = meta_emit(program, "CONST", args=None)

    assert meta_instr.kind == "TIR_Instruction"
    assert len(program.instructions) == 1
    emitted = program.instructions[0]
    assert emitted.args == []
    assert emitted.op == "CONST"
    assert emitted.typ == "int32"
    assert emitted.grade == "pure"


def test_unwrap_meta_tir_accepts_metaobject_and_program():
    program = TIRProgram()
    meta = MetaObject("TIR", program)

    assert _unwrap_meta_tir(meta) is program
    assert _unwrap_meta_tir(program) is program


def test_unwrap_meta_tir_rejects_incorrect_values():
    with pytest.raises(RuntimeError):
        _unwrap_meta_tir(MetaObject("Node", object()))

    with pytest.raises(RuntimeError):
        _unwrap_meta_tir(object())


def test_list_meta_ops_reports_available_helpers():
    ops = list_meta_ops()

    assert set(ops) == {"reflect", "meta_emit", "list_meta_ops"}
