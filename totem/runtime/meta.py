"""Reflection primitives used by the Totem runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import Node, Scope
from .tir import TIRProgram

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from .tir import TIRInstruction


class MetaObject:
    """A serializable reflection of a Totem runtime object."""

    def __init__(self, kind, data):
        self.kind = kind
        self.data = data

    def __repr__(self):  # pragma: no cover - debugging helper
        if self.kind == "Node":
            n: Node = self.data
            return f"<MetaNode {n.op}:{n.typ}@{n.scope.name} [{n.grade}]>"
        if self.kind == "Scope":
            s: Scope = self.data
            return f"<MetaScope {s.name} nodes={len(s.nodes)}>"
        if self.kind == "TIR":
            t: TIRProgram = self.data
            return f"<MetaTIR {len(t.instructions)} instrs>"
        return f"<MetaObject {self.kind}>"

    def to_dict(self):
        """Return a JSON-safe representation."""
        if self.kind == "TIR":
            program: TIRProgram = self.data
            return [instr.__dict__ for instr in program.instructions]
        if self.kind in ("Node", "Scope"):
            return self.data.__dict__
        return {"kind": self.kind}


def reflect(obj):
    """Produce a MetaObject view of any Totem structure."""

    if isinstance(obj, Scope):
        return MetaObject("Scope", obj)
    if isinstance(obj, Node):
        return MetaObject("Node", obj)
    if isinstance(obj, TIRProgram):
        return MetaObject("TIR", obj)
    return MetaObject("Value", obj)


def meta_emit(
    program: TIRProgram,
    op,
    typ="int32",
    grade="pure",
    args=None,
    scope_path="root.meta",
):
    """Dynamically extend a TIR program with a new instruction."""

    args = args or []
    vid = program.emit(op, typ, grade, args, scope_path)
    instr = program.instructions[-1]
    return MetaObject("TIR_Instruction", instr)


def _unwrap_meta_tir(value):
    if isinstance(value, MetaObject):
        if value.kind != "TIR":
            raise RuntimeError("Meta operation requires a TIR MetaObject")
        return value.data
    if isinstance(value, TIRProgram):
        return value
    raise RuntimeError("Meta operation requires a TIR MetaObject")


def list_meta_ops():
    return {
        "reflect": "Return a MetaObject view of a Totem structure",
        "meta_emit": "Append a new TIR instruction (Meta effect)",
        "list_meta_ops": "List available reflective primitives",
    }


__all__ = [
    "MetaObject",
    "_unwrap_meta_tir",
    "list_meta_ops",
    "meta_emit",
    "reflect",
]
