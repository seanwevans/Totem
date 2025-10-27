"""Totem Intermediate Representation and lowering utilities."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Iterable, Optional

from ..constants import EFFECT_GRADES


class TIRInstruction:
    """Single SSA-like instruction."""

    def __init__(
        self,
        id,
        op,
        typ,
        grade,
        args,
        scope_path,
        metadata=None,
        produces=None,
    ):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = args
        self.scope_path = scope_path
        self.metadata = metadata or {}
        self.produces = produces

    def __repr__(self):  # pragma: no cover - representation helper
        def fmt_arg(arg):
            if isinstance(arg, dict):
                target = arg.get("target")
                kind = arg.get("kind")
                if kind and target:
                    return f"{kind}:{target}"
                if target:
                    return str(target)
                return json.dumps(arg, sort_keys=True)
            return str(arg)

        args = ", ".join(fmt_arg(arg) for arg in self.args)
        return f"{self.id} = {self.op}({args}) : {self.typ} [{self.grade}]"


class TIRProgram:
    """In-memory representation of a Totem IR program."""

    def __init__(self):
        self.instructions: list[TIRInstruction] = []
        self.next_id = 0
        self.constructor_tags: dict[tuple[str, int], int] = {}
        self.next_tag = 0

    def new_id(self):
        vid = f"v{self.next_id}"
        self.next_id += 1
        return vid

    def emit(self, op, typ, grade, args, scope_path, metadata=None, produces=None):
        vid = self.new_id()
        instr = TIRInstruction(vid, op, typ, grade, args, scope_path, metadata, produces)
        self.instructions.append(instr)
        return vid

    def clone(self):
        program = TIRProgram()
        program.instructions = [deepcopy(instr) for instr in self.instructions]
        program.next_id = self.next_id
        program.constructor_tags = dict(self.constructor_tags)
        program.next_tag = self.next_tag
        return program

    def constructor_tag(self, op, arity):
        key = (op, arity)
        if key not in self.constructor_tags:
            self.constructor_tags[key] = self.next_tag
            self.next_tag += 1
        return self.constructor_tags[key]

    def desugar_pattern_matches(self):
        """Lower MATCH instructions into SWITCHes on constructor tags."""

        lowered = []
        for instr in self.instructions:
            if instr.op != "MATCH":
                lowered.append(instr)
                continue

            cases = instr.metadata.get("cases", [])
            default = instr.metadata.get("default")
            switch_meta = {
                "cases": [
                    {
                        "tag": case.get("tag"),
                        "result": case.get("result"),
                        "constructor": case.get("constructor"),
                    }
                    for case in cases
                ]
            }
            if default is not None:
                switch_meta["default"] = default

            lowered.append(
                TIRInstruction(
                    instr.id,
                    "SWITCH",
                    instr.typ,
                    instr.grade,
                    [
                        arg.get("target") if isinstance(arg, dict) else arg
                        for arg in instr.args
                    ],
                    instr.scope_path,
                    switch_meta,
                )
            )

        self.instructions = lowered
        return self

    def __repr__(self):  # pragma: no cover - debugging helper
        return "\n".join(map(str, self.instructions))


@dataclass(frozen=True)
class TranspilationResult:
    """Result of lowering Totem source into Totem IR."""

    source: str
    tree: "Scope"
    tir: TIRProgram
    errors: tuple[str, ...]
    optimized: bool


def _mlir_type(typ):
    """Map Totem types to MLIR types."""

    mapping = {
        "int32": "i32",
        "int64": "i64",
        "float": "f32",
        "double": "f64",
    }
    return mapping.get(typ, "i32")


def emit_mlir_module(tir):
    """Lower a TIR program to a textual MLIR module."""

    lattice = ", ".join(f'"{grade}"' for grade in EFFECT_GRADES)
    lines = [
        f"module attributes {{totem.effect_lattice = [{lattice}]}} {{",
        "  func.func @main() -> () {",
    ]

    value_map = {}

    for instr in tir.instructions:
        result_name = instr.id
        operands = []
        operand_types = []
        borrow_kinds = []

        for arg in instr.args:
            if isinstance(arg, dict):
                target = arg.get("target")
                kind = arg.get("kind")
            else:
                target = arg
                kind = None

            if not target:
                continue

            operand = value_map.get(target, target)
            operands.append(f"%{operand}")
            operand_types.append(_mlir_type(instr.typ))
            if kind:
                borrow_kinds.append(f'"{kind}"')

        operand_sig = ", ".join(operand_types)
        operand_sig = f"{operand_sig}" if operand_sig else ""

        attrs = [f'grade = "{instr.grade}"']
        if borrow_kinds:
            attrs.append(f"borrow_kinds = [{', '.join(borrow_kinds)}]")

        attr_str = " " + "{" + ", ".join(attrs) + "}" if attrs else ""

        operand_list = ", ".join(operands)
        line = (
            f"    %{result_name} = \"totem.{instr.op.lower()}\"({operand_list}) : "
            f"({operand_sig}) -> {_mlir_type(instr.typ)}{attr_str}"
        )
        lines.append(line.rstrip())

        value_map[instr.id] = result_name
        if instr.produces:
            value_map[instr.produces] = result_name

    lines.append("    func.return")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines)


PURE_CONSTANTS = {
    "A": 1,
    "D": 2,
    "F": 5,
}


def emit_llvm_ir(tir):
    """Emit a simple LLVM IR view for the pure portion of a TIR program."""

    pure_instrs = [instr for instr in tir.instructions if instr.grade == "pure"]
    if not pure_instrs:
        return "; Totem program has no pure segment to lower"

    lines = [
        "; Totem pure segment lowered to LLVM IR",
        "define void @totem_main() {",
        "entry:",
    ]

    value_map = {}
    declared = set()

    def map_operand(target):
        return f"%{value_map.get(target, target)}"

    for instr in pure_instrs:
        result_name = instr.id

        if instr.op in PURE_CONSTANTS:
            const_val = PURE_CONSTANTS[instr.op]
            lines.append(f"  %{result_name} = add i32 0, {const_val}")
        else:
            operands = []
            for arg in instr.args:
                if isinstance(arg, dict):
                    target = arg.get("target")
                else:
                    target = arg
                if not target:
                    continue
                operands.append(map_operand(target))

            operand_parts = [f"i32 {op}" for op in operands]
            callee = f"@totem_{instr.op.lower()}"
            declared.add(callee)
            call_operands = ", ".join(operand_parts)
            lines.append(
                f"  %{result_name} = call i32 {callee}({call_operands})"
                if call_operands
                else f"  %{result_name} = call i32 {callee}()"
            )

        value_map[instr.id] = result_name
        if instr.produces:
            value_map[instr.produces] = result_name

    lines.append("  ret void")
    lines.append("}")

    for callee in sorted(declared):
        lines.append(f"declare i32 {callee}(...)")

    return "\n".join(lines).rstrip()


class BytecodeInstruction:
    """Executable instruction in the bytecode VM."""

    __slots__ = ("origin_id", "op", "grade", "args", "produces")

    def __init__(self, origin_id, op, grade, args=None, produces=None):
        self.origin_id = origin_id
        self.op = op
        self.grade = grade
        self.args = args or []
        self.produces = produces


class BytecodeProgram:
    """Linear bytecode representation assembled from TIR."""

    def __init__(self, instructions=None):
        self.instructions = instructions or []

    def append(self, instruction):
        self.instructions.append(instruction)


class BytecodeResult:
    """Execution artefact from the bytecode VM."""

    def __init__(self, grade, log, stack, env):
        self.grade = grade
        self.log = log
        self.stack = stack
        self.env = env


class BytecodeVM:
    """A minimal stack-based interpreter for Totem TIR."""

    def __init__(self):
        self.stack = []
        self.env = {}
        self.log = []
        self._effect_index = 0

    def execute(self, program):
        for instr in program.instructions:
            self._step(instr)

        final_grade = EFFECT_GRADES[self._effect_index]
        return BytecodeResult(final_grade, list(self.log), list(self.stack), dict(self.env))

    # -- internal helpers -------------------------------------------------

    def _step(self, instr):
        grade_index = self._grade_index(instr.grade)
        self._effect_index = max(self._effect_index, grade_index)
        value, log_entries = self._apply_operation(instr)

        if value is not None:
            self.stack.append(value)
        self.env[instr.origin_id] = value
        if instr.produces:
            self.env[instr.produces] = value

        if log_entries:
            if isinstance(log_entries, (list, tuple)):
                self.log.extend(log_entries)
            else:
                self.log.append(log_entries)

    def _grade_index(self, grade):
        try:
            return EFFECT_GRADES.index(grade)
        except ValueError:
            return 0

    def _apply_operation(self, instr):
        op = instr.op

        if op == "A":
            value = 1
            return value, [f"A:{value}"]
        if op == "B":
            self.env["counter"] = self.env.get("counter", 0) + 1
            value = self.env["counter"]
            return value, [f"B:inc->{value}"]
        if op == "C":
            value = "input_data"
            return value, [f"C:read->{value}"]
        if op == "D":
            value = 2
            return value, [f"D:{value}"]
        if op == "E":
            base = 0
            if instr.args:
                target = instr.args[0][1]
                base = self.env.get(target, 0)
            value = base + 3
            return value, [f"E:{value}"]
        if op == "F":
            value = 5
            return value, [f"F:{value}"]
        if op == "G":
            target = instr.args[0][1] if instr.args else None
            borrowed = self.env.get(target, "?")
            value = True
            return value, [f"G:write({borrowed})"]

        # Fallback: produce zero value with a log entry for traceability.
        value = 0
        return value, [f"{op}:{value}"]


def assemble_bytecode(tir):
    """Linearise a TIR program into bytecode instructions."""

    program = BytecodeProgram()
    for instr in tir.instructions:
        args = []
        for arg in instr.args:
            if isinstance(arg, dict):
                args.append((arg.get("kind"), arg.get("target")))
            else:
                args.append((None, arg))
        program.append(BytecodeInstruction(instr.id, instr.op, instr.grade, args, instr.produces))
    return program


def run_bytecode(program):
    """Execute a BytecodeProgram and return the resulting effect/log/stack."""

    vm = BytecodeVM()
    return vm.execute(program)


__all__ = [
    "BytecodeInstruction",
    "BytecodeProgram",
    "BytecodeResult",
    "BytecodeVM",
    "PURE_CONSTANTS",
    "TIRInstruction",
    "TIRProgram",
    "TranspilationResult",
    "_mlir_type",
    "assemble_bytecode",
    "emit_llvm_ir",
    "emit_mlir_module",
    "run_bytecode",
]
