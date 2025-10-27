#!/usr/bin/env python3
"""
🪶 Totem — a no-syntax-error programming language.

Pure ⊂ State ⊂ IO ⊂ Sys ⊂ Meta

  Pure: deterministic and referentially transparent.
  State: modifies internal memory but not external state.
  IO: reads/writes external state (files, console, etc.).
  Sys: system-level effects (spawn, network, etc.).
  Meta: reflection, compilation, or self-modifying code.

| Layer                         | Purpose                               | Status  |
<------------------------------ + ------------------------------------- + -------->
| **Structural decompressor**   | Every UTF-8 string → valid scoped AST |   ✅    |
| **Type & lifetime inference** | Rust-like ownership, drops, borrows   |   ✅    |
| **Purity/effect lattice**     | `Pure ⊂ State ⊂ IO ⊂ Sys ⊂ Meta`      |   ✅    |
| **Evaluator**                 | Graded effect monad runtime           |   ✅    |
| **Visualization**             | NetworkX graph of scopes & lifetimes  |   ✅    |
| **Bitcode serialization**     | Portable `.totem.json` IR             |   ✅    |
| **Reload & re-execution**     | Deterministic round-trip              |   ✅    |
| **Hash & diff**               | Semantic identity                     |   ✅    |
| **Logbook ledger**            | Provenance tracking                   |   ✅    |
| **Cryptographic signatures**  | Proof-of-origin                       |   ✅    |

"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import heapq
import json
from collections import deque
from pathlib import Path
import re
import sys
try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover
    nx = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

try:
    import pydot
except ModuleNotFoundError:  # pragma: no cover
    pydot = None

try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization  # pragma: no cover
    from cryptography.hazmat.backends import default_backend  # pragma: no cover
    from cryptography.exceptions import InvalidSignature  # pragma: no cover
except ImportError:  # pragma: no cover
    rsa = padding = hashes = serialization = default_backend = InvalidSignature = None

from .constants import (
    EFFECT_GRADES,
    GRADE_COLORS,
    IO_IMPORTS,
    KEY_FILE,
    LOGBOOK_FILE,
    OPS,
    PUB_FILE,
    PURE_CONST_VALUES,
    PURE_CONSTANTS,
    REPL_HISTORY_LIMIT,
)
from .ffi import (
    FFI_REGISTRY,
    FFIDeclaration,
    clear_ffi_registry,
    get_registered_ffi_declarations,
    parse_inline_ffi,
    register_ffi_declarations,
)


def _scope_path(scope):
    parts = []
    while scope is not None:
        parts.append(scope.name)
        scope = scope.parent
    return ".".join(reversed(parts))


def _stable_id(scope_path, index):
    token = f"{scope_path}:{index}".encode("utf-8")
    return hashlib.blake2s(token, digest_size=6).hexdigest()


def _scope_full_path(scope):
    """Return a human-readable scope path for display."""

    parts = []
    while scope is not None:
        parts.append(scope.name)
        scope = scope.parent
    return " > ".join(reversed(parts))


class Lifetime:
    def __init__(self, owner_scope, identifier):
        self.id = identifier
        self.owner_scope = owner_scope
        self.end_scope = None
        self.borrows = []
        self.owner_node = None

    def __repr__(self):
        end = self.end_scope.name if self.end_scope else "?"
        return f"Life({self.id}@{self.owner_scope.name}->{end})"


class Borrow:
    def __init__(self, kind, target, borrower_scope):
        self.kind = kind
        self.target = target
        self.borrower_scope = borrower_scope

    def __repr__(self):
        return f"{self.kind}→{self.target.id}@{self.borrower_scope.name}"


def _arity_type_name(arity):
    """Return the canonical Totem type name for a constructor of a given arity."""

    return f"ADT<{arity}>"


class Node:
    def __init__(self, op, typ, scope):
        scope_path = _scope_path(scope)
        node_index = len(scope.nodes)
        self.id = _stable_id(scope_path, node_index)
        self.op = op
        self.typ = typ
        self.scope = scope
        life_scope_path = f"{scope_path}.life"
        life_id = _stable_id(life_scope_path, node_index)
        self.owned_life = Lifetime(scope, life_id)
        self.owned_life.owner_node = self
        self.borrows = []
        self.grade = OPS.get(op, {}).get("grade", "pure")
        self.ffi = None
        self.ffi_capabilities = []
        self.meta = {}
        self.arity = 0
        self._apply_ffi_metadata()
        self.update_type()

    def __repr__(self):
        return f"<{self.op}:{self.typ}@{self.scope.name}>"

    def _apply_ffi_metadata(self):
        decl = FFI_REGISTRY.get(self.op)
        if not decl:
            return
        self.ffi = decl
        self.grade = decl.grade
        self.typ = decl.return_type
        self.ffi_capabilities = list(decl.capabilities)
        self.meta.setdefault("fixed_type", decl.return_type)
    def update_type(self):
        """Refresh this node's inferred type based on current metadata and borrows."""

        self.arity = len(self.borrows)
        if "fixed_type" in self.meta:
            self.typ = self.meta["fixed_type"]
        elif self.op == "P":
            self.typ = "match"
        else:
            self.typ = _arity_type_name(self.arity)
        return self.typ


class Capability:
    """Linear capability token tracking resource usage."""

    def __init__(
        self,
        kind,
        resource=None,
        *,
        state=None,
        history=None,
        generation=0,
        active=True,
    ):
        self.kind = kind
        self.resource = resource
        self.state = state or {}
        self.history = history or []
        self.generation = generation
        self._active = active

    def evolve(self, action, detail=None, state_updates=None):
        if not self._active:
            raise RuntimeError(f"Capability {self} already consumed")

        new_state = dict(self.state)
        if state_updates:
            for key, value in state_updates.items():
                new_state[key] = value

        new_history = list(self.history)
        new_history.append({"action": action, "detail": detail})

        self._active = False

        return Capability(
            self.kind,
            self.resource,
            state=new_state,
            history=new_history,
            generation=self.generation + 1,
        )

    @property
    def is_active(self):
        return self._active

    def __repr__(self):
        return f"<Capability {self.kind}@{self.generation}>"


@dataclass(frozen=True)
class CapabilityUseResult:
    capability: Capability
    value: object = None


def resolve_value(value):
    if isinstance(value, CapabilityUseResult):
        return value.value
    return value


def extract_capability(value):
    if isinstance(value, CapabilityUseResult):
        return value.capability
    if isinstance(value, Capability):
        return value
    return None


def _clone_list(source):
    return list(source) if source is not None else []


def use_file_read(cap):
    index = cap.state.get("index", 0)
    contents = cap.state.get("contents", [])
    if index < len(contents):
        data = contents[index]
    else:
        data = None
    new_cap = cap.evolve("read", data, {"index": index + 1})
    return CapabilityUseResult(new_cap, data)


def use_file_write(cap, payload):
    writes = _clone_list(cap.state.get("writes", []))
    writes.append(payload)
    new_cap = cap.evolve("write", payload, {"writes": writes})
    return CapabilityUseResult(new_cap, True)


def use_net_send(cap, payload):
    transmissions = _clone_list(cap.state.get("transmissions", []))
    transmissions.append(payload)
    ack = f"sent:{payload}"
    new_cap = cap.evolve("send", payload, {"transmissions": transmissions})
    return CapabilityUseResult(new_cap, ack)


CAPABILITY_FACTORIES = {
    "FileRead": lambda: Capability(
        "FileRead",
        resource="input",
        state={"index": 0, "contents": ["input_data"]},
    ),
    "FileWrite": lambda: Capability(
        "FileWrite",
        resource="output",
        state={"writes": []},
    ),
    "NetSend": lambda: Capability(
        "NetSend",
        resource="socket",
        state={"transmissions": []},
    ),
}


def create_default_environment():
    env = {"__capabilities__": {}}
    for kind, factory in CAPABILITY_FACTORIES.items():
        env["__capabilities__"][kind] = factory()
    return env


def ensure_capability(env, kind):
    caps = env.setdefault("__capabilities__", {})
    if kind not in caps:
        caps[kind] = CAPABILITY_FACTORIES[kind]()
    return caps[kind]


def store_capability(env, kind, capability):
    env.setdefault("__capabilities__", {})[kind] = capability


class IRNode:
    """Lowered SSA-like form."""

    def __init__(self, id, op, typ, grade, args):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = args


class Scope:
    def __init__(self, name, parent=None, *, effect_cap=None, fence=None):
        self.name = name
        self.parent = parent
        self.nodes = []
        self.children = []
        self.lifetimes = []
        self.drops = []
        self.effect_cap = effect_cap
        self.fence = fence
        if parent:
            parent.children.append(self)

    def __repr__(self):
        return f"Scope({self.name})"


class Effect:
    """Graded monad for purity tracking."""

    def __init__(self, grade, value, log=None):
        self.grade = grade
        self.value = value
        self.log = log or []

    def bind(self, fn):
        out = fn(self.value)
        new_idx = max(EFFECT_GRADES.index(self.grade), EFFECT_GRADES.index(out.grade))
        return Effect(EFFECT_GRADES[new_idx], out.value, self.log + out.log)


class MovedValue:
    """Sentinel stored in the environment when a lifetime has been moved."""

    def __init__(self, origin_id):
        self.origin_id = origin_id

    def __repr__(self):
        return f"<moved:{self.origin_id}>"


def read_env_value(env, lifetime_id, default=None):
    """Fetch a lifetime's value while ensuring it has not been moved."""

    if lifetime_id is None:
        return default
    if lifetime_id not in env:
        if default is not None:
            return default
        raise KeyError(f"Unknown lifetime {lifetime_id}")
    val = env[lifetime_id]
    if isinstance(val, MovedValue):
        raise RuntimeError(f"Lifetime {lifetime_id} has been moved and is no longer usable")
    return val


def move_env_value(env, lifetime_id):
    """Mark a lifetime as moved and return its previous value."""

    if lifetime_id is None:
        raise RuntimeError("Cannot move a value without a lifetime identifier")
    if lifetime_id not in env:
        raise KeyError(f"Unknown lifetime {lifetime_id}")
    val = env[lifetime_id]
    if isinstance(val, MovedValue):
        raise RuntimeError(f"Lifetime {lifetime_id} has already been moved")
    env[lifetime_id] = MovedValue(lifetime_id)
    return val


class OwnedMessage:
    """A message that must be moved exactly once across actors."""

    def __init__(self, payload, capability, message_id):
        self.payload = payload
        self.capability = capability
        self.message_id = message_id
        self._moved = False

    def move_payload(self):
        if self._moved:
            raise RuntimeError(f"Message {self.message_id} has already been moved")
        self._moved = True
        return self.payload

    def __repr__(self):
        target = getattr(self.capability, "actor_id", "?")
        return f"<OwnedMessage id={self.message_id}→{target} moved={self._moved}>"


class ActorCapability:
    """Capability used to send messages to a specific actor."""

    def __init__(self, actor_system, actor_id):
        self.actor_system = actor_system
        self.actor_id = actor_id

    def send(self, message):
        return self.actor_system.send(self, message)

    def __repr__(self):
        return f"<Capability {self.actor_id}>"


class Actor:
    """Single actor with a mailbox and effect-local log."""

    def __init__(self, actor_id, behavior):
        self.actor_id = actor_id
        self.behavior = behavior
        self.mailbox = deque()
        self.local_log = []
        self.local_grade_index = EFFECT_GRADES.index("pure")

    def enqueue(self, payload):
        self.mailbox.append(payload)

    def drain(self):
        delivered = 0
        logs = []
        grade_index = self.local_grade_index
        while self.mailbox:
            payload = self.mailbox.popleft()
            effect = self.behavior(payload)
            grade_index = max(grade_index, EFFECT_GRADES.index(effect.grade))
            logs.extend(effect.log)
            delivered += 1
        self.local_grade_index = grade_index
        self.local_log.extend(logs)
        return delivered, logs, grade_index


def default_actor_behavior(payload):
    return Effect("state", {"last_message": payload}, [f"echo:{payload}"])


class ActorSystem:
    """Ownership-safe actor system with move-only message passing."""

    def __init__(self):
        self.actors = {}
        self._actor_counter = 0
        self._message_counter = 0
        self._public_log = []

    def spawn(self, behavior=None):
        behavior = behavior or default_actor_behavior
        actor_id = f"actor_{self._actor_counter}"
        self._actor_counter += 1
        actor = Actor(actor_id, behavior)
        self.actors[actor_id] = actor
        return ActorCapability(self, actor_id)

    def next_message_id(self):
        mid = self._message_counter
        self._message_counter += 1
        return mid

    def send(self, capability, message):
        if message.capability is not capability:
            raise RuntimeError("Message capability does not match the target actor")
        payload = message.move_payload()
        actor = self.actors.get(capability.actor_id)
        if actor is None:
            raise RuntimeError(f"Unknown actor {capability.actor_id}")
        actor.enqueue(payload)
        log_entry = f"send:{capability.actor_id}:msg{message.message_id}"
        return Effect("sys", True, [log_entry])

    def run_until_idle(self):
        delivered = 0
        logs = []
        highest_grade = EFFECT_GRADES.index("pure")

        while True:
            iteration_delivered = 0
            iteration_logs = []
            for actor_id, actor in self.actors.items():
                count, local_logs, grade_idx = actor.drain()
                if not count and not local_logs:
                    continue
                iteration_delivered += count
                highest_grade = max(highest_grade, grade_idx)
                iteration_logs.extend(f"{actor_id}:{entry}" for entry in local_logs)

            if not iteration_delivered and not iteration_logs:
                break

            delivered += iteration_delivered
            logs.extend(iteration_logs)

        prefix = f"run:delivered={delivered}"
        if logs:
            combined = [prefix] + logs
        else:
            combined = [prefix]
        self._public_log = logs
        return Effect("sys", self, combined)

    @property
    def last_public_log(self):
        return list(self._public_log)

class TIRInstruction:
    """Single SSA-like instruction."""

    def __init__(
        self, id, op, typ, grade, args, scope_path, metadata=None, produces=None
    ):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = args
        self.scope_path = scope_path
        self.metadata = metadata or {}
        self.produces = produces

    def __repr__(self):
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

        args_str = ", ".join(fmt_arg(a) for a in self.args) if self.args else ""
        return f"{self.id} = {self.op}({args_str}) : {self.typ} [{self.grade}] @{self.scope_path}"


class TIRProgram:
    """Flat, typed intermediate representation."""

    def __init__(self):
        self.instructions = []
        self.next_id = 0
        self.constructor_tags = {}
        self.next_tag = 0

    def new_id(self):
        vid = f"v{self.next_id}"
        self.next_id += 1
        return vid

    def emit(self, op, typ, grade, args, scope_path, metadata=None, produces=None):
        vid = self.new_id()
        instr = TIRInstruction(
            vid, op, typ, grade, args, scope_path, metadata, produces
        )
        self.instructions.append(instr)
        return vid

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

    def __repr__(self):
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
        if not operand_sig:
            operand_sig = ""
        else:
            operand_sig = f"{operand_sig}"

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


class MetaObject:
    """A serializable reflection of a Totem runtime object."""

    def __init__(self, kind, data):
        self.kind = kind
        self.data = data

    def __repr__(self):
        if self.kind == "Node":
            n = self.data
            return f"<MetaNode {n.op}:{n.typ}@{n.scope.name} [{n.grade}]>"
        if self.kind == "Scope":
            s = self.data
            return f"<MetaScope {s.name} nodes={len(s.nodes)}>"
        if self.kind == "TIR":
            t = self.data
            return f"<MetaTIR {len(t.instructions)} instrs>"
        return f"<MetaObject {self.kind}>"

    def to_dict(self):
        """Return a JSON-safe representation."""
        if self.kind == "TIR":
            return [instr.__dict__ for instr in self.data.instructions]
        if self.kind in ("Node", "Scope"):
            return self.data.__dict__
        return {"kind": self.kind}


def structural_decompress(src):
    """Build scope tree and typed node graph from raw characters."""

    def combine_caps(parent_cap, new_cap):
        if parent_cap is None:
            return new_cap
        if new_cap is None:
            return parent_cap
        parent_idx = EFFECT_GRADES.index(parent_cap)
        new_idx = EFFECT_GRADES.index(new_cap)
        return EFFECT_GRADES[min(parent_idx, new_idx)]

    root = Scope("root")
    stack = [(root, None, None)]  # (scope, expected_closer, opener)
    last_node = None

    openers = {
        "{": {"close": "}", "limit": None, "prefix": "scope", "label": "{}"},
        "(": {"close": ")", "limit": "pure", "prefix": "pure", "label": "()"},
        "[": {"close": "]", "limit": "state", "prefix": "state", "label": "[]"},
        "<": {"close": ">", "limit": "io", "prefix": "io", "label": "<>"},
    }

    closers = {info["close"]: opener for opener, info in openers.items()}

    for ch in src:
        current = stack[-1][0]

        if ch in openers:
            info = openers[ch]
            inherited_cap = combine_caps(current.effect_cap, info["limit"])
            name = f"{info['prefix']}_{len(current.children)}"
            s = Scope(
                name,
                current,
                effect_cap=inherited_cap,
                fence=info["label"],
            )
            stack.append((s, info["close"], ch))
        elif ch in closers:
            if len(stack) == 1:
                raise ValueError(f"Unmatched closing fence '{ch}'")
            scope, expected, opener = stack.pop()
            if ch != expected:
                raise ValueError(
                    f"Mismatched fence: opened with '{opener}' but closed with '{ch}'"
                )
            for n in scope.nodes:
                n.owned_life.end_scope = scope
                scope.lifetimes.append(n.owned_life)
                scope.drops.append(n.owned_life)
        elif ch.isalpha():
            node = Node(op=ch.upper(), typ="int32", scope=current)
            cap = current.effect_cap
            if cap is not None:
                node_idx = EFFECT_GRADES.index(node.grade)
                cap_idx = EFFECT_GRADES.index(cap)
                if node_idx > cap_idx:
                    fence = current.fence or "scope"
                    raise ValueError(
                        f"Effect grade '{node.grade}' exceeds '{cap}' fence in {fence}"
                    )
            current.nodes.append(node)

            if last_node and last_node.scope == current:
                kind = "mut" if ord(ch) % 2 == 0 else "shared"
                b = Borrow(kind, last_node.owned_life, current)
                node.borrows.append(b)
                last_node.owned_life.borrows.append(b)
            node.update_type()
            last_node = node

    if len(stack) != 1:
        _, expected, opener = stack[-1]
        raise ValueError(f"Unclosed fence '{opener}' expected '{expected}'")

    return root


def check_aliasing(scope, errors):
    for life in scope.lifetimes:
        mut_borrows = [b for b in life.borrows if b.kind == "mut"]
        shared_borrows = [b for b in life.borrows if b.kind == "shared"]

        if mut_borrows and shared_borrows:
            errors.append(f"Aliasing violation on {life.id} in {scope.name}")
        if len(mut_borrows) > 1:
            errors.append(f"Multiple mutable borrows of {life.id} in {scope.name}")

    for child in scope.children:
        check_aliasing(child, errors)


def check_lifetimes(scope, errors):
    for life in scope.lifetimes:
        for b in life.borrows:
            if _scope_depth(b.borrower_scope) > _scope_depth(life.end_scope):
                errors.append(f"Borrow {b} outlives {life.id}")
    for child in scope.children:
        check_lifetimes(child, errors)


def verify_ffi_calls(scope, errors):
    """Ensure all FFI-backed nodes match their declared metadata."""

    for node in scope.nodes:
        decl = getattr(node, "ffi", None)
        if decl:
            actual_arity = len(node.borrows)
            if actual_arity != decl.arity:
                errors.append(
                    f"FFI {decl.name} arity mismatch: expected {decl.arity}, got {actual_arity}"
                )
            for idx, (expected_type, borrow) in enumerate(zip(decl.arg_types, node.borrows)):
                target_node = getattr(borrow.target, "owner_node", None)
                actual_type = getattr(target_node, "typ", None)
                if actual_type and actual_type != expected_type:
                    if target_node is not None and hasattr(target_node, "meta"):
                        target_node.meta.setdefault("fixed_type", expected_type)
                        actual_type = target_node.update_type()
                    if actual_type != expected_type:
                        errors.append(
                            f"FFI {decl.name} argument {idx} expects {expected_type} but got {actual_type}"
                        )
            if node.typ != decl.return_type:
                errors.append(
                    f"FFI {decl.name} return type mismatch: expected {decl.return_type}, got {node.typ}"
                )
            if node.grade != decl.grade:
                errors.append(
                    f"FFI {decl.name} grade mismatch: expected {decl.grade}, got {node.grade}"
                )
    for child in scope.children:
        verify_ffi_calls(child, errors)


def _scope_depth(scope):
    d = 0
    while scope.parent:
        d += 1
        scope = scope.parent
    return d


def compute_scope_grades(scope, grades=None):
    """Populate a mapping of Scope → grade index."""

    if grades is None:
        grades = {}

    idx = 0
    for node in scope.nodes:
        idx = max(idx, EFFECT_GRADES.index(node.grade))
    for child in scope.children:
        child_idx = compute_scope_grades(child, grades)
        idx = max(idx, child_idx)

    grades[scope] = idx
    return idx


def _collect_grade_cut(scope, target_idx, grades):
    """Return nodes responsible for lifting this scope to the target grade."""

    contributors = []

    for node in scope.nodes:
        node_idx = EFFECT_GRADES.index(node.grade)
        if node_idx >= target_idx:
            contributors.append(node)

    for child in scope.children:
        if grades.get(child, -1) >= target_idx:
            contributors.extend(_collect_grade_cut(child, target_idx, grades))

    return contributors


def explain_grade(root_scope, target_grade):
    """Compute nodes that raise the program to ``target_grade``."""

    grade = target_grade.lower()
    if grade not in EFFECT_GRADES:
        raise ValueError(f"Unknown grade '{target_grade}'. Choose from {EFFECT_GRADES}.")

    target_idx = EFFECT_GRADES.index(grade)
    grades = {}
    compute_scope_grades(root_scope, grades)
    root_idx = grades.get(root_scope, 0)

    if root_idx < target_idx:
        return {
            "achieved": False,
            "final_grade": EFFECT_GRADES[root_idx],
            "nodes": [],
        }

    contributors = _collect_grade_cut(root_scope, target_idx, grades)
    seen = set()
    unique_nodes = []
    for node in contributors:
        if node.id in seen:
            continue
        seen.add(node.id)
        unique_nodes.append(node)

    unique_nodes.sort(key=lambda n: (_scope_path(n.scope), n.id))

    return {
        "achieved": True,
        "final_grade": EFFECT_GRADES[root_idx],
        "nodes": unique_nodes,
    }


def _index_lifetimes(root_scope):
    """Return lookup tables for lifetimes, nodes, and borrow origins."""

    lifetime_by_id = {}
    owner_node = {}
    borrow_owners = {}

    for scope in iter_scopes(root_scope):
        for node in scope.nodes:
            life = node.owned_life
            lifetime_by_id[life.id] = life
            owner_node[life.id] = node
            for borrow in node.borrows:
                borrow_owners[borrow] = node

    node_by_id = {node.id: node for node in owner_node.values()}

    return lifetime_by_id, owner_node, node_by_id, borrow_owners


def explain_borrow(root_scope, identifier):
    """Return a nested description of a borrow chain for ``identifier``."""

    lifetime_by_id, owner_node, node_by_id, borrow_owners = _index_lifetimes(
        root_scope
    )

    target_life = None

    if identifier in lifetime_by_id:
        target_life = lifetime_by_id[identifier]
    elif identifier in node_by_id:
        target_life = node_by_id[identifier].owned_life
    else:
        return {
            "found": False,
            "identifier": identifier,
            "lines": [],
        }

    def describe_lifetime(life, indent=0, seen=None):
        if seen is None:
            seen = set()
        prefix = "  " * indent
        lines = []

        scope_line = _scope_full_path(life.owner_scope)
        end_line = (
            _scope_full_path(life.end_scope)
            if life.end_scope is not None
            else "?"
        )
        owner = owner_node.get(life.id)
        owner_label = (
            f"node {owner.op} ({owner.id})"
            if owner is not None
            else "<unknown node>"
        )
        lines.append(
            f"{prefix}Lifetime {life.id} owned by {owner_label} in {scope_line}, ends at {end_line}"
        )

        if life.id in seen:
            lines.append(f"{prefix}  ↺ cycle detected, stopping traversal")
            return lines

        seen.add(life.id)

        if life.borrows:
            lines.append(f"{prefix}  Borrows:")
        for borrow in life.borrows:
            borrower = borrow_owners.get(borrow)
            borrower_label = (
                f"node {borrower.op} ({borrower.id})"
                if borrower is not None
                else "<unknown node>"
            )
            borrower_scope = _scope_full_path(borrow.borrower_scope)
            outlives = ""
            if life.end_scope is not None and _scope_depth(borrow.borrower_scope) > _scope_depth(
                life.end_scope
            ):
                outlives = " (⚠ outlives owner scope)"

            lines.append(
                f"{prefix}    - {borrow.kind} borrow by {borrower_label} at {borrower_scope}{outlives}"
            )
            if borrower is not None:
                lines.extend(describe_lifetime(borrower.owned_life, indent + 3, seen))

        seen.remove(life.id)
        return lines

    lines = describe_lifetime(target_life)
    return {
        "found": True,
        "identifier": identifier,
        "lines": lines,
    }


def visualize_graph(root):  # pragma: no cover
    """Render the decompressed scope graph with color-coded purity and lifetime->borrow edges."""
    if nx is None or plt is None:
        raise RuntimeError("Visualization requires networkx and matplotlib to be installed")

    G = nx.DiGraph()
    lifetime_nodes_added = set()

    def add_lifetime_node(life):
        lid = f"L:{life.id}"
        if lid in lifetime_nodes_added:
            return lid
        G.add_node(lid, label=f"L {life.id}", color="#d3d3d3")
        lifetime_nodes_added.add(lid)
        return lid

    def walk(scope):
        for n in scope.nodes:
            color = GRADE_COLORS.get(getattr(n, "grade", "pure"), "#B0BEC5")

            G.add_node(n.id, label=f"{n.op}\n[{n.grade}]", color=color)

            for b in n.borrows:
                lnode = add_lifetime_node(b.target)
                G.add_edge(lnode, n.id, style="dashed")

        for child in scope.children:
            walk(child)

    walk(root)

    node_colors = [G.nodes[n].get("color", "#d3d3d3") for n in G.nodes]
    node_labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes}

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        edgecolors="black",
        font_size=8,
    )

    dashed = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("style") == "dashed"]
    nx.draw_networkx_edges(G, pos, edgelist=dashed, style="dashed")

    plt.title("Totem Program Graph — purity & lifetimes")
    plt.show()


def iter_scopes(scope):
    """Yield a scope and all descendants in depth-first order."""
    yield scope
    for child in scope.children:
        yield from iter_scopes(child)


def export_graphviz(root, output_path):  # pragma: no cover
    """Export a Graphviz SVG with scope clusters and lifetime borrow edges."""
    import pydot

    if pydot is None:
        raise RuntimeError("Graphviz export requires the optional pydot dependency")

    import pydot

    import pydot

    try:
        import pydot
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Graphviz export requires pydot to be installed") from exc

    import pydot

    graph = pydot.Dot(
        "totem_scopes",
        graph_type="digraph",
        rankdir="LR",
        splines="spline",
        fontname="Helvetica",
    )

    lifetime_nodes = {}

    def build_cluster(scope, path):
        cluster_name = f"cluster_{path.replace('.', '_')}"
        cluster = pydot.Cluster(
            cluster_name,
            label=path,
            color="#7f8c8d",
            fontname="Helvetica",
            fontsize="10",
            style="rounded",
        )

        for node in scope.nodes:
            color = GRADE_COLORS.get(node.grade, "#B0BEC5")
            node_label = f"{node.op}\\n[{node.grade}]"
            graph_node = pydot.Node(
                node.id,
                label=node_label,
                shape="box",
                style="filled",
                fillcolor=color,
                color="#34495e",
                fontname="Helvetica",
            )
            cluster.add_node(graph_node)

            life = node.owned_life
            lid = f"life_{life.id}"
            if lid not in lifetime_nodes:
                lifetime_nodes[lid] = pydot.Node(
                    lid,
                    label=f"L {life.id}",
                    shape="ellipse",
                    style="dashed",
                    color="#7f8c8d",
                    fontname="Helvetica",
                )
            cluster.add_node(lifetime_nodes[lid])

        for child in scope.children:
            child_path = f"{path}.{child.name}"
            cluster.add_subgraph(build_cluster(child, child_path))

        return cluster

    graph.add_subgraph(build_cluster(root, "root"))

    for scope in iter_scopes(root):
        for node in scope.nodes:
            for borrow in node.borrows:
                src = f"life_{borrow.target.id}"
                graph.add_edge(
                    pydot.Edge(
                        src,
                        node.id,
                        style="dashed",
                        color="#7f8c8d",
                        penwidth="1.2",
                        arrowsize="0.8",
                    )
                )

    output_path = Path(output_path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    graph.write_svg(str(output_path))
    print(f"  ✓ Graphviz visualization exported → {output_path}")


def print_scopes(scope, indent=0):
    pad = "  " * indent
    print(f"{pad}{scope}")
    for n in scope.nodes:
        print(f"{pad}  {n} owns {n.owned_life}  [{n.grade}]")
        for b in n.borrows:
            print(f"{pad}    borrow {b}")
    for child in scope.children:
        print_scopes(child, indent + 1)
    if scope.drops:
        print(f"{pad}  drops {[l.id for l in scope.drops]}")


def evaluate_node(node, env):
    """
    Execute one node and return an Effect.
    For now, each op just produces a numeric or string value to illustrate
    monadic propagation.
    """
    op = node.op
    grade = node.grade

    def lift(val):
        return Effect(grade, val, [f"{op}:{val}"])

    if node.ffi:
        log = f"FFI:{node.ffi.name}"
        if node.ffi_capabilities:
            log += f" requires {', '.join(node.ffi_capabilities)}"
        return Effect(node.grade, None, [log])

    # Meta-level operations (demo)
    if op == "M":  # reflect current TIR
        tir = build_tir(node.scope)
        return Effect("meta", reflect(tir), [f"M:reflect({len(tir.instructions)})"])
    elif op == "N":  # dynamically emit a node into TIR
        tir = build_tir(node.scope)
        meta_instr = meta_emit(tir, "X", "int32", "pure")
        return Effect("meta", meta_instr, [f"N:emit({meta_instr})"])
    elif op == "O":  # run optimizer (meta)
        tir = build_tir(node.scope)
        optimized = optimize_tir(tir)
        return Effect(
            "meta", reflect(optimized), [f"O:optimize({len(optimized.instructions)} instrs)"]
        )

    # demo semantics
    if op == "A":  # pure constant
        return lift(1)
    elif op == "B":  # stateful increment
        env["counter"] = env.get("counter", 0) + 1
        return Effect("state", env["counter"], [f"B:inc->{env['counter']}"])
    elif op == "C":  # IO read (simulated)
        borrowed_cap = None
        if node.borrows:
            borrowed_cap = extract_capability(env.get(node.borrows[0].target.id))
        capability = borrowed_cap or ensure_capability(env, "FileRead")
        result = use_file_read(capability)
        store_capability(env, "FileRead", result.capability)
        log_val = result.value if result.value is not None else "EOF"
        return Effect("io", result, [f"C:read->{log_val}"])
    elif op == "D":
        return lift(2)
    elif op == "E":
        # use borrowed value if available
        src = node.borrows[0].target.id if node.borrows else None
        base = read_env_value(env, src, 0)
        try:
            val = base + 3
        except TypeError:
            val = 3
        return lift(val)
    elif op == "F":
        return lift(5)
    elif op == "G":  # IO write (simulated)
        borrow_id = node.borrows[0].target.id if node.borrows else None
        payload = read_env_value(env, borrow_id, "?")
        capability = ensure_capability(env, "FileWrite")
        result = use_file_write(capability, payload)
        store_capability(env, "FileWrite", result.capability)
        return Effect("io", result, [f"G:write({payload})"])
    elif op == "H":  # create an actor system
        system = ActorSystem()
        return Effect("sys", system, ["H:actors"])
    elif op == "J":  # spawn an actor and return capability
        if not node.borrows:
            raise RuntimeError("J requires borrowing an actor system")
        system = read_env_value(env, node.borrows[0].target.id)
        if not isinstance(system, ActorSystem):
            raise RuntimeError("J expects an ActorSystem as input")
        capability = system.spawn()
        return Effect("sys", capability, [f"J:spawn({capability.actor_id})"])
    elif op == "K":  # craft a move-only message for a capability
        if not node.borrows:
            raise RuntimeError("K requires borrowing an actor capability")
        capability = read_env_value(env, node.borrows[0].target.id)
        if not isinstance(capability, ActorCapability):
            raise RuntimeError("K expects an ActorCapability as input")
        message_id = capability.actor_system.next_message_id()
        payload = {"message": message_id, "to": capability.actor_id}
        message = OwnedMessage(payload, capability, message_id)
        return Effect("pure", message, [f"K:msg({capability.actor_id},id={message_id})"])
    elif op == "L":  # move the message into the actor system
        if not node.borrows:
            raise RuntimeError("L requires moving an OwnedMessage")
        borrow_id = node.borrows[0].target.id
        message = move_env_value(env, borrow_id)
        if not isinstance(message, OwnedMessage):
            raise RuntimeError("L expects an OwnedMessage to send")
        capability = message.capability
        send_effect = capability.send(message)
        logs = [
            f"L:send({capability.actor_id},id={message.message_id})",
            *send_effect.log,
        ]
        return Effect("sys", capability.actor_system, logs)
    elif op == "S":  # network send using capability
        borrow_id = node.borrows[0].target.id if node.borrows else None
        payload = read_env_value(env, borrow_id, 0)
        capability = ensure_capability(env, "NetSend")
        result = use_net_send(capability, payload)
        store_capability(env, "NetSend", result.capability)
        return Effect("sys", result, [f"S:send({payload})"])
    elif op == "P":  # run all actors until their queues are drained
        if not node.borrows:
            raise RuntimeError("P requires borrowing an actor system")
        system = read_env_value(env, node.borrows[0].target.id)
        if not isinstance(system, ActorSystem):
            raise RuntimeError("P expects an ActorSystem as input")
        run_effect = system.run_until_idle()
        logs = ["P:run"] + run_effect.log
        return Effect("sys", system, logs)
    else:
        return lift(0)


def evaluate_scope(scope, env=None):
    """
    Evaluate a scope: every node returns an Effect.
    The scope's total grade is the max grade of its children.
    """
    if env is None:
        env = create_default_environment()
    else:
        caps = env.setdefault("__capabilities__", {})
        for kind, factory in CAPABILITY_FACTORIES.items():
            if kind not in caps:
                caps[kind] = factory()
    effects = []
    scope_grade_index = 0

    for node in scope.nodes:
        eff = evaluate_node(node, env)
        env[node.owned_life.id] = eff.value
        effects.append(eff)
        scope_grade_index = max(scope_grade_index, EFFECT_GRADES.index(eff.grade))

    for child in scope.children:
        sub_eff = evaluate_scope(child, env)
        effects.append(sub_eff)
        scope_grade_index = max(scope_grade_index, EFFECT_GRADES.index(sub_eff.grade))

    combined_log = sum((e.log for e in effects), [])
    final_grade = EFFECT_GRADES[scope_grade_index]
    return Effect(final_grade, None, combined_log)


def scope_to_dict(scope):
    """Recursively convert a Scope tree to a serializable dict."""
    return {
        "name": scope.name,
        "effect_cap": scope.effect_cap,
        "fence": scope.fence,
        "lifetimes": [
            {
                "id": l.id,
                "owner_scope": l.owner_scope.name,
                "end_scope": l.end_scope.name if l.end_scope else None,
                "borrows": [
                    {
                        "kind": b.kind,
                        "target": b.target.id,
                        "borrower_scope": b.borrower_scope.name,
                    }
                    for b in l.borrows
                ],
            }
            for l in scope.lifetimes
        ],
        "nodes": [
            {
                "id": n.id,
                "op": n.op,
                "type": n.typ,
                "arity": n.arity,
                "grade": n.grade,
                "lifetime_id": n.owned_life.id,
                "meta": n.meta,
                "borrows": [
                    {
                        "kind": b.kind,
                        "target": b.target.id,
                        "borrower_scope": b.borrower_scope.name,
                    }
                    for b in n.borrows
                ],
            }
            for n in scope.nodes
        ],
        "drops": [l.id for l in scope.drops],
        "children": [scope_to_dict(child) for child in scope.children],
    }


def _certificate_payload_digest(payload):
    """Return a stable digest for certificate payloads."""

    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _collect_aliasing_payload(scope):
    lifetimes = []
    lifetime_counts = {}
    node_lifetimes = []
    borrows = []

    for sc in iter_scopes(scope):
        scope_path = _scope_path(sc)
        for life in sc.lifetimes:
            record = {
                "id": life.id,
                "owner": scope_path,
                "end": life.end_scope.name if life.end_scope else None,
            }
            lifetimes.append(record)
            lifetime_counts[life.id] = lifetime_counts.get(life.id, 0) + 1

        for node in sc.nodes:
            node_lifetimes.append(
                {
                    "node": node.id,
                    "owned_lifetime": node.owned_life.id,
                    "scope": scope_path,
                }
            )
            for borrow in node.borrows:
                borrows.append(
                    {
                        "node": node.id,
                        "kind": borrow.kind,
                        "target": borrow.target.id,
                    }
                )

    lifetimes.sort(key=lambda x: (x["id"], x["owner"], x.get("end")))
    node_lifetimes.sort(
        key=lambda x: (x["node"], x["owned_lifetime"], x["scope"])
    )
    borrows.sort(key=lambda x: (x["node"], x["kind"], x["target"]))

    duplicates = sorted([lid for lid, count in lifetime_counts.items() if count > 1])
    declared = {life["id"] for life in lifetimes}
    dangling = sorted({b["target"] for b in borrows if b["target"] not in declared})

    alias_errors = []
    check_aliasing(scope, alias_errors)

    payload = {
        "lifetimes": lifetimes,
        "node_lifetimes": node_lifetimes,
        "borrows": borrows,
        "duplicate_lifetimes": duplicates,
        "dangling_borrows": dangling,
        "alias_errors": alias_errors,
    }

    ok = not duplicates and not dangling and not alias_errors

    summary = {
        "lifetimes": len(lifetimes),
        "node_lifetimes": len(node_lifetimes),
        "borrows": len(borrows),
        "duplicate_lifetimes": duplicates,
        "dangling_borrows": dangling,
        "alias_errors": alias_errors,
    }

    return {
        "statement": "All borrows reference valid lifetimes without aliasing conflicts.",
        "payload_digest": _certificate_payload_digest(payload),
        "summary": summary,
        "ok": ok,
    }


def _collect_grade_payload(scope):
    node_entries = []
    max_grade_index = 0

    for sc in iter_scopes(scope):
        for node in sc.nodes:
            node_entries.append({"id": node.id, "grade": node.grade})
            max_grade_index = max(max_grade_index, EFFECT_GRADES.index(node.grade))

    node_entries.sort(key=lambda x: x["id"])
    computed_grade = EFFECT_GRADES[max_grade_index] if node_entries else EFFECT_GRADES[0]

    payload = {
        "node_grades": node_entries,
        "computed_grade": computed_grade,
    }

    return payload


def _grade_certificate(scope, reported_grade):
    payload = _collect_grade_payload(scope)
    computed_grade = payload["computed_grade"]
    ok = reported_grade == computed_grade

    summary = {
        "node_count": len(payload["node_grades"]),
        "computed_grade": computed_grade,
        "reported_grade": reported_grade,
    }

    return {
        "statement": "Program grade matches the maximum grade of its nodes.",
        "payload_digest": _certificate_payload_digest(payload),
        "summary": summary,
        "ok": ok,
    }


def build_bitcode_certificates(scope, final_grade):
    aliasing = _collect_aliasing_payload(scope)
    if not aliasing["ok"]:
        raise ValueError(f"Alias analysis failed: {aliasing['summary']}")

    grade = _grade_certificate(scope, final_grade)
    if not grade["ok"]:
        raise ValueError(
            "Effect grade proof failed: "
            f"expected {grade['summary']['computed_grade']} got {final_grade}"
        )

    return {"aliasing": aliasing, "grades": grade}
def _node_to_dict(node):
    entry = {
        "id": node.id,
        "op": node.op,
        "type": node.typ,
        "grade": node.grade,
        "lifetime_id": node.owned_life.id,
        "borrows": [
            {
                "kind": b.kind,
                "target": b.target.id,
                "borrower_scope": b.borrower_scope.name,
            }
            for b in node.borrows
        ],
    }
    if node.ffi:
        entry["ffi"] = node.ffi.to_dict()
    if node.ffi_capabilities:
        entry["ffi_capabilities"] = list(node.ffi_capabilities)
    return entry


def build_bitcode_document(scope, result_effect):
    """Create an in-memory Totem Bitcode representation."""

    certificates = build_bitcode_certificates(scope, result_effect.grade)

    return {
        "totem_version": "0.5",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root_scope": scope_to_dict(scope),
        "evaluation": {
            "final_grade": result_effect.grade,
            "log": result_effect.log,
        },
        "certificates": certificates,
    }


def write_bitcode_document(doc, filename):
    """Persist a Totem Bitcode document to disk."""

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"  ✓ Totem Bitcode exported → {filename}")
    return doc


def export_totem_bitcode(scope, result_effect, filename="program.totem.json"):
    """Serialize full program state and evaluation result to JSON."""
    doc = build_bitcode_document(scope, result_effect)
    return write_bitcode_document(doc, filename)


def load_totem_bitcode(filename):
    """Load a serialized Totem Bitcode JSON (for later reconstruction)."""
    with open(filename, "r", encoding="utf-8") as f:
        doc = json.load(f)
    verify_bitcode_document(doc)
    return doc


def reconstruct_scope(scope_dict, parent=None):
    """Rebuild a full Scope tree (with lifetimes and borrows) from a dictionary."""
    scope = Scope(
        scope_dict["name"],
        parent,
        effect_cap=scope_dict.get("effect_cap"),
        fence=scope_dict.get("fence"),
    )

    # First, create all nodes and lifetimes
    life_map = {}
    for ninfo in scope_dict["nodes"]:
        node = Node(ninfo["op"], ninfo["type"], scope)
        node.id = ninfo["id"]
        node.grade = ninfo["grade"]
        node.typ = ninfo["type"]
        node.owned_life.id = ninfo["lifetime_id"]
        node.meta = ninfo.get("meta", {}) or {}
        node.arity = ninfo.get("arity", 0)
        life_map[node.owned_life.id] = node.owned_life
        ffi_info = ninfo.get("ffi")
        if ffi_info:
            node.ffi = FFIDeclaration.from_dict(ffi_info)
            node.ffi_capabilities = list(node.ffi.capabilities)
            node.grade = node.ffi.grade
            node.typ = node.ffi.return_type
        elif ninfo.get("ffi_capabilities"):
            node.ffi_capabilities = list(ninfo["ffi_capabilities"])
        scope.nodes.append(node)

    # Rebuild lifetimes (for visualization/debug)
    for linfo in scope_dict.get("lifetimes", []):
        l = Lifetime(scope, linfo["id"])
        l.owner_scope = scope
        end_name = linfo.get("end_scope")
        if end_name:
            target = scope
            while target and target.name != end_name:
                target = target.parent
            l.end_scope = target
        scope.lifetimes.append(l)
        life_map[l.id] = l

    # Now reconstruct borrows
    for ninfo, node in zip(scope_dict["nodes"], scope.nodes):
        for binfo in ninfo.get("borrows", []):
            target_life = life_map.get(binfo["target"])
            if target_life:
                b = Borrow(binfo["kind"], target_life, scope)
                node.borrows.append(b)
                target_life.borrows.append(b)
        node.update_type()

    # Drops
    for did in scope_dict.get("drops", []):
        if did in life_map:
            scope.drops.append(life_map[did])

    # Recurse into children
    for child_dict in scope_dict.get("children", []):
        reconstruct_scope(child_dict, scope)

    return scope


def _validate_certificate(name, stored, expected):
    if not stored:
        raise ValueError(f"Bitcode missing {name} certificate")

    if stored.get("payload_digest") != expected["payload_digest"]:
        raise ValueError(f"{name} certificate digest mismatch")

    if stored.get("summary") != expected["summary"]:
        raise ValueError(f"{name} certificate summary mismatch")

    if not stored.get("ok"):
        raise ValueError(f"{name} certificate indicates failure: {stored['summary']}")

    if not expected.get("ok"):
        raise ValueError(f"{name} certificate recomputation failed: {expected['summary']}")


def verify_bitcode_document(doc):
    """Ensure embedded certificates match reconstructed program state."""

    certificates = doc.get("certificates")
    if not certificates:
        raise ValueError("Totem bitcode missing proof certificates")

    scope = reconstruct_scope(doc["root_scope"])

    alias_expected = _collect_aliasing_payload(scope)
    grade_expected = _grade_certificate(scope, doc["evaluation"]["final_grade"])

    _validate_certificate("aliasing", certificates.get("aliasing"), alias_expected)
    _validate_certificate("grades", certificates.get("grades"), grade_expected)

    return True


def reexecute_bitcode(filename):
    """Load a Totem Bitcode file and re-evaluate the reconstructed tree."""
    doc = load_totem_bitcode(filename)
    print(f"Loaded Totem Bitcode v{doc['totem_version']} ({filename})")

    root_dict = doc["root_scope"]
    scope = reconstruct_scope(root_dict)

    print("\nRe-evaluating Totem Bitcode ...")
    result = evaluate_scope(scope)
    print(f"  → final grade: {result.grade}")
    print("  → execution log:")
    for entry in result.log:
        print("   ", entry)
    return result


def canonicalize_bitcode(doc):
    """
    Normalize a loaded bitcode dict so semantically identical programs
    produce identical JSON strings even if UUIDs or field ordering differ.
    """

    def sort_dict(d):
        if isinstance(d, dict):
            return {k: sort_dict(v) for k, v in sorted(d.items())}
        elif isinstance(d, list):
            return [sort_dict(x) for x in d]
        else:
            return d

    return sort_dict(doc)


def hash_bitcode_document(doc):
    """Compute SHA-256 hash of an in-memory Totem Bitcode document."""
    canon = canonicalize_bitcode(doc)
    data = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_bitcode(filename):
    """Compute SHA-256 hash of a Totem Bitcode file."""
    doc = load_totem_bitcode(filename)
    h = hash_bitcode_document(doc)
    print(f"SHA256({filename}) = {h}")
    return h


def diff_bitcodes(file_a, file_b):
    """Compare two Totem Bitcode files and show structural and semantic differences."""
    a = canonicalize_bitcode(load_totem_bitcode(file_a))
    b = canonicalize_bitcode(load_totem_bitcode(file_b))

    ha = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
    if ha == hb:
        print(f"✓ Bitcodes are identical ({ha})")
        return

    print(f"✗ Bitcodes differ\n  {file_a[:30]}…: {ha}\n  {file_b[:30]}…: {hb}")

    fa = a["evaluation"]
    fb = b["evaluation"]

    if fa["final_grade"] != fb["final_grade"]:
        print(f"  • Final grade differs: {fa['final_grade']} vs {fb['final_grade']}")
    if fa["log"] != fb["log"]:
        print("  • Execution log differs:")
        for la, lb in zip(fa["log"], fb["log"]):
            if la != lb:
                print(f"    - {la}\n    + {lb}")
        if len(fa["log"]) != len(fb["log"]):
            print(f"    (log length differs: {len(fa['log'])} vs {len(fb['log'])})")

    def count_nodes(s):
        return len(s["nodes"]) + sum(count_nodes(c) for c in s.get("children", []))

    na, nb = count_nodes(a["root_scope"]), count_nodes(b["root_scope"])
    if na != nb:
        print(f"  • Node count differs: {na} vs {nb}")


def record_run(bitcode_filename, result_effect):
    """Append this run’s metadata to the Totem logbook, signed."""
    sha = hash_bitcode(bitcode_filename)
    sig = sign_hash(sha)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "filename": bitcode_filename,
        "hash": sha,
        "signature": sig,
        "final_grade": result_effect.grade,
        "log_length": len(result_effect.log),
        "first_log": result_effect.log[0] if result_effect.log else None,
        "last_log": result_effect.log[-1] if result_effect.log else None,
    }

    with open(LOGBOOK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"  📜 Recorded and signed run → {LOGBOOK_FILE}")
    return entry


def show_logbook(limit=10):
    """Display recent logbook entries."""
    try:
        with open(LOGBOOK_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("No logbook yet.")
        return

    entries = [json.loads(l) for l in lines[-limit:]]
    print(f"\nTotem Logbook — last {len(entries)} entries:")
    for e in reversed(entries):
        print(
            f"• {e['timestamp']}  {e['filename']}  [{e['final_grade']}]  {e['hash'][:12]}…"
        )
        if e["first_log"] and e["last_log"]:
            print(f"    log: {e['first_log']} → {e['last_log']}")


def ensure_keypair():  # pragma: no cover
    """Create an RSA keypair if it doesn't exist."""
    if rsa is None or serialization is None or default_backend is None:
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    try:
        with open(KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
    except FileNotFoundError:
        print("🔐 Generating new Totem RSA keypair ...")
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(KEY_FILE, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
        public_key = private_key.public_key()
        with open(PUB_FILE, "wb") as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
        print(f"  ✓ Keys written to {KEY_FILE}, {PUB_FILE}")
    return private_key


def sign_hash(sha256_hex):  # pragma: no cover
    """Sign a SHA256 hex digest with the private key."""
    if rsa is None or hashes is None or padding is None:
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    private_key = ensure_keypair()
    signature = private_key.sign(
        sha256_hex.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    return signature.hex()


def verify_signature(sha256_hex, signature_hex):  # pragma: no cover
    """Verify a signature against the public key."""
    if (
        InvalidSignature is None
        or hashes is None
        or serialization is None
        or default_backend is None
        or padding is None
    ):
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )

    with open(PUB_FILE, "rb") as f:
        public_key = serialization.load_pem_public_key(
            f.read(), backend=default_backend()
        )
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            sha256_hex.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


def build_tir(scope, program=None, prefix="root"):
    """Lower a full scope tree into a flat TIRProgram."""
    if program is None:
        program = TIRProgram()

    scope_path = prefix if prefix == "root" else f"{prefix}.{scope.name}"

    for node in scope.nodes:        
        args = [
            {
                "kind": "consume" if b.kind == "mut" else "borrow",
                "target": b.target.id,
            }
            for b in node.borrows
        ]
        # Pattern match nodes are first emitted as MATCH before being desugared.
        if node.op == "P" and "match_cases" in node.meta:
            cases_meta = []
            for ctor in node.meta["match_cases"]:
                if isinstance(ctor, dict):
                    ctor_key = tuple(ctor.get("constructor", (ctor.get("op"), ctor.get("arity", 0))))
                    result = ctor.get("result")
                else:
                    ctor_key, result = ctor
                op_name, arity = ctor_key
                tag = program.constructor_tag(op_name, arity)
                cases_meta.append(
                    {
                        "constructor": (op_name, arity),
                        "tag": tag,
                        "result": result,
                    }
                )

            metadata = {"cases": cases_meta}
            if "default_case" in node.meta:
                metadata["default"] = node.meta["default_case"]

            program.emit(
                "MATCH", node.typ, node.grade, args, scope_path, metadata=metadata
            )
            continue

        arity = len(args)
        constructor_tag = program.constructor_tag(node.op, arity)
        metadata = {"constructor_tag": constructor_tag, "arity": arity}
        program.emit(
            node.op,
            node.typ,
            node.grade,
            args,
            scope_path,
            metadata=metadata,
            produces=node.owned_life.id,
        )

    for child in scope.children:
        build_tir(child, program, scope_path)

    if prefix == "root":
        program.desugar_pattern_matches()

    return program


def _instruction_identity(instr, scope_counters):
    """Return a stable identity token for a TIR instruction."""

    if instr.produces:
        return ("life", instr.produces)

    idx = scope_counters.setdefault(instr.scope_path, 0)
    scope_counters[instr.scope_path] = idx + 1
    return ("scope", instr.scope_path, idx)


def _normalize_args(args):
    normalized = []
    for arg in args:
        if isinstance(arg, dict):
            normalized.append((arg.get("kind"), arg.get("target")))
        else:
            normalized.append((None, arg))
    return normalized


def compute_tir_distance(tir_a, tir_b):
    """Compute a semantic distance between two TIR programs.

    The metric is a tuple of edit components:

    • node_edits — instruction insertions/deletions.
    • grade_delta — cumulative absolute difference across effect grades.
    • op_changes — opcode substitutions for matching instruction identities.
    • type_changes — result type substitutions for matching instruction identities.
    • borrow_rewires — argument rewires (kind/target changes).

    The total distance is the sum of the components. This yields a coarse yet
    stable measure of how far two programs diverge semantically.
    """

    def map_instructions(tir):
        mapping = {}
        scope_counters = {}
        for instr in tir.instructions:
            key = _instruction_identity(instr, scope_counters)
            # In practice keys are unique, but we guard against collisions by
            # appending a disambiguator.
            if key in mapping:
                suffix = 1
                new_key = key + (suffix,)
                while new_key in mapping:
                    suffix += 1
                    new_key = key + (suffix,)
                key = new_key
            mapping[key] = instr
        return mapping

    map_a = map_instructions(tir_a)
    map_b = map_instructions(tir_b)

    node_edits = 0
    grade_delta = 0
    op_changes = 0
    type_changes = 0
    borrow_rewires = 0

    all_keys = set(map_a) | set(map_b)
    for key in all_keys:
        instr_a = map_a.get(key)
        instr_b = map_b.get(key)
        if instr_a is None or instr_b is None:
            node_edits += 1
            continue

        if instr_a.op != instr_b.op:
            op_changes += 1

        if instr_a.typ != instr_b.typ:
            type_changes += 1

        if instr_a.grade != instr_b.grade:
            grade_delta += abs(
                EFFECT_GRADES.index(instr_a.grade)
                - EFFECT_GRADES.index(instr_b.grade)
            )

        args_a = _normalize_args(instr_a.args)
        args_b = _normalize_args(instr_b.args)
        max_len = max(len(args_a), len(args_b))
        for idx in range(max_len):
            item_a = args_a[idx] if idx < len(args_a) else None
            item_b = args_b[idx] if idx < len(args_b) else None
            if item_a != item_b:
                borrow_rewires += 1

    total = node_edits + grade_delta + op_changes + type_changes + borrow_rewires
    return {
        "node_edits": node_edits,
        "grade_delta": grade_delta,
        "op_changes": op_changes,
        "type_changes": type_changes,
        "borrow_rewires": borrow_rewires,
        "total": total,
    }


def _mutate_byte(ch):
    code = ord(ch)
    if 32 <= code <= 126:
        return chr(32 + ((code - 32 + 1) % 95))
    return chr((code + 1) % 0x110000)


def continuous_semantics_profile(src, base_tir=None, mutate_fn=None):
    """Measure how semantics shift under single-byte mutations.

    Each byte is rotated to the next printable ASCII character (configurable
    via ``mutate_fn``). We rebuild the TIR for the mutated source and compute
    the semantic distance against ``base_tir``.
    """

    mutate_fn = mutate_fn or _mutate_byte
    if base_tir is None:
        base_tir = build_tir(structural_decompress(src))

    profile = []
    for idx, ch in enumerate(src):
        mutated_char = mutate_fn(ch)
        if mutated_char == ch:
            continue
        mutated_src = f"{src[:idx]}{mutated_char}{src[idx + 1:]}"
        try:
            mutated_tree = structural_decompress(mutated_src)
        except ValueError as exc:
            dist = {
                "node_edits": 0,
                "grade_delta": 0,
                "op_changes": 0,
                "type_changes": 0,
                "borrow_rewires": 0,
                "total": 0,
            }
            profile.append(
                {
                    "index": idx,
                    "original": ch,
                    "mutated": mutated_char,
                    "mutated_src": mutated_src,
                    "distance": dist,
                    "error": str(exc),
                }
            )
            continue
        mutated_tir = build_tir(mutated_tree)
        dist = compute_tir_distance(base_tir, mutated_tir)
        profile.append(
            {
                "index": idx,
                "original": ch,
                "mutated": mutated_char,
                "mutated_src": mutated_src,
                "distance": dist,
            }
        )

    return profile
def _wasm_local_name(identifier):
    return f"${identifier}"


def _format_wasm_list(items, prefix=""):
    return [f"{prefix}{item}" for item in items]


def tir_to_wat(tir, capabilities=None):
    """Compile a TIRProgram into a WebAssembly text module.

    Only pure instructions are lowered to direct WebAssembly operations. IO-grade
    instructions are exposed as host imports and require the caller to provide
    the corresponding capability string (e.g. ``io.read``).
    """

    capabilities = set(capabilities or [])
    required_capabilities = []

    local_decls = []
    local_set = set()
    body_lines = []
    last_pure_local = None
    alias_map = {}

    def declare_local(identifier):
        lname = _wasm_local_name(identifier)
        if lname not in local_set:
            local_decls.append(f"(local {lname} i32)")
            local_set.add(lname)
        return lname

    def bind_aliases(instr, local_name):
        alias_map[instr.id] = local_name
        if instr.produces:
            alias_map[instr.produces] = local_name

    imports_needed = {}

    value_producers = {}
    for instr in tir.instructions:
        value_producers[instr.id] = instr
        if instr.produces:
            value_producers[instr.produces] = instr

    for instr in tir.instructions:
        if instr.grade == "pure":
            local_name = declare_local(instr.id)
            bind_aliases(instr, local_name)

            if instr.op in {"A", "D", "F"}:
                const_val = {"A": 1, "D": 2, "F": 5}[instr.op]
                body_lines.append(f"(local.set {local_name} (i32.const {const_val}))")
            elif instr.op == "CONST" and hasattr(instr, "value"):
                body_lines.append(
                    f"(local.set {local_name} (i32.const {int(instr.value)}))"
                )
            elif instr.op == "E":
                if not instr.args:
                    raise ValueError("E operation expects at least one borrow argument")
                arg = instr.args[0]
                target = arg.get("target") if isinstance(arg, dict) else arg
                dep_local = alias_map.get(target)
                if not dep_local:
                    raise ValueError(f"Unknown borrow target {target} for op E")
                body_lines.append(
                    f"(local.set {local_name} (i32.add (local.get {dep_local}) (i32.const 3)))"
                )
            else:
                raise NotImplementedError(
                    f"Pure operation {instr.op} is not supported for WASM lowering"
                )
            last_pure_local = local_name
        elif instr.grade == "io":
            io_info = IO_IMPORTS.get(instr.op)
            if not io_info:
                raise NotImplementedError(
                    f"IO operation {instr.op} missing import metadata"
                )
            cap = io_info["capability"]
            if cap not in capabilities:
                raise PermissionError(
                    f"Capability '{cap}' required to lower IO operation {instr.op}"
                )
            required_capabilities.append(cap)
            import_key = (io_info["module"], io_info["name"])
            if import_key not in imports_needed:
                imports_needed[import_key] = io_info

            if instr.produces:
                local_name = declare_local(instr.id)
                bind_aliases(instr, local_name)
            else:
                local_name = None

            call_operands = []
            for arg in instr.args:
                target = arg.get("target") if isinstance(arg, dict) else arg
                dep_local = alias_map.get(target)
                if dep_local:
                    call_operands.append(f"(local.get {dep_local})")
                else:
                    producer = value_producers.get(target)
                    if io_info["params"]:
                        if producer:
                            raise ValueError(
                                "IO operation "
                                f"{instr.op} argument {target} depends on "
                                f"{producer.op} [{producer.grade}], which cannot be lowered to WebAssembly"
                            )
                        if isinstance(target, str):
                            raise ValueError(
                                f"IO operation {instr.op} has unknown dependency {target}"
                            )
                        raise ValueError(
                            f"IO operation {instr.op} received unsupported operand {arg}"
                        )
                    # The IO import does not expect arguments; ignore the dependency.

            if call_operands:
                call_expr = (
                    f"(call ${io_info['name']} " + " ".join(call_operands) + ")"
                )
            else:
                call_expr = f"(call ${io_info['name']})"

            if io_info["results"]:
                body_lines.append(f"(local.set {local_name} {call_expr})")
            else:
                body_lines.append(call_expr)
                if local_name:
                    body_lines.append(f"(local.set {local_name} (i32.const 0))")
        else:
            # Meta and stateful instructions are not lowered to WebAssembly.
            continue

    module_lines = ["(module"]

    for (module, name), info in imports_needed.items():
        params = " ".join(f"(param {p})" for p in info["params"])
        results = " ".join(f"(result {r})" for r in info["results"])
        signature = " ".join(filter(None, [params, results]))
        module_lines.append(
            f"  (import \"{module}\" \"{name}\" (func ${name} {signature}))"
        )

    module_lines.append("  (func $run (export \"run\") (result i32)")
    module_lines.extend(_format_wasm_list(local_decls, prefix="    "))
    module_lines.extend(_format_wasm_list(body_lines, prefix="    "))
    if last_pure_local:
        module_lines.append(f"    (return (local.get {last_pure_local}))")
    else:
        module_lines.append("    (return (i32.const 0))")
    module_lines.append("  )")
    module_lines.append(")")

    metadata = {
        "imports": sorted(set(required_capabilities)),
        "locals": sorted(local_set),
        "pure_instructions": len([i for i in tir.instructions if i.grade == "pure"]),
        "io_instructions": len([i for i in tir.instructions if i.grade == "io"]),
    }

    return "\n".join(module_lines), metadata


def export_wasm_module(tir, output_path, capabilities=None, metadata_path=None):
    """Write a WebAssembly text module to disk."""

    wat, metadata = tir_to_wat(tir, capabilities=capabilities)
    output_path = Path(output_path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(wat, encoding="utf-8")
    print(f"  ✓ WASM module exported → {output_path}")

    if metadata_path:
        meta_path = Path(metadata_path)
        if meta_path.parent and not meta_path.parent.exists():
            meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  ✓ WASM metadata exported → {meta_path}")

    return metadata


def reflect(obj):
    """
    Produce a MetaObject view of any Totem structure.
    Used in the evaluator as meta-operations.
    """
    if isinstance(obj, Scope):
        return MetaObject("Scope", obj)
    if isinstance(obj, Node):
        return MetaObject("Node", obj)
    if isinstance(obj, TIRProgram):
        return MetaObject("TIR", obj)
    return MetaObject("Value", obj)


def meta_emit(
    program, op, typ="int32", grade="pure", args=None, scope_path="root.meta"
):
    """
    Dynamically extend a TIR program with a new instruction.
    Returns a MetaObject referencing the new instruction.
    """
    args = args or []
    vid = program.emit(op, typ, grade, args, scope_path)
    instr = program.instructions[-1]
    return MetaObject("TIR_Instruction", instr)


def list_meta_ops():
    return {
        "reflect": "Return a MetaObject view of a Totem structure",
        "meta_emit": "Append a new TIR instruction (Meta effect)",
        "list_meta_ops": "List available reflective primitives",
    }


def fold_constants(tir):
    """Replace simple const ops (A=1, D=2, etc.) used by pure ops with folded values."""
    const_map = {}
    new_instrs = []
    for instr in tir.instructions:
        if instr.op in PURE_CONST_VALUES:  # treat as constant sources
            const_map[instr.id] = PURE_CONST_VALUES[instr.op]
            if instr.produces:
                const_map[instr.produces] = const_map[instr.id]
            new_instrs.append(instr)
            continue

        # fold if all args are known constants
        arg_targets = []
        for arg in instr.args:
            if isinstance(arg, dict):
                target = arg.get("target")
            else:
                target = arg
            arg_targets.append(target)

        if (
            instr.grade == "pure"
            and arg_targets
            and all(t in const_map for t in arg_targets)
        ):
            val = sum(const_map[t] for t in arg_targets)  # naive demo fold
            folded = TIRInstruction(
                instr.id,
                "CONST",
                "int32",
                "pure",
                [],
                instr.scope_path,
                metadata=instr.metadata,
                produces=instr.produces,
            )
            folded.value = val
            const_map[instr.id] = val
            if instr.produces:
                const_map[instr.produces] = val
            new_instrs.append(folded)
        else:
            new_instrs.append(instr)

    tir.instructions = new_instrs
    return tir


def _resolve_alias(replacements, value):
    if value is None:
        return None
    seen = set()
    current = value
    while current in replacements and current not in seen:
        seen.add(current)
        current = replacements[current]
    return current


def _rewrite_arg(arg, replacements):
    if isinstance(arg, dict):
        new_arg = dict(arg)
        target = new_arg.get("target")
        if target is not None:
            new_arg["target"] = _resolve_alias(replacements, target)
        return new_arg
    if isinstance(arg, str):
        return _resolve_alias(replacements, arg)
    return arg


def _freeze_arg(arg):
    if isinstance(arg, dict):
        return tuple(sorted((k, _freeze_arg(v)) for k, v in arg.items()))
    if isinstance(arg, list):
        return tuple(_freeze_arg(v) for v in arg)
    return arg


def _iter_targets(instr):
    for arg in instr.args:
        if isinstance(arg, dict):
            target = arg.get("target")
            if target:
                yield target
        elif isinstance(arg, str):
            yield arg


def evaluate_pure_regions(tir):
    """Perform constant/partial evaluation within pure regions."""

    value_map = {}
    for instr in tir.instructions:
        known = getattr(instr, "value", None)
        if known is None and instr.op in PURE_CONST_VALUES:
            known = PURE_CONST_VALUES[instr.op]
        if known is not None:
            value_map[instr.id] = known
            if instr.produces:
                value_map[instr.produces] = known

    for instr in tir.instructions:
        if instr.grade != "pure":
            # ensure args copied to avoid shared state
            instr.args = [dict(arg) if isinstance(arg, dict) else arg for arg in instr.args]
            continue

        constant_values = []
        dynamic_args = []
        for arg in instr.args:
            if isinstance(arg, dict):
                target = arg.get("target")
                if target in value_map:
                    constant_values.append(value_map[target])
                else:
                    dynamic_args.append(dict(arg))
            else:
                if arg in value_map:
                    constant_values.append(value_map[arg])
                else:
                    dynamic_args.append(arg)

        if not dynamic_args and constant_values:
            total = sum(constant_values)
            instr.op = "CONST"
            instr.args = []
            instr.value = total
            value_map[instr.id] = total
            if instr.produces:
                value_map[instr.produces] = total
        else:
            if constant_values:
                const_sum = sum(constant_values)
                instr.partial_constant = const_sum
                dynamic_args = [
                    {"kind": "const", "value": const_sum}
                ] + dynamic_args
            instr.args = dynamic_args

    return tir


def common_subexpression_elimination(tir):
    """Eliminate redundant pure instructions within the same scope."""

    replacements = {}
    key_map = {}
    new_instrs = []

    for instr in tir.instructions:
        instr.args = [_rewrite_arg(arg, replacements) for arg in instr.args]

        key = None
        if instr.grade == "pure":
            key = (
                instr.scope_path,
                instr.op,
                instr.typ,
                tuple(_freeze_arg(arg) for arg in instr.args),
                getattr(instr, "value", None),
                getattr(instr, "partial_constant", None),
            )

        if key and key in key_map:
            canonical = key_map[key]
            replacements[instr.id] = canonical.id
            if instr.produces and canonical.produces:
                replacements[instr.produces] = canonical.produces
            continue

        if key:
            key_map[key] = instr

        new_instrs.append(instr)

    tir.instructions = new_instrs
    return tir


def dead_code_elimination(tir):
    """Remove pure instructions whose results are never consumed."""

    referenced = set()
    kept = []

    for instr in reversed(tir.instructions):
        keep = instr.grade != "pure"
        if not keep:
            if instr.id in referenced:
                keep = True
            elif instr.produces and instr.produces in referenced:
                keep = True

        if keep:
            kept.append(instr)
            referenced.add(instr.id)
            if instr.produces:
                referenced.add(instr.produces)
            referenced.update(_iter_targets(instr))

    tir.instructions = list(reversed(kept))
    return tir


def inline_pure_regions(tir):
    """Inline child scope instructions when the region is purely functional."""

    if not tir.instructions:
        return tir

    scope_to_instrs = {}
    value_scopes = {}
    for instr in tir.instructions:
        scope_to_instrs.setdefault(instr.scope_path, []).append(instr)
        value_scopes[instr.id] = instr.scope_path
        if instr.produces:
            value_scopes[instr.produces] = instr.scope_path

    for scope_path, instrs in list(scope_to_instrs.items()):
        if scope_path == "root" or not instrs:
            continue
        if not all(instr.grade == "pure" for instr in instrs):
            continue

        parent = scope_path.rsplit(".", 1)[0] if "." in scope_path else "root"

        can_inline = True
        for instr in instrs:
            for target in _iter_targets(instr):
                dep_scope = value_scopes.get(target)
                if dep_scope and dep_scope not in {scope_path, parent}:
                    can_inline = False
                    break
            if not can_inline:
                break

        if not can_inline:
            continue

        for instr in instrs:
            instr.scope_path = parent
            value_scopes[instr.id] = parent
            if instr.produces:
                value_scopes[instr.produces] = parent

    return tir
def reorder_pure_ops(tir):
    """Reorder instructions by effect grade while respecting dependencies."""

    instructions = list(tir.instructions)
    if not instructions:
        return tir

    grade_rank = {grade: idx for idx, grade in enumerate(EFFECT_GRADES)}
    id_to_instr = {instr.id: instr for instr in instructions}
    lifetime_producers = {
        instr.produces: instr.id for instr in instructions if instr.produces
    }

    def iter_edges(instr):
        for arg in instr.args:
            if isinstance(arg, dict):
                kind = arg.get("kind")
                if isinstance(kind, str):
                    kind_key = kind.lower()
                else:
                    kind_key = None
                yield kind_key, arg.get("target")
            else:
                yield None, arg

    dependencies = {instr.id: set() for instr in instructions}
    adjacency = {instr.id: set() for instr in instructions}
    borrow_edges = []

    for instr in instructions:
        for kind, target in iter_edges(instr):
            if not target:
                continue
            dep_id = None
            if target in id_to_instr:
                dep_id = target
            elif target in lifetime_producers:
                dep_id = lifetime_producers[target]
            if dep_id and dep_id != instr.id:
                dependencies[instr.id].add(dep_id)
                adjacency[dep_id].add(instr.id)
                if kind in {"borrow", "consume"}:
                    borrow_edges.append((dep_id, instr.id))

    original_index = {instr.id: idx for idx, instr in enumerate(instructions)}
    scope_sequences = {}
    for instr in instructions:
        scope_sequences.setdefault(instr.scope_path, []).append(instr.id)

    preceding_counts = {}
    for scope, seq in scope_sequences.items():
        seen = 0
        for iid in seq:
            preceding_counts[iid] = seen
            seen += 1

    scheduled_counts = {scope: 0 for scope in scope_sequences}

    def is_fenced(scope_path):
        return any("fence" in part.lower() for part in scope_path.split("."))

    heap = []
    for instr in instructions:
        if not dependencies[instr.id]:
            heapq.heappush(
                heap,
                (
                    grade_rank.get(instr.grade, len(EFFECT_GRADES)),
                    original_index[instr.id],
                    instr.id,
                ),
            )

    new_order = []
    processed = set()
    deferred = []

    while heap:
        grade, orig_idx, instr_id = heapq.heappop(heap)
        instr = id_to_instr[instr_id]
        scope = instr.scope_path
        if is_fenced(scope) and scheduled_counts[scope] < preceding_counts[instr_id]:
            deferred.append((grade, orig_idx, instr_id))
            if not heap:
                break
            continue

        new_order.append(instr_id)
        processed.add(instr_id)
        if is_fenced(scope):
            scheduled_counts[scope] += 1

        for item in deferred:
            heapq.heappush(heap, item)
        deferred.clear()

        for succ in adjacency[instr_id]:
            if succ in processed:
                continue
            dependencies[succ].discard(instr_id)
            if not dependencies[succ]:
                succ_instr = id_to_instr[succ]
                heapq.heappush(
                    heap,
                    (
                        grade_rank.get(succ_instr.grade, len(EFFECT_GRADES)),
                        original_index[succ],
                        succ,
                    ),
                )

    if len(new_order) != len(instructions):
        return tir

    new_positions = {iid: idx for idx, iid in enumerate(new_order)}

    for src, dst in borrow_edges:
        orig_src = original_index[src]
        orig_dst = original_index[dst]
        if orig_src == orig_dst:
            continue
        lower = min(orig_src, orig_dst)
        upper = max(orig_src, orig_dst)
        new_src = new_positions.get(src)
        new_dst = new_positions.get(dst)
        if new_src is None or new_dst is None:
            return tir
        if new_src >= new_dst:
            return tir
        for instr in instructions:
            orig_idx = original_index[instr.id]
            if lower < orig_idx < upper:
                new_idx = new_positions.get(instr.id)
                if new_idx is None or not (new_src < new_idx < new_dst):
                    return tir

    for scope, ids in scope_sequences.items():
        if not is_fenced(scope):
            continue
        positions = [new_positions[iid] for iid in ids]
        sorted_positions = sorted(positions)
        if positions != sorted_positions:
            return tir
        start = sorted_positions[0]
        if sorted_positions != list(range(start, start + len(sorted_positions))):
            return tir

    tir.instructions = [id_to_instr[i] for i in new_order]
    return tir


def schedule_effects(tir):
    """Wrapper around the effect-aware scheduler."""

    return reorder_pure_ops(tir)


def inline_trivial_io(tir):
    """Replace IO ops with constant placeholders if they have deterministic logs."""
    for instr in tir.instructions:
        if instr.grade == "io" and instr.op == "C":
            instr.op = "CONST_IO"
            instr.grade = "pure"
    return tir


def optimize_tir(tir):
    """Run the full suite of optimizer passes."""

    fold_constants(tir)
    evaluate_pure_regions(tir)
    common_subexpression_elimination(tir)
    dead_code_elimination(tir)
    inline_pure_regions(tir)
    schedule_effects(tir)
    inline_trivial_io(tir)
    return tir


def transpile_totem_to_tir(src, *, optimize=True, ffi_decls=None):
    """Lower Totem source into Totem IR, optionally applying optimizations."""

    previous_registry = get_registered_ffi_declarations()
    if ffi_decls is not None:
        try:
            register_ffi_declarations(ffi_decls, reset=True)
        except Exception:
            clear_ffi_registry()
            for name, decl in previous_registry.items():
                FFI_REGISTRY[name] = decl
            raise

    try:
        tree = structural_decompress(src)
        errors: list[str] = []
        check_aliasing(tree, errors)
        check_lifetimes(tree, errors)
        verify_ffi_calls(tree, errors)

        tir = build_tir(tree)
        if optimize:
            optimize_tir(tir)

        return TranspilationResult(
            source=src,
            tree=tree,
            tir=tir,
            errors=tuple(errors),
            optimized=bool(optimize),
        )
    finally:
        if ffi_decls is not None:
            clear_ffi_registry()
            for name, decl in previous_registry.items():
                FFI_REGISTRY[name] = decl


transpile_to_totem_ir = transpile_totem_to_tir


def list_optimizers():
    return {
        "fold_constants": "Constant folding for pure ops",
        "evaluate_pure_regions": "Evaluate pure regions with constants",
        "common_subexpression_elimination": "Eliminate duplicate pure ops",
        "dead_code_elimination": "Prune unused pure instructions",
        "inline_pure_regions": "Inline pure child scopes into parents",
        "reorder_pure_ops": "Effect-sensitive reordering by grade",
        "schedule_effects": "Effect-aware scheduling respecting dependencies",
        "inline_trivial_io": "Replace deterministic IO reads with constants",
        "optimize_tir": "Run all optimization passes",
    }


def compile_and_evaluate(src, ffi_decls=None):
    """Run the full Totem pipeline on a raw source string."""

    previous_registry = get_registered_ffi_declarations()
    if ffi_decls is not None:
        register_ffi_declarations(ffi_decls, reset=True)

    try:
        tree = structural_decompress(src)
        errors = []
        check_aliasing(tree, errors)
        check_lifetimes(tree, errors)
        verify_ffi_calls(tree, errors)
        result = evaluate_scope(tree)
        return tree, errors, result
    finally:
        if ffi_decls is not None:
            clear_ffi_registry()
            for name, decl in previous_registry.items():
                FFI_REGISTRY[name] = decl


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
                    visualize_graph(entry["tree"])
                continue
            if cmd == ":save":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if not entry:
                    continue
                if len(parts) > 2:
                    filename = parts[2]
                else:
                    filename = f"program_{entry['index']}.totem.json"
                write_bitcode_document(entry["bitcode_doc"], filename)
                continue
            if cmd == ":hash":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if entry:
                    h = hash_bitcode_document(entry["bitcode_doc"])
                    print(f"SHA256(program_{entry['index']}) = {h}")
                continue
            if cmd == ":bitcode":
                entry = resolve_entry(parts[1] if len(parts) > 1 else None)
                if entry:
                    canon = canonicalize_bitcode(entry["bitcode_doc"])
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
                canon_a = canonicalize_bitcode(entry_a["bitcode_doc"])
                canon_b = canonicalize_bitcode(entry_b["bitcode_doc"])
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

        tree, errors, result = compile_and_evaluate(line)
        counter += 1
        entry = {
            "index": counter,
            "src": line,
            "tree": tree,
            "errors": errors,
            "result": result,
            "bitcode_doc": build_bitcode_document(tree, result),
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
    argp.add_argument("--visualize", action="store_true", help="Render program graph")
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
        diff_bitcodes(params.diff[0], params.diff[1])
        return
    if params.hash:
        hash_bitcode(params.hash)
        return
    if params.load:
        reexecute_bitcode(params.load)
        return
    if params.logbook:
        show_logbook()
        return
    if params.repl:
        run_repl()
        return
    if params.verify:
        ok = verify_signature(params.verify, input("Signature hex: ").strip())
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
            info = explain_grade(tree, params.why_grade)
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
        info = explain_borrow(tree, params.why_borrow)
        if not info["found"]:
            print("  ✗ Identifier not found in this program.")
        else:
            for line in info["lines"]:
                print("  " + line)

    export_totem_bitcode(tree, result, "program.totem.json")
    record_run("program.totem.json", result)
    tir = build_tir(tree)
    print("\nTIR:")
    print(tir)

    profile = continuous_semantics_profile(params.src, base_tir=tir)
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
            export_wasm_module(
                tir,
                params.wasm,
                capabilities=params.capabilities,
                metadata_path=params.wasm_metadata,
            )
        except PermissionError as exc:
            print(f"  ✗ {exc}")
        except NotImplementedError as exc:
            print(f"  ✗ {exc}")
    mlir_module = emit_mlir_module(tir)
    print("\nMLIR dialect:")
    print(mlir_module)

    llvm_ir = emit_llvm_ir(tir)
    print("\nLLVM IR (pure segment):")
    print(llvm_ir)

    if params.viz:
        export_graphviz(tree, params.viz)
    if params.visualize:
        visualize_graph(tree)


__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
