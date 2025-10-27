"""Compilation and optimization pipeline for Totem."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Iterable

from ..constants import EFFECT_GRADES, IO_IMPORTS, PURE_CONST_VALUES
from ..ffi import (
    FFI_REGISTRY,
    FFIDeclaration,
    clear_ffi_registry,
    get_registered_ffi_declarations,
    register_ffi_declarations,
)
from .analysis import (
    check_aliasing,
    check_lifetimes,
    evaluate_scope,
    structural_decompress,
    verify_ffi_calls,
)
from .core import Scope
from .tir import TIRInstruction, TIRProgram, TranspilationResult


def build_tir(scope, program=None, prefix="root", include_attached=False):
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
                    ctor_key = tuple(
                        ctor.get("constructor", (ctor.get("op"), ctor.get("arity", 0)))
                    )
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
        if getattr(child, "meta_role", None) == "attached" and not include_attached:
            continue
        build_tir(child, program, scope_path, include_attached=include_attached)

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
                EFFECT_GRADES.index(instr_a.grade) - EFFECT_GRADES.index(instr_b.grade)
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
                call_expr = f"(call ${io_info['name']} " + " ".join(call_operands) + ")"
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
            f'  (import "{module}" "{name}" (func ${name} {signature}))'
        )

    module_lines.append('  (func $run (export "run") (result i32)')
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
            instr.args = [
                dict(arg) if isinstance(arg, dict) else arg for arg in instr.args
            ]
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
                dynamic_args = [{"kind": "const", "value": const_sum}] + dynamic_args
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


__all__ = [
    "build_tir",
    "_instruction_identity",
    "_normalize_args",
    "compute_tir_distance",
    "_mutate_byte",
    "continuous_semantics_profile",
    "_wasm_local_name",
    "_format_wasm_list",
    "tir_to_wat",
    "export_wasm_module",
    "fold_constants",
    "_resolve_alias",
    "_rewrite_arg",
    "_freeze_arg",
    "_iter_targets",
    "evaluate_pure_regions",
    "common_subexpression_elimination",
    "dead_code_elimination",
    "inline_pure_regions",
    "reorder_pure_ops",
    "schedule_effects",
    "inline_trivial_io",
    "optimize_tir",
    "transpile_totem_to_tir",
    "transpile_to_totem_ir",
    "list_optimizers",
    "compile_and_evaluate",
]
