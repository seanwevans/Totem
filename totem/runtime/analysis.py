"""Analysis and evaluation utilities for the Totem runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable
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
    from matplotlib import animation as mpl_animation
except ModuleNotFoundError:  # pragma: no cover
    mpl_animation = None

try:
    import pydot
except ModuleNotFoundError:  # pragma: no cover
    pydot = None

from ..constants import EFFECT_GRADES, GRADE_COLORS
from ..ffi import (
    FFI_REGISTRY,
    clear_ffi_registry,
    get_registered_ffi_declarations,
    register_ffi_declarations,
)
from .actors import ActorCapability, ActorSystem, OwnedMessage
from .capabilities import (
    CAPABILITY_FACTORIES,
    create_default_environment,
    ensure_capability,
    extract_capability,
    store_capability,
    use_file_read,
    use_file_write,
    use_net_send,
)
from .core import (
    Borrow,
    Effect,
    Lifetime,
    Node,
    Scope,
    _scope_full_path,
    _scope_path,
    move_env_value,
    read_env_value,
)
from .meta import MetaObject, _unwrap_meta_tir
from .tir import TIRProgram, assemble_bytecode, run_bytecode

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
            if (
                last_node
                and last_node.scope is current
                and getattr(last_node, "grade", "pure") == "meta"
            ):
                last_node.attached_scopes.append(s)
                s.meta_role = "attached"
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
            parent_scope = scope.parent
            if parent_scope and parent_scope.nodes:
                last_node = parent_scope.nodes[-1]
            else:
                last_node = None
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
        if getattr(child, "meta_role", None) == "attached":
            continue
        check_aliasing(child, errors)


def check_lifetimes(scope, errors):
    for life in scope.lifetimes:
        for b in life.borrows:
            if _scope_depth(b.borrower_scope) > _scope_depth(life.end_scope):
                errors.append(f"Borrow {b} outlives {life.id}")
    for child in scope.children:
        if getattr(child, "meta_role", None) == "attached":
            continue
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
        if getattr(child, "meta_role", None) == "attached":
            continue
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
        if getattr(child, "meta_role", None) == "attached":
            continue
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


def visualize_graph(root, script="purity"):  # pragma: no cover
    """Render the decompressed scope graph based on a small visualization DSL.

    The *script* argument accepts one or more comma/semicolon/plus separated
    directives:

    ``purity``
        Render the classic purity grade map.
    ``fence``
        Render a fence/effect-cap map highlighting scope fences.
    ``lifetime`` or ``animate:lifetime``
        Produce a temporal animation that highlights node evaluations and the
        lifetime borrows they introduce.
    """

    if nx is None or plt is None:
        raise RuntimeError("Visualization requires networkx and matplotlib to be installed")

    if script is True:
        script = "purity"
    if script is None:
        script = "purity"
    if not isinstance(script, str):
        raise TypeError("Visualization script must be a string of directives")

    commands = [
        part.strip().lower()
        for part in re.split(r"[;,+]+", script)
        if part.strip()
    ]
    if not commands:
        commands = ["purity"]

    fence_palette = [
        "#FFE082",
        "#F8BBD0",
        "#C5E1A5",
        "#B39DDB",
        "#B0BEC5",
        "#FFAB91",
        "#80CBC4",
        "#90CAF9",
    ]

    graph = nx.DiGraph()
    lifetime_nodes = {}
    node_scope = {}
    node_grade = {}
    events = []

    def add_lifetime_node(life):
        lid = f"L:{life.id}"
        if lid not in lifetime_nodes:
            graph.add_node(
                lid,
                label=f"L {life.id}",
                color="#d3d3d3",
                kind="lifetime",
            )
            lifetime_nodes[lid] = life
        return lid

    def walk(scope):
        for node in scope.nodes:
            grade = getattr(node, "grade", "pure")
            color = GRADE_COLORS.get(grade, "#B0BEC5")
            graph.add_node(
                node.id,
                label=f"{node.op}\n[{grade}]",
                color=color,
                kind="op",
                grade=grade,
            )
            node_scope[node.id] = scope
            node_grade[node.id] = grade

            life_node = add_lifetime_node(node.owned_life)

            borrow_edges = []
            for borrow in node.borrows:
                life = add_lifetime_node(borrow.target)
                graph.add_edge(
                    life,
                    node.id,
                    style="dashed",
                    borrow_kind=borrow.kind,
                )
                borrow_edges.append((life, node.id))

            events.append(
                {
                    "node": node.id,
                    "life": life_node,
                    "edges": borrow_edges,
                    "description": f"{node.op} [{grade}] @ {_scope_full_path(scope)}",
                }
            )

        for child in scope.children:
            walk(child)

    walk(root)

    labels = {n: graph.nodes[n].get("label", str(n)) for n in graph.nodes}
    base_colors = {n: graph.nodes[n].get("color", "#d3d3d3") for n in graph.nodes}
    positions = nx.spring_layout(graph, seed=42)
    dashed_edges = [
        (u, v)
        for (u, v, data) in graph.edges(data=True)
        if data.get("style") == "dashed"
    ]

    def render_static(color_for_node, title):
        plt.figure()
        node_colors = [color_for_node(node) for node in graph.nodes]
        nx.draw(
            graph,
            positions,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            edgecolors="black",
            font_size=8,
        )
        if dashed_edges:
            nx.draw_networkx_edges(graph, positions, edgelist=dashed_edges, style="dashed")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    fence_cache = {}

    def fence_color(node):
        info = graph.nodes[node]
        if info.get("kind") == "lifetime":
            return "#d3d3d3"
        scope = node_scope.get(node)
        fence = getattr(scope, "fence", None)
        if fence not in fence_cache:
            if fence is None:
                fence_cache[fence] = "#ECEFF1"
            else:
                index = len(fence_cache) % len(fence_palette)
                fence_cache[fence] = fence_palette[index]
        return fence_cache[fence]

    def purity_color(node):
        info = graph.nodes[node]
        if info.get("kind") == "lifetime":
            return info.get("color", "#d3d3d3")
        grade = info.get("grade") or node_grade.get(node, "pure")
        return GRADE_COLORS.get(grade, "#B0BEC5")

    def animate_lifetimes():
        if mpl_animation is None:
            raise RuntimeError(
                "Lifetime animation requires matplotlib's animation module to be available"
            )
        if not events:
            render_static(purity_color, "Purity map (no events to animate)")
            return

        fig, ax = plt.subplots()

        def draw_frame(index):
            event = events[index]
            highlight_nodes = {event["node"], event["life"]}
            ax.clear()
            node_colors = []
            for node in graph.nodes:
                color = base_colors[node]
                if node in highlight_nodes:
                    color = "#FFF176"
                node_colors.append(color)

            nx.draw(
                graph,
                positions,
                ax=ax,
                with_labels=True,
                labels=labels,
                node_color=node_colors,
                edgecolors="black",
                font_size=8,
            )
            if dashed_edges:
                nx.draw_networkx_edges(
                    graph, positions, ax=ax, edgelist=dashed_edges, style="dashed"
                )
            if event["edges"]:
                nx.draw_networkx_edges(
                    graph,
                    positions,
                    ax=ax,
                    edgelist=event["edges"],
                    width=2.5,
                    edge_color="#E65100",
                )
            ax.set_title(
                "Temporal lifetime animation\n" + event["description"]
            )
            ax.margins(0.2)
            return []

        animation = mpl_animation.FuncAnimation(
            fig,
            draw_frame,
            frames=len(events),
            interval=1200,
            repeat=True,
        )
        # Keep a reference to the animation alive until ``plt.show`` returns. Without this,
        # Matplotlib may garbage-collect the animation before it renders, producing a warning
        # and preventing the animation from playing.
        fig._totem_animation = animation
        plt.tight_layout()
        plt.show()

    for command in commands:
        if command in {"purity", "map:purity", "purity-map"}:
            render_static(purity_color, "Totem Program Graph — purity map")
        elif command in {"fence", "map:fence", "fence-map"}:
            fence_cache.clear()
            render_static(fence_color, "Totem Program Graph — fence map")
        elif command in {"lifetime", "animate:lifetime", "lifetime-animation"}:
            animate_lifetimes()
        else:
            raise ValueError(f"Unknown visualization directive '{command}'")


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
    from .compiler import build_tir, optimize_tir

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
    if op == "M":  # reflect current scope into a mutable TIR session
        base_tir = build_tir(node.scope, include_attached=False)
        session = base_tir.clone()
        meta_obj = MetaObject("TIR", session)
        return Effect("meta", meta_obj, [f"M:reflect({len(base_tir.instructions)})"])
    elif op == "N":  # compile attached meta scope into the borrowed TIR
        if not node.borrows:
            raise RuntimeError("N requires borrowing a TIR MetaObject")
        borrowed_id = node.borrows[0].target.id
        meta_value = read_env_value(env, borrowed_id)
        program = _unwrap_meta_tir(meta_value)
        before = len(program.instructions)
        attached_scopes = node.attached_scopes
        for idx, script_scope in enumerate(attached_scopes):
            scope_path = f"meta_script_{script_scope.name}_{idx}"
            build_tir(
                script_scope,
                program,
                prefix=scope_path,
                include_attached=True,
            )
        added = len(program.instructions) - before
        log = [f"N:emit({added} instrs)"]
        return Effect("meta", meta_value, log)
    elif op == "O":  # run optimizer (meta)
        if not node.borrows:
            raise RuntimeError("O requires borrowing a TIR MetaObject")
        borrowed_id = node.borrows[0].target.id
        meta_value = read_env_value(env, borrowed_id)
        program = _unwrap_meta_tir(meta_value)
        optimized = optimize_tir(program)
        meta_obj = MetaObject("TIR", optimized)
        return Effect(
            "meta", meta_obj, [f"O:optimize({len(optimized.instructions)} instrs)"]
        )
    elif op == "Q":  # execute the borrowed TIR using the bytecode VM
        if not node.borrows:
            raise RuntimeError("Q requires borrowing a TIR MetaObject")
        borrowed_id = node.borrows[0].target.id
        meta_value = read_env_value(env, borrowed_id)
        program = _unwrap_meta_tir(meta_value)
        bytecode = assemble_bytecode(program)
        result = run_bytecode(bytecode)
        log = [f"Q:run({len(program.instructions)} instrs)", *result.log]
        return Effect("meta", result, log)

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
        if getattr(child, "meta_role", None) == "attached":
            continue
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
        "children": [
            scope_to_dict(child)
            for child in scope.children
            if getattr(child, "meta_role", None) != "attached"
        ],
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
        if getattr(sc, "meta_role", None) == "attached":
            continue
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
        if getattr(sc, "meta_role", None) == "attached":
            continue
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


def _ensure_lifetime_registration(scope):
    """Ensure every node-owned lifetime is registered with its scope."""

    for sc in iter_scopes(scope):
        seen_ids = {life.id for life in sc.lifetimes}
        drop_ids = {life.id for life in sc.drops}
        for node in sc.nodes:
            life = node.owned_life
            if life.end_scope is None:
                life.end_scope = sc
            if life.id not in seen_ids:
                sc.lifetimes.append(life)
                seen_ids.add(life.id)
            if life.id not in drop_ids:
                sc.drops.append(life)
                drop_ids.add(life.id)


@dataclass(frozen=True)
class IncrementalAnalysis:
    """Result payload returned by :class:`IncrementalVerifier`."""

    source: str
    tree: "Scope"
    computed_grade: str
    aliasing: dict
    grade_certificate: dict

    @property
    def ok(self):
        """Return ``True`` when both aliasing and grade proofs succeed."""

        return bool(self.aliasing.get("ok")) and bool(
            self.grade_certificate.get("ok")
        )

    @property
    def diagnostics(self):
        """Return a flat list of human-readable diagnostics."""

        messages = []
        summary = self.aliasing.get("summary", {})
        alias_errors = summary.get("alias_errors", [])
        messages.extend(alias_errors)

        if not self.grade_certificate.get("ok", True):
            details = self.grade_certificate.get("summary", {})
            expected = details.get("computed_grade")
            reported = details.get("reported_grade")
            messages.append(
                f"Grade mismatch: expected {expected} but analyser reported {reported}"
            )

        return messages


class IncrementalVerifier:
    """Incrementally analyse Totem source during interactive editing."""

    def __init__(self, source="", ffi_decls=None):
        self._source = source
        self._ffi_decls = ffi_decls
        self._dirty = True
        self._last_tree = None
        self._last_result = None

    @property
    def source(self):
        """Current buffered source under analysis."""

        return self._source

    def set_source(self, source):
        """Replace the entire source buffer and re-run analysis."""

        self._source = source
        self._dirty = True
        return self.lint()

    def set_ffi_declarations(self, ffi_decls):
        """Update the FFI declarations used for subsequent analyses."""

        self._ffi_decls = ffi_decls
        self._dirty = True

    def apply_edit(self, start, end, text):
        """Apply a text edit defined by ``[start:end]`` → ``text``."""

        length = len(self._source)
        if start < 0 or end < start or end > length:
            raise ValueError("Edit bounds are out of range")

        self._source = f"{self._source[:start]}{text}{self._source[end:]}"
        self._dirty = True
        return self.lint()

    def lint(self):
        """Return the most recent analysis, recomputing when necessary."""

        if not self._dirty and self._last_result is not None:
            return self._last_result

        previous_registry = get_registered_ffi_declarations()
        if self._ffi_decls is not None:
            try:
                register_ffi_declarations(self._ffi_decls, reset=True)
            except Exception:
                clear_ffi_registry()
                for name, decl in previous_registry.items():
                    FFI_REGISTRY[name] = decl
                raise

        try:
            runtime_mod = sys.modules.get('totem.runtime')
            decompress = getattr(runtime_mod, 'structural_decompress', structural_decompress)
            tree = decompress(self._source)
            _ensure_lifetime_registration(tree)
            aliasing = _collect_aliasing_payload(tree)
            grade_payload = _collect_grade_payload(tree)
            computed_grade = grade_payload["computed_grade"]
            grade_certificate = _grade_certificate(tree, computed_grade)
        finally:
            if self._ffi_decls is not None:
                clear_ffi_registry()
                for name, decl in previous_registry.items():
                    FFI_REGISTRY[name] = decl

        self._last_tree = tree
        self._last_result = IncrementalAnalysis(
            source=self._source,
            tree=tree,
            computed_grade=computed_grade,
            aliasing=aliasing,
            grade_certificate=grade_certificate,
        )
        self._dirty = False
        return self._last_result

    @property
    def last_tree(self):
        """Return the scope tree from the last completed analysis."""

        return self._last_tree
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




__all__ = [
    'structural_decompress',
    'check_aliasing',
    'check_lifetimes',
    'verify_ffi_calls',
    '_scope_depth',
    'compute_scope_grades',
    '_collect_grade_cut',
    'explain_grade',
    '_index_lifetimes',
    'explain_borrow',
    'visualize_graph',
    'iter_scopes',
    'export_graphviz',
    'print_scopes',
    'evaluate_node',
    'evaluate_scope',
    'scope_to_dict',
    '_certificate_payload_digest',
    '_collect_aliasing_payload',
    '_collect_grade_payload',
    '_grade_certificate',
    'build_bitcode_certificates',
    '_ensure_lifetime_registration',
    'IncrementalAnalysis',
    'IncrementalVerifier',
    '_node_to_dict'
]
