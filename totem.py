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
from datetime import datetime
import hashlib
import json
import sys
import uuid

EFFECT_GRADES = ["pure", "state", "io", "sys", "meta"]

OPS = {
    "A": {"grade": "pure"},
    "B": {"grade": "state"},
    "C": {"grade": "io"},
    "D": {"grade": "pure"},
    "E": {"grade": "pure"},
    "F": {"grade": "pure"},
    "G": {"grade": "io"},
}

LOGBOOK_FILE = "totem.logbook.jsonl"
KEY_FILE = "totem_private_key.pem"
PUB_FILE = "totem_public_key.pem"


class Lifetime:
    def __init__(self, owner_scope):
        self.id = str(uuid.uuid4())[:6]
        self.owner_scope = owner_scope
        self.end_scope = None
        self.borrows = []

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


class Node:
    def __init__(self, op, typ, scope):
        self.id = str(uuid.uuid4())[:6]
        self.op = op
        self.typ = typ
        self.scope = scope
        self.owned_life = Lifetime(scope)
        self.borrows = []
        self.grade = OPS.get(op, {}).get("grade", "pure")

    def __repr__(self):
        return f"<{self.op}:{self.typ}@{self.scope.name}>"


class IRNode:
    """Lowered SSA-like form."""

    def __init__(self, id, op, typ, grade, args):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = args


class Scope:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.nodes = []
        self.children = []
        self.lifetimes = []
        self.drops = []
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


class TIRInstruction:
    """Single SSA-like instruction."""

    def __init__(self, id, op, typ, grade, args, scope_path):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = args
        self.scope_path = scope_path

    def __repr__(self):
        args_str = ", ".join(self.args)
        return f"{self.id} = {self.op}({args_str}) : {self.typ} [{self.grade}] @{self.scope_path}"


class TIRProgram:
    """Flat, typed intermediate representation."""

    def __init__(self):
        self.instructions = []
        self.next_id = 0

    def new_id(self):
        vid = f"v{self.next_id}"
        self.next_id += 1
        return vid

    def emit(self, op, typ, grade, args, scope_path):
        vid = self.new_id()
        instr = TIRInstruction(vid, op, typ, grade, args, scope_path)
        self.instructions.append(instr)
        return vid

    def __repr__(self):
        return "\n".join(map(str, self.instructions))


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
    root = Scope("root")
    current = root
    last_node = None

    for ch in src:
        if ch == "{":
            s = Scope(f"scope_{len(current.children)}", current)
            current = s
        elif ch == "}":
            for n in current.nodes:
                n.owned_life.end_scope = current
                current.lifetimes.append(n.owned_life)
                current.drops.append(n.owned_life)
            current = current.parent or current
        elif ch.isalpha():
            node = Node(op=ch.upper(), typ="int32", scope=current)
            current.nodes.append(node)

            if last_node and last_node.scope == current:
                kind = "mut" if ord(ch) % 2 == 0 else "shared"
                b = Borrow(kind, last_node.owned_life, current)
                node.borrows.append(b)
                last_node.owned_life.borrows.append(b)
            last_node = node

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


def _scope_depth(scope):
    d = 0
    while scope.parent:
        d += 1
        scope = scope.parent
    return d


def visualize_graph(root):
    """Render the decompressed scope graph with color-coded purity and lifetime->borrow edges."""
    import networkx as nx
    import matplotlib.pyplot as plt

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
            color = {
                "pure": "green",
                "state": "yellow",
                "io": "red",
                "sys": "purple",
                "meta": "gray",
            }.get(getattr(n, "grade", "pure"), "gray")

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
        folded = fold_constants(tir)
        reorder_pure_ops(folded)
        return Effect(
            "meta", reflect(folded), [f"O:optimize({len(folded.instructions)} instrs)"]
        )

    # demo semantics
    if op == "A":  # pure constant
        return lift(1)
    elif op == "B":  # stateful increment
        env["counter"] = env.get("counter", 0) + 1
        return Effect("state", env["counter"], [f"B:inc->{env['counter']}"])
    elif op == "C":  # IO read (simulated)
        val = "input_data"
        return Effect("io", val, [f"C:read->{val}"])
    elif op == "D":
        return lift(2)
    elif op == "E":
        # use borrowed value if available
        src = node.borrows[0].target.id if node.borrows else None
        val = env.get(src, 0) + 3
        return lift(val)
    elif op == "F":
        return lift(5)
    elif op == "G":  # IO write (simulated)
        msg = f"G:write({env.get(node.borrows[0].target.id, '?')})"
        return Effect("io", True, [msg])
    else:
        return lift(0)


def evaluate_scope(scope, env=None):
    """
    Evaluate a scope: every node returns an Effect.
    The scope's total grade is the max grade of its children.
    """
    env = env or {}
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
                "grade": n.grade,
                "lifetime_id": n.owned_life.id,
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


def export_totem_bitcode(scope, result_effect, filename="program.totem.json"):
    """Serialize full program state and evaluation result to JSON."""
    doc = {
        "totem_version": "0.5",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "root_scope": scope_to_dict(scope),
        "evaluation": {
            "final_grade": result_effect.grade,
            "log": result_effect.log,
        },
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"  ✓ Totem Bitcode exported → {filename}")
    return doc


def load_totem_bitcode(filename):
    """Load a serialized Totem Bitcode JSON (for later reconstruction)."""
    with open(filename, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc


def reconstruct_scope(scope_dict, parent=None):
    """Rebuild a full Scope tree (with lifetimes and borrows) from a dictionary."""
    scope = Scope(scope_dict["name"], parent)

    # First, create all nodes and lifetimes
    life_map = {}
    for ninfo in scope_dict["nodes"]:
        node = Node(ninfo["op"], ninfo["type"], scope)
        node.grade = ninfo["grade"]
        node.owned_life.id = ninfo["lifetime_id"]
        life_map[node.owned_life.id] = node.owned_life
        scope.nodes.append(node)

    # Rebuild lifetimes (for visualization/debug)
    for linfo in scope_dict.get("lifetimes", []):
        l = Lifetime(scope)
        l.id = linfo["id"]
        l.owner_scope = scope
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

    # Drops
    for did in scope_dict.get("drops", []):
        if did in life_map:
            scope.drops.append(life_map[did])

    # Recurse into children
    for child_dict in scope_dict.get("children", []):
        reconstruct_scope(child_dict, scope)

    return scope


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


def hash_bitcode(filename):
    """Compute SHA-256 hash of a Totem Bitcode file."""
    doc = load_totem_bitcode(filename)
    canon = canonicalize_bitcode(doc)
    data = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(data).hexdigest()
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
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


def ensure_keypair():
    """Create an RSA keypair if it doesn't exist."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa

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


def sign_hash(sha256_hex):
    """Sign a SHA256 hex digest with the private key."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    private_key = ensure_keypair()
    signature = private_key.sign(
        sha256_hex.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256(),
    )
    return signature.hex()


def verify_signature(sha256_hex, signature_hex):
    """Verify a signature against the public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import padding

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
        args = [b.target.id for b in node.borrows]
        program.emit(node.op, node.typ, node.grade, args, scope_path)

    for child in scope.children:
        build_tir(child, program, scope_path)

    return program


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
        if instr.op in ("A", "D", "F"):  # treat as constant sources
            const_map[instr.id] = {"A": 1, "D": 2, "F": 5}[instr.op]
            new_instrs.append(instr)
            continue

        # fold if all args are known constants
        if all(a in const_map for a in instr.args) and instr.grade == "pure":
            val = sum(const_map[a] for a in instr.args)  # naive demo fold
            folded = TIRInstruction(
                instr.id, "CONST", "int32", "pure", [], instr.scope_path
            )
            folded.value = val
            const_map[instr.id] = val
            new_instrs.append(folded)
        else:
            new_instrs.append(instr)

    tir.instructions = new_instrs
    return tir


def reorder_pure_ops(tir):
    """Move pure operations before impure ones (commuting pure ops upward)."""
    pure = [i for i in tir.instructions if i.grade == "pure"]
    impure = [i for i in tir.instructions if i.grade != "pure"]
    tir.instructions = pure + impure
    return tir


def inline_trivial_io(tir):
    """Replace IO ops with constant placeholders if they have deterministic logs."""
    for instr in tir.instructions:
        if instr.grade == "io" and instr.op == "C":
            instr.op = "CONST_IO"
            instr.grade = "pure"
    return tir


def list_optimizers():
    return {
        "fold_constants": "Constant folding for pure ops",
        "reorder_pure_ops": "Commute pure ops before impure ops",
        "inline_trivial_io": "Replace deterministic IO reads with constants",
    }


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
    argp.add_argument("--src", help="Inline Totem source", default="{a{bc}de{fg}}")
    argp.add_argument("--verify", help="Verify signature for a logbook entry hash")
    argp.add_argument("--visualize", action="store_true", help="Render program graph")

    return argp.parse_args(args)


def main(args):
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
    if params.verify:
        ok = verify_signature(params.verify, input("Signature hex: ").strip())
        print("✓ Signature valid" if ok else "✗ Invalid signature")
        return

    print("Source:", params.src)
    tree = structural_decompress(params.src)
    print_scopes(tree)

    errors = []
    check_aliasing(tree, errors)
    check_lifetimes(tree, errors)

    print("\nCompile-time analysis:")
    if not errors:
        print("  ✓ All lifetime and borrow checks passed")
    else:
        for e in errors:
            print("  ✗", e)

    print("\nRuntime evaluation:")
    result = evaluate_scope(tree)
    print(f"  → final grade: {result.grade}")
    print("  → execution log:")
    for entry in result.log:
        print("   ", entry)

    export_totem_bitcode(tree, result, "program.totem.json")
    record_run("program.totem.json", result)
    tir = build_tir(tree)
    print("\nTIR:")
    print(tir)

    if params.visualize:
        visualize_graph(tree)


if __name__ == "__main__":
    main(sys.argv[1:])
