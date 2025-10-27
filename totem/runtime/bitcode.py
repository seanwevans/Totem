"""Totem Bitcode serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from ..constants import LOGBOOK_FILE
from ..ffi import FFIDeclaration
from .analysis import (
    build_bitcode_certificates,
    evaluate_scope,
    scope_to_dict,
    _collect_aliasing_payload,
    _grade_certificate,
)
from .core import Borrow, Lifetime, Node, Scope
from .compiler import build_tir
import sys

from . import crypto as _crypto


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
        raise ValueError(
            f"{name} certificate recomputation failed: {expected['summary']}"
        )


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
    runtime_mod = sys.modules.get("totem.runtime")
    signer = getattr(runtime_mod, "sign_hash", getattr(_crypto, "sign_hash", None))
    if signer is None:
        raise RuntimeError(
            "Cryptography support is unavailable; install the 'cryptography' package"
        )
    sig = signer(sha)

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

    logbook_path = getattr(runtime_mod, "LOGBOOK_FILE", LOGBOOK_FILE)
    with open(logbook_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"  📜 Recorded and signed run → {logbook_path}")
    return entry


def show_logbook(limit=10):
    """Display recent logbook entries."""
    logbook_path = getattr(
        sys.modules.get("totem.runtime"), "LOGBOOK_FILE", LOGBOOK_FILE
    )

    try:
        with open(logbook_path, "r", encoding="utf-8") as f:
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


__all__ = [
    "build_bitcode_document",
    "write_bitcode_document",
    "export_totem_bitcode",
    "load_totem_bitcode",
    "reconstruct_scope",
    "_validate_certificate",
    "verify_bitcode_document",
    "reexecute_bitcode",
    "canonicalize_bitcode",
    "hash_bitcode_document",
    "hash_bitcode",
    "diff_bitcodes",
    "record_run",
    "show_logbook",
]
