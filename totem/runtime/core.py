"""Core runtime data structures for Totem."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Optional

from ..constants import EFFECT_GRADES, OPS
from ..ffi import FFI_REGISTRY


def _scope_path(scope: "Scope | None") -> str:
    parts: list[str] = []
    while scope is not None:
        parts.append(scope.name)
        scope = scope.parent
    return ".".join(reversed(parts))


def _stable_id(scope_path: str, index: int) -> str:
    token = f"{scope_path}:{index}".encode("utf-8")
    return hashlib.blake2s(token, digest_size=6).hexdigest()


def _scope_full_path(scope: "Scope | None") -> str:
    """Return a human-readable scope path for display."""

    parts: list[str] = []
    while scope is not None:
        parts.append(scope.name)
        scope = scope.parent
    return " > ".join(reversed(parts))


class Lifetime:
    def __init__(self, owner_scope: "Scope", identifier: str):
        self.id = identifier
        self.owner_scope = owner_scope
        self.end_scope: Optional["Scope"] = None
        self.borrows: list["Borrow"] = []
        self.owner_node: Optional["Node"] = None

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        end = self.end_scope.name if self.end_scope else "?"
        return f"Life({self.id}@{self.owner_scope.name}->{end})"


class Borrow:
    def __init__(self, kind: str, target: Lifetime, borrower_scope: "Scope"):
        self.kind = kind
        self.target = target
        self.borrower_scope = borrower_scope

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"{self.kind}→{self.target.id}@{self.borrower_scope.name}"


def _arity_type_name(arity: int) -> str:
    """Return the canonical Totem type name for a constructor of a given arity."""

    return f"ADT<{arity}>"


class Node:
    def __init__(self, op: str, typ: str, scope: "Scope"):
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
        self.borrows: list[Borrow] = []
        self.grade = OPS.get(op, {}).get("grade", "pure")
        self.ffi = None
        self.ffi_capabilities: list[str] = []
        self.meta: dict[str, Any] = {}
        self.attached_scopes: list["Scope"] = []
        self.arity = 0
        self._apply_ffi_metadata()
        self.update_type()

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<{self.op}:{self.typ}@{self.scope.name}>"

    def _apply_ffi_metadata(self) -> None:
        decl = FFI_REGISTRY.get(self.op)
        if not decl:
            return
        self.ffi = decl
        self.grade = decl.grade
        self.typ = decl.return_type
        self.ffi_capabilities = list(decl.capabilities)
        self.meta.setdefault("fixed_type", decl.return_type)

    def update_type(self) -> str:
        """Refresh this node's inferred type based on current metadata and borrows."""

        self.arity = len(self.borrows)
        if "fixed_type" in self.meta:
            self.typ = self.meta["fixed_type"]
        elif self.op == "P":
            self.typ = "match"
        else:
            self.typ = _arity_type_name(self.arity)
        return self.typ


class IRNode:
    """Lowered SSA-like form."""

    def __init__(self, id: str, op: str, typ: str, grade: str, args: Iterable[Any]):
        self.id = id
        self.op = op
        self.typ = typ
        self.grade = grade
        self.args = list(args)


class Scope:
    def __init__(
        self, name: str, parent: "Scope | None" = None, *, effect_cap=None, fence=None
    ):
        self.name = name
        self.parent = parent
        self.nodes: list[Node] = []
        self.children: list[Scope] = []
        self.lifetimes: list[Lifetime] = []
        self.drops: list[Lifetime] = []
        self.effect_cap = effect_cap
        self.fence = fence
        self.meta_role = None
        if parent:
            parent.children.append(self)

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"Scope({self.name})"


class Effect:
    """Graded monad for purity tracking."""

    def __init__(self, grade: str, value: Any, log: Optional[list[str]] = None):
        self.grade = grade
        self.value = value
        self.log = log or []

    def bind(self, fn: Callable[[Any], "Effect"]) -> "Effect":
        out = fn(self.value)
        new_idx = max(EFFECT_GRADES.index(self.grade), EFFECT_GRADES.index(out.grade))
        return Effect(EFFECT_GRADES[new_idx], out.value, self.log + out.log)


class MovedValue:
    """Sentinel stored in the environment when a lifetime has been moved."""

    def __init__(self, origin_id: str):
        self.origin_id = origin_id

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<moved:{self.origin_id}>"


def read_env_value(
    env: dict[str, Any], lifetime_id: str | None, default: Any = None
) -> Any:
    """Fetch a lifetime's value while ensuring it has not been moved."""

    if lifetime_id is None:
        return default
    if lifetime_id not in env:
        if default is not None:
            return default
        raise KeyError(f"Unknown lifetime {lifetime_id}")
    val = env[lifetime_id]
    if isinstance(val, MovedValue):
        raise RuntimeError(
            f"Lifetime {lifetime_id} has been moved and is no longer usable"
        )
    return val


def move_env_value(env: dict[str, Any], lifetime_id: str | None) -> Any:
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


__all__ = [
    "Borrow",
    "Effect",
    "IRNode",
    "Lifetime",
    "MovedValue",
    "Node",
    "Scope",
    "_arity_type_name",
    "_scope_full_path",
    "_scope_path",
    "_stable_id",
    "move_env_value",
    "read_env_value",
]
