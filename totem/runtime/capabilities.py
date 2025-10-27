"""Capability primitives for the Totem runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Capability:
    """Linear capability token tracking resource usage."""

    def __init__(
        self,
        kind: str,
        resource: str | None = None,
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        generation: int = 0,
        active: bool = True,
        permissions: set[str] | frozenset[str] | None = None,
        revoked: bool = False,
    ) -> None:
        self.kind = kind
        self.resource = resource
        self.state = state or {}
        self.history = history or []
        self.generation = generation
        self._active = active
        self.permissions = None if permissions is None else frozenset(permissions)
        self.revoked = revoked

    def evolve(
        self,
        action: str,
        detail: Any = None,
        state_updates: dict[str, Any] | None = None,
        *,
        permission_subset: set[str] | None = None,
        revoke: bool = False,
    ) -> "Capability":
        if not self.is_active:
            raise RuntimeError(f"Capability {self} already consumed")

        new_state = dict(self.state)
        if state_updates:
            for key, value in state_updates.items():
                new_state[key] = value

        new_history = list(self.history)
        new_history.append({"action": action, "detail": detail})

        if revoke and permission_subset is not None:
            raise ValueError("Cannot restrict permissions while revoking a capability")

        if permission_subset is not None:
            subset = frozenset(permission_subset)
            if self.permissions is not None and not subset.issubset(self.permissions):
                missing = sorted(subset.difference(self.permissions))
                raise ValueError(
                    f"Capability {self} cannot grant permissions outside of its scope: {missing}"
                )
            new_permissions = subset
            new_revoked = False
            new_active = True
        elif revoke:
            new_permissions = frozenset()
            new_revoked = True
            new_active = False
        else:
            new_permissions = self.permissions
            new_revoked = self.revoked
            new_active = True

        self._active = False

        return Capability(
            self.kind,
            self.resource,
            state=new_state,
            history=new_history,
            generation=self.generation + 1,
            permissions=new_permissions,
            active=new_active,
            revoked=new_revoked,
        )

    @property
    def is_active(self) -> bool:
        return self._active and not self.revoked

    def has_permission(self, permission: str) -> bool:
        if self.revoked:
            return False
        if self.permissions is None:
            return True
        return permission in self.permissions

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return f"<Capability {self.kind}@{self.generation}>"


@dataclass(frozen=True)
class CapabilityUseResult:
    capability: Capability
    value: Any = None


def resolve_value(value: Any) -> Any:
    if isinstance(value, CapabilityUseResult):
        return value.value
    return value


def extract_capability(value: Any) -> Capability | None:
    if isinstance(value, CapabilityUseResult):
        return value.capability
    if isinstance(value, Capability):
        return value
    return None


def _clone_list(source: list[Any] | None) -> list[Any]:
    return list(source) if source is not None else []


def use_file_read(cap: Capability) -> CapabilityUseResult:
    if not cap.has_permission("read"):
        raise RuntimeError(f"Capability {cap} does not permit read operations")
    index = cap.state.get("index", 0)
    contents = cap.state.get("contents", [])
    if index < len(contents):
        data = contents[index]
    else:
        data = None
    new_cap = cap.evolve("read", data, {"index": index + 1})
    return CapabilityUseResult(new_cap, data)


def use_file_write(cap: Capability, payload: Any) -> CapabilityUseResult:
    if not cap.has_permission("write"):
        raise RuntimeError(f"Capability {cap} does not permit write operations")
    writes = _clone_list(cap.state.get("writes", []))
    writes.append(payload)
    new_cap = cap.evolve("write", payload, {"writes": writes})
    return CapabilityUseResult(new_cap, True)


def use_net_send(cap: Capability, payload: Any) -> CapabilityUseResult:
    if not cap.has_permission("send"):
        raise RuntimeError(f"Capability {cap} does not permit network send operations")
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
        permissions={"read"},
    ),
    "FileWrite": lambda: Capability(
        "FileWrite",
        resource="output",
        state={"writes": []},
        permissions={"write"},
    ),
    "NetSend": lambda: Capability(
        "NetSend",
        resource="socket",
        state={"transmissions": []},
        permissions={"send"},
    ),
}


def create_default_environment() -> dict[str, Any]:
    env: dict[str, Any] = {"__capabilities__": {}}
    for kind, factory in CAPABILITY_FACTORIES.items():
        env["__capabilities__"][kind] = factory()
    return env


def ensure_capability(env: dict[str, Any], kind: str) -> Capability:
    caps = env.setdefault("__capabilities__", {})
    if kind not in caps:
        caps[kind] = CAPABILITY_FACTORIES[kind]()
    return caps[kind]


def store_capability(env: dict[str, Any], kind: str, capability: Capability) -> None:
    env.setdefault("__capabilities__", {})[kind] = capability


__all__ = [
    "CAPABILITY_FACTORIES",
    "Capability",
    "CapabilityUseResult",
    "create_default_environment",
    "ensure_capability",
    "extract_capability",
    "resolve_value",
    "store_capability",
    "use_file_read",
    "use_file_write",
    "use_net_send",
]
