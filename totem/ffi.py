"""Foreign function interface declarations and registry."""

from dataclasses import dataclass
import json
import re

from .constants import EFFECT_GRADES


@dataclass
class FFIDeclaration:
    """Metadata describing a host-provided foreign function."""

    name: str
    grade: str
    arg_types: list
    return_type: str
    capabilities: list | None = None

    def __post_init__(self):
        self.name = (self.name or "").strip().upper()
        if not self.name:
            raise ValueError("FFI declaration requires a name")
        self.grade = (self.grade or "").strip().lower()
        if self.grade not in EFFECT_GRADES:
            raise ValueError(
                f"FFI declaration {self.name} has unknown grade: {self.grade}"
            )
        self.arg_types = [a.strip() for a in (self.arg_types or []) if a.strip()]
        self.return_type = (self.return_type or "").strip() or "void"
        caps = self.capabilities or []
        self.capabilities = [c.strip() for c in caps if c.strip()]

    @property
    def arity(self):
        return len(self.arg_types)

    def to_dict(self):
        return {
            "name": self.name,
            "grade": self.grade,
            "arg_types": list(self.arg_types),
            "return_type": self.return_type,
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("FFI declaration must be built from a mapping")
        name = data.get("name")
        grade = data.get("grade")
        arg_types = data.get("arg_types") or data.get("args") or []
        return_type = data.get("return_type") or data.get("returns")
        capabilities = (
            data.get("capabilities")
            or data.get("requires")
            or data.get("capability_requirements")
            or []
        )
        return cls(name, grade, arg_types, return_type, capabilities)


FFI_REGISTRY = {}


INLINE_FFI_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<grade>[a-z]+)\s*"
    r"\((?P<args>[^)]*)\)\s*->\s*(?P<ret>[^|]+?)\s*"
    r"(?:\|\s*requires\s*(?P<caps>.+))?$"
)


def parse_inline_ffi(schema):
    """Parse a simple inline FFI schema into declarations."""

    if not schema:
        return []

    declarations = []
    for line in schema.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = INLINE_FFI_PATTERN.match(entry)
        if not match:
            raise ValueError(f"Invalid inline FFI declaration: {entry}")
        args = match.group("args").strip()
        arg_types = [a.strip() for a in args.split(",") if a.strip()] if args else []
        caps_text = match.group("caps")
        caps = []
        if caps_text:
            caps = [c.strip() for c in caps_text.split(",") if c.strip()]
        declarations.append(
            FFIDeclaration(
                name=match.group("name"),
                grade=match.group("grade"),
                arg_types=arg_types,
                return_type=match.group("ret").strip(),
                capabilities=caps,
            )
        )
    return declarations


def _normalize_ffi_declarations(spec):
    """Normalize any supported FFI spec into FFIDeclaration objects."""

    if spec is None:
        return []
    if isinstance(spec, FFIDeclaration):
        return [spec]
    if isinstance(spec, str):
        trimmed = spec.strip()
        if not trimmed:
            return []
        if trimmed[0] in "[{":
            data = json.loads(trimmed)
            return _normalize_ffi_declarations(data)
        return parse_inline_ffi(trimmed)
    if isinstance(spec, dict):
        # Support either a single declaration dict or a wrapper with "ffi" key.
        if "declarations" in spec and isinstance(spec["declarations"], list):
            return _normalize_ffi_declarations(spec["declarations"])
        if "ffi" in spec and isinstance(spec["ffi"], list):
            return _normalize_ffi_declarations(spec["ffi"])
        return [FFIDeclaration.from_dict(spec)]
    if isinstance(spec, (list, tuple)):
        decls = []
        for item in spec:
            decls.extend(_normalize_ffi_declarations(item))
        return decls
    raise TypeError(f"Unsupported FFI spec type: {type(spec)!r}")


def register_ffi_declarations(spec, *, reset=False):
    """Register one or more FFI declarations in the global registry."""

    if reset:
        FFI_REGISTRY.clear()
    for decl in _normalize_ffi_declarations(spec):
        if decl.name in FFI_REGISTRY:
            raise ValueError(f"Duplicate FFI declaration for {decl.name}")
        FFI_REGISTRY[decl.name] = decl


def clear_ffi_registry():
    """Remove all registered FFI declarations."""

    FFI_REGISTRY.clear()


def get_registered_ffi_declarations():
    """Return a snapshot of the currently registered declarations."""

    return {name: decl for name, decl in FFI_REGISTRY.items()}


__all__ = [
    "FFIDeclaration",
    "FFI_REGISTRY",
    "parse_inline_ffi",
    "register_ffi_declarations",
    "clear_ffi_registry",
    "get_registered_ffi_declarations",
]
