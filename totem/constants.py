"""Shared constant values for the Totem runtime."""

EFFECT_GRADES = ["pure", "state", "io", "sys", "meta"]

GRADE_COLORS = {
    "pure": "#8BC34A",
    "state": "#FFEB3B",
    "io": "#FF7043",
    "sys": "#9575CD",
    "meta": "#B0BEC5",
}

OPS = {
    "A": {"grade": "pure"},
    "B": {"grade": "state"},
    "C": {"grade": "io"},
    "D": {"grade": "pure"},
    "E": {"grade": "pure"},
    "F": {"grade": "pure"},
    "G": {"grade": "io"},
    "H": {"grade": "sys"},
    "J": {"grade": "sys"},
    "K": {"grade": "pure"},
    "L": {"grade": "sys"},
    "P": {"grade": "sys"},
    "S": {"grade": "sys"},
    "M": {"grade": "meta"},
    "N": {"grade": "meta"},
    "O": {"grade": "meta"},
    "Q": {"grade": "meta"},
}

IO_IMPORTS = {
    "C": {
        "capability": "io.read",
        "module": "totem_io",
        "name": "io_read",
        "params": [],
        "results": ["i32"],
    },
    "G": {
        "capability": "io.write",
        "module": "totem_io",
        "name": "io_write",
        "params": ["i32"],
        "results": [],
    },
}

PURE_CONST_VALUES = {"A": 1, "D": 2, "F": 5}

LOGBOOK_FILE = "totem.logbook.jsonl"
KEY_FILE = "totem_private_key.pem"
PUB_FILE = "totem_public_key.pem"
REPL_HISTORY_LIMIT = 10

PURE_CONSTANTS = {
    "A": 1,
    "D": 2,
    "F": 5,
}

__all__ = [
    "EFFECT_GRADES",
    "GRADE_COLORS",
    "OPS",
    "IO_IMPORTS",
    "PURE_CONST_VALUES",
    "PURE_CONSTANTS",
    "LOGBOOK_FILE",
    "KEY_FILE",
    "PUB_FILE",
    "REPL_HISTORY_LIMIT",
]
