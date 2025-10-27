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

from . import core as _core
from . import capabilities as _capabilities
from . import actors as _actors
from . import tir as _tir
from . import meta as _meta
from . import analysis as _analysis
from . import bitcode as _bitcode
from . import crypto as _crypto
from . import compiler as _compiler
from .cli import main, parse_args, run_repl
from ..constants import KEY_FILE, LOGBOOK_FILE, PUB_FILE
from ..ffi import (
    FFI_REGISTRY,
    FFIDeclaration,
    clear_ffi_registry,
    get_registered_ffi_declarations,
    parse_inline_ffi,
    register_ffi_declarations,
)

from .core import *
from .capabilities import *
from .actors import *
from .tir import *
from .meta import *
from .analysis import *
from .bitcode import *
from .crypto import *
from .compiler import *

__all__ = []
for module in (_core, _capabilities, _actors, _tir, _meta, _analysis, _bitcode, _crypto, _compiler):
    __all__.extend(getattr(module, '__all__', []))
__all__ += ['main', 'parse_args', 'run_repl', 'parse_inline_ffi', 'FFI_REGISTRY', 'FFIDeclaration', 'clear_ffi_registry', 'get_registered_ffi_declarations', 'register_ffi_declarations', 'KEY_FILE', 'LOGBOOK_FILE', 'PUB_FILE']
__all__ = list(dict.fromkeys(__all__))
