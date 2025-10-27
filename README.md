[![CI](https://github.com/seanwevans/Totem/actions/workflows/ci.yml/badge.svg)](https://github.com/seanwevans/Totem/actions/workflows/ci.yml)
![Coverage](./coverage.svg)

# 🪶 Totem — A No-Syntax-Error Programming Language

## 🌐 Overview

**Totem** is a self-contained language and runtime built around a single principle:

> *No syntax errors. Every string compiles.*

Totem treats source text as a compressed representation of structure, effects, and computation.  
The compiler acts as a *structural decompressor*, inferring lifetimes, purity, and control flow.

### 🧩 Design Stack

| Layer | Description | Status |
|--------|-------------|--------|
| **Structural Decompressor** | Every UTF-8 string → valid scoped AST | ✅ |
| **Type & Lifetime Inference** | Rust-like ownership, drops, borrows | ✅ |
| **Purity/Effect Lattice** | `Pure ⊂ State ⊂ IO ⊂ Sys ⊂ Meta` | ✅ |
| **Evaluator** | Graded effect monad runtime | ✅ |
| **Visualization** | NetworkX graph of scopes & lifetimes | ✅ |
| **Bitcode Serialization** | Portable `.totem.json` IR | ✅ |
| **Reload & Re-Execution** | Deterministic round-trip | ✅ |
| **Hash & Diff** | Semantic identity | ✅ |
| **Logbook Ledger** | Provenance tracking | ✅ |
| **Cryptographic Signatures** | Proof-of-origin | ✅ |

---

## ⚙️ Core Ideas

### 1. **No Syntax Errors**
Every byte sequence is structurally valid. Braces define scopes; letters define operations.

```bash
{a{bc}de{fg}}
```

This expands into nested scopes with lifetimes, borrows, and drops — all inferred automatically.

### 2. **Ownership & Borrowing**
Totem automatically assigns lifetimes and enforces Rust-style aliasing rules:
- Mutable (`mut`) and shared (`shared`) borrows are exclusive.
- No borrow can outlive its parent lifetime.

### 3. **Effect Lattice**
Purity is compositional:

```
Pure ⊂ State ⊂ IO ⊂ Sys ⊂ Meta
```

Each node propagates its grade through a *graded monad*, preserving effect isolation at runtime.

### 4. **Typed Intermediate Representation (TIR)**
Totem lowers decompressed scopes into a simple SSA-like IR:

```plaintext
v0 = A() : int32 [pure] @root.scope_0
v1 = B() : int32 [state] @root.scope_0.scope_0
v2 = C(v1) : int32 [io] @root.scope_0.scope_0
```

This TIR serves as the canonical portable format — similar to LLVM bitcode, but fully serializable.

### 5. **Provenance & Verification**
Every compiled run is cryptographically signed and recorded in a ledger:

```bash
  📜 Recorded and signed run → totem.logbook.jsonl
  SHA256(program.totem.json) = <digest>
```

This enables reproducible builds and verifiable provenance.

---

## 🧮 Example Session

```bash
$ ./totem.py --src "{a{bc}de{fg}}"
Source: {a{bc}de{fg}}
Compile-time analysis:
  ✓ All lifetime and borrow checks passed

Runtime evaluation:
  → final grade: io
  → execution log:
    A:1
    D:2
    E:5
    B:inc->1
    C:read->input_data
    F:5
    G:write(5)
  ✓ Totem Bitcode exported → program.totem.json
  📜 Recorded and signed run → totem.logbook.jsonl
```

### Reload and Verify

```bash
$ ./totem.py --load program.totem.json
$ ./totem.py --hash program.totem.json
$ ./totem.py --logbook
$ ./totem.py --diff program1.totem.json program2.totem.json
```

---

## 🔐 Provenance Chain

Totem automatically maintains an append-only **logbook ledger**, storing:

- SHA256 hash of the bitcode  
- RSA signature  
- Execution metadata (grade, logs)  
- Timestamped entries for reproducibility

Keys are stored locally in:
```
totem_private_key.pem
totem_public_key.pem
```

---

## 🧭 Meta Layer

Totem supports *self-reflective meta-operations* (`Meta` grade):

| Operation | Description |
|------------|-------------|
| `reflect()` | Returns a `MetaObject` representation of a runtime structure |
| `meta_emit()` | Dynamically extends the TIR at runtime |
| `fold_constants()` | Performs constant folding |
| `reorder_pure_ops()` | Effect-sensitive sort that respects dependencies |
| `inline_trivial_io()` | Inlines deterministic IO as constants |

---

## 📊 Visualization

Visualize lifetimes and borrows with:

```bash
$ ./totem.py --visualize
```

The graph displays:
- Nodes colored by purity (`green=pure`, `yellow=state`, `red=io`)
- Dashed edges for borrows
- Nested scopes as clusters

---

## ⚡ WebAssembly Backend

Totem can lower the **pure portion of the TIR** into a WebAssembly module. All
`IO`-graded nodes become **host imports guarded by explicit capabilities** so the
runtime must opt-in to side effects.

```bash
$ ./totem.py \
    --src "{a{bc}de{fg}}" \
    --wasm web/program.wat \
    --wasm-metadata web/program.wasm.json \
    --capability io.read \
    --capability io.write
```

- Pure instructions (`A`, `D`, `E`, `F`) are emitted as local computations.
- IO instructions (`C`, `G`) become imports (`totem_io.io_read/io_write`).
- Granting a capability is mandatory; missing permissions raise an error.

The optional metadata file records which capabilities were imported, how many
pure vs. IO instructions were seen, and the locals allocated in the generated
module.

### 🖥️ Browser Demo

Serve the `web/` directory to see the WASM backend and lifetime viz together:

```bash
$ python totem.py --src "{a{bc}de{fg}}" \
    --wasm web/program.wat \
    --wasm-metadata web/program.wasm.json \
    --capability io.read --capability io.write
$ cp program.totem.json web/
$ python -m http.server --directory web
```

Then open `http://localhost:8000/demo.html` to:

1. Compile the emitted WAT to WASM in the browser using `wabt.js`.
2. Run the exported `run()` function with capability-gated IO shims.
3. Visualize the Totem scope graph via D3, reusing the serialized bitcode.

---

## 🧱 Architecture Roadmap

| Phase | Goal | Description |
|-------|------|-------------|
| I | Core Runtime | Structural decompressor, evaluator, bitcode |
| II | Meta Runtime | Reflection, TIR mutation, optimizers |
| III | Formal Semantics | Add symbolic typing, proof-carrying code |
| IV | Totem VM | Execute TIR as bytecode or LLVM IR |
| V | Distributed Provenance | Peer-signed logbook synchronization |
