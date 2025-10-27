# 🪶 Totem  

**Totem** is a total, decompressive, typed language.  
Every UTF-8 string is a valid Totem program — there are no syntax errors, ever.

Instead of parsing text, Totem **inflates** information.  
The compiler acts as a **semantic decompressor**, mapping raw bytes into a fully typed, scope-aware, and lifetime-safe computation graph.

Totem unifies **type theory**, **information theory**, and **systems safety** into a single algebraic model:

```
TOTEM := (Σ*, D, ⊗, T, E)

where
  Σ* = UTF-8 program space
  D  = total decompressor → typed AST graph
  ⊗  = monoidal composition operator
  T  = type / lifetime / effect lattice
  E  = total evaluator
```

---

## Why Totem?

| Principle | Meaning |
|------------|----------|
| **Totality** | Every UTF-8 input inflates into a well-typed program. No syntax errors, no undefined behavior. |
| **Arity Algebra** | Function signatures and types emerge automatically from argument structure (arity). |
| **Scoped Geometry** | `{}` and `()` build real lexical and evaluation scopes; ownership and lifetimes follow geometry. |
| **Purity Gradient** | Effects are modeled as a graded monad (`pure ⊂ state ⊂ io ⊂ sys ⊂ meta`). |
| **Safe by Construction** | Lifetimes and borrows are inferred automatically; aliasing and use-after-free are impossible. |
| **Decompressive Compilation** | The compiler expands compressed meaning into structured, typed computation. |
| **Continuous Semantics** | Similar bytes produce similar programs — enabling mutation, learning, and program synthesis. |

---

## Example

```totem
{a{bc}de{fg}}
```

Inflates to a nested scope graph:

```
Scope(root)
  Scope(scope_0)
    A:int32
    Scope(scope_1)
      B:int32  → shared borrow of A
      C:int32  → mut borrow of B
      drops [B, C]
    D:int32  → shared borrow of A
    E:int32  → mut borrow of D
    Scope(scope_2)
      F:int32  → shared borrow of E
      G:int32  → mut borrow of F
      drops [F, G]
    drops [A, D, E]
```

Compile-time checks guarantee:
- All borrows are valid.
- No mutable + shared aliasing.
- No borrow outlives its owner.
- All values are dropped safely at end of scope.

---

## Architecture

```
UTF-8 bytes
    ↓
[ Decompressor ]
    → builds scopes `{}` `()`
    → infers operator arity and types
    ↓
[ Type & Lifetime Graph ]
    → computes ownership, borrows, drops
    ↓
[ Purity / Effect Analysis ]
    → isolates impure nodes in effect scopes
    ↓
[ Evaluator / Backend ]
    → executes total graph deterministically
```

Every stage is **total** — no parsing errors, no undefined states.

---

## Philosophy

> Programming is compression.  
> Compilation is decompression.  
> Safety is geometry.

Totem treats computation as **information un-folding**:  
code is not written, it is *unpacked* from data.  
Every byte belongs, every operation finds its lawful place in the algebra.

---

## Current Status

- ✅ Structural decompressor → builds scopes and typed nodes  
- ✅ Lifetime & borrow inference  
- ✅ Compile-time safety analysis  
- 🔜 Purity / effect monad layer  
- 🔜 Typed IR and interpreter  
- 🔜 Rust / MLIR backend  
- 🔜 Visualizer for scopes, lifetimes, and effects

---

## Roadmap

1. **Purity-aware evaluation** — execute total graphs with effect isolation.  
2. **Typed IR (TIR)** — a linear SSA form for Totem graphs.  
3. **Backend targets** — generate Rust or LLVM IR.  
4. **Visualizer** — inspect the decompressed program graph in real time.  
5. **Continuous semantics** — experiment with evolutionary and neural programming over Totem space.  

---

## License

MIT (for now). Totem is an experiment in total languages and safe compilation.

---

## Credits

**Totem** is an original research language concept by Sean Evans.  
It blends ideas from Rust, type theory, category theory, and information geometry  
to explore a world where *no program is ever invalid*.
