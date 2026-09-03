"""The two public entry points must compose without the caller unwrapping."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import (
    TIRProgram,
    TranspilationResult,
    as_tir_program,
    assemble_bytecode,
    build_tir,
    compute_tir_distance,
    emit_llvm_ir,
    emit_mlir_module,
    export_wasm_module,
    run_bytecode,
    structural_decompress,
    tir_to_wat,
    transpile_totem_to_tir,
)


def test_tir_to_wat_accepts_a_transpilation_result():
    result = transpile_totem_to_tir("a")
    assert isinstance(result, TranspilationResult)

    from_result, meta_result = tir_to_wat(result)
    from_tir, meta_tir = tir_to_wat(result.tir)

    assert from_result == from_tir
    assert meta_result == meta_tir


@pytest.mark.parametrize(
    "consumer",
    [emit_mlir_module, emit_llvm_ir],
)
def test_text_emitters_accept_both_types(consumer):
    result = transpile_totem_to_tir("{ad}", optimize=False)
    assert consumer(result) == consumer(result.tir)


def test_assemble_bytecode_accepts_a_transpilation_result():
    result = transpile_totem_to_tir("{ad}", optimize=False)
    assert run_bytecode(assemble_bytecode(result)).stack == (
        run_bytecode(assemble_bytecode(result.tir)).stack
    )


def test_compute_tir_distance_accepts_either_side():
    left = transpile_totem_to_tir("{ad}", optimize=False)
    right = transpile_totem_to_tir("{ae}", optimize=False)

    expected = compute_tir_distance(left.tir, right.tir)
    assert compute_tir_distance(left, right) == expected
    assert compute_tir_distance(left, right.tir) == expected
    assert compute_tir_distance(left.tir, right) == expected


def test_export_wasm_module_accepts_a_transpilation_result(tmp_path):
    result = transpile_totem_to_tir("a", optimize=False)
    metadata = export_wasm_module(result, tmp_path / "module.wat")

    assert (tmp_path / "module.wat").read_text(encoding="utf-8").startswith("(module")
    assert metadata["pure_instructions"] == 1


def test_as_tir_program_passes_programs_through_unchanged():
    program = build_tir(structural_decompress("{ad}"))
    assert as_tir_program(program) is program

    result = transpile_totem_to_tir("{ad}")
    assert as_tir_program(result) is result.tir


def test_as_tir_program_accepts_duck_typed_programs():
    class Stub:
        instructions = []

    stub = Stub()
    assert as_tir_program(stub) is stub


def test_as_tir_program_rejects_unrelated_objects():
    with pytest.raises(TypeError, match="TIRProgram or TranspilationResult"):
        as_tir_program(object())

    with pytest.raises(TypeError):
        as_tir_program("{ad}")


def test_transpiled_program_is_optimized_in_place():
    """Coercion must not hand back a copy: optimizing must reach result.tir."""

    from totem import optimize_tir

    result = transpile_totem_to_tir("{adad}", optimize=False)
    assert optimize_tir(result) is result.tir
