import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

from totem import build_tir, emit_llvm_ir, emit_mlir_module, structural_decompress


def _build_sample_tir(src):
    scope = structural_decompress(src)
    return build_tir(scope)


def test_emit_mlir_module_contains_lattice_and_ops():
    tir = _build_sample_tir("{ad}")
    mlir = emit_mlir_module(tir)

    assert "totem.effect_lattice" in mlir
    assert '"totem.a"' in mlir
    assert 'grade = "pure"' in mlir


def test_emit_llvm_ir_mentions_constants_and_calls():
    tir = _build_sample_tir("{ade}")
    llvm = emit_llvm_ir(tir)

    assert "define void @totem_main" in llvm
    assert "add i32 0, 1" in llvm
    assert "call i32 @totem_e" in llvm
