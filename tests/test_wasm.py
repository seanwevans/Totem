import pytest

from totem import TIRProgram, build_tir, structural_decompress, tir_to_wat


def sample_tir():
    root = structural_decompress("{a{bc}de{fg}}")
    return build_tir(root)


def test_tir_to_wat_requires_io_capabilities():
    tir = sample_tir()
    with pytest.raises(PermissionError):
        tir_to_wat(tir)


def test_tir_to_wat_exports_imports_and_metadata():
    tir = sample_tir()
    wat, metadata = tir_to_wat(tir, capabilities={"io.read", "io.write"})

    assert wat.startswith("(module")
    assert '(import "totem_io" "io_read"' in wat
    assert '(import "totem_io" "io_write"' in wat
    assert "(func $run" in wat
    assert "(return (local.get" in wat

    assert metadata["imports"] == ["io.read", "io.write"]
    assert metadata["pure_instructions"] >= 1
    assert metadata["io_instructions"] >= 1


def test_tir_to_wat_rejects_io_args_from_non_lowered_instr():
    program = TIRProgram()
    state_val = program.emit("B", "int32", "state", args=[], scope_path="root")
    program.emit(
        "G",
        "void",
        "io",
        args=[{"target": state_val}],
        scope_path="root",
    )

    with pytest.raises(ValueError, match="cannot be lowered to WebAssembly"):
        tir_to_wat(program, capabilities={"io.write"})


def test_single_character_e_program_compiles():
    """A one-byte program must lower; E with no borrow denotes 3."""

    wat, metadata = tir_to_wat(build_tir(structural_decompress("e")))

    assert "(i32.const 3)" in wat
    assert "(return (local.get" in wat
    assert metadata["pure_instructions"] == 1


def test_zero_borrow_e_agrees_with_the_evaluators():
    """The WASM backend must not disagree with the interpreters about E."""

    from totem import assemble_bytecode, compile_and_evaluate, run_bytecode

    tir = build_tir(structural_decompress("e"))
    wat, _ = tir_to_wat(tir)

    assert run_bytecode(assemble_bytecode(tir)).stack == [3]
    assert compile_and_evaluate("e")[2].log == ["E:3"]
    assert "(local.set $v0 (i32.const 3))" in wat


def test_borrowed_e_still_adds_three():
    tir = build_tir(structural_decompress("ae"))
    wat, _ = tir_to_wat(tir)

    assert "i32.add" in wat
    assert "(i32.const 3)" in wat
