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
