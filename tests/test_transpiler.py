from totem import (
    FFIDeclaration,
    TranspilationResult,
    build_tir,
    check_aliasing,
    check_lifetimes,
    clear_ffi_registry,
    get_registered_ffi_declarations,
    optimize_tir,
    register_ffi_declarations,
    structural_decompress,
    transpile_to_totem_ir,
    transpile_totem_to_tir,
    verify_ffi_calls,
)


def _manual_pipeline(src: str, *, optimize: bool):
    tree = structural_decompress(src)
    errors: list[str] = []
    check_aliasing(tree, errors)
    check_lifetimes(tree, errors)
    verify_ffi_calls(tree, errors)
    tir = build_tir(tree)
    if optimize:
        optimize_tir(tir)
    return tree, tuple(errors), tir


def test_transpiler_matches_manual_pipeline():
    src = "{a{bc}de{fg}}"

    expected_tree, expected_errors, expected_tir = _manual_pipeline(src, optimize=True)
    result = transpile_totem_to_tir(src)

    assert isinstance(result, TranspilationResult)
    assert result.source == src
    assert result.optimized is True
    assert result.errors == expected_errors
    assert repr(result.tir) == repr(expected_tir)
    assert result.tree.name == expected_tree.name
    assert len(result.tree.nodes) == len(expected_tree.nodes)


def test_transpiler_optional_optimization_controls_pipeline():
    src = "{aabb}"

    _, manual_errors, manual_tir = _manual_pipeline(src, optimize=False)
    result_no_opt = transpile_totem_to_tir(src, optimize=False)

    assert result_no_opt.optimized is False
    assert result_no_opt.errors == manual_errors
    assert repr(result_no_opt.tir) == repr(manual_tir)
    assert len(result_no_opt.tir.instructions) == len(manual_tir.instructions)

    result_opt = transpile_totem_to_tir(src, optimize=True)
    assert result_opt.optimized is True
    assert len(result_opt.tir.instructions) < len(result_no_opt.tir.instructions)
    assert isinstance(transpile_to_totem_ir(src), TranspilationResult)


def test_transpiler_restores_ffi_registry():
    base_decl = FFIDeclaration("BASE", "pure", [], "int32")
    register_ffi_declarations([base_decl], reset=True)
    baseline = get_registered_ffi_declarations()

    extra_decl = FFIDeclaration("EXTRA", "io", [], "int32")
    result = transpile_totem_to_tir("ab", ffi_decls=[extra_decl])

    assert isinstance(result, TranspilationResult)
    assert get_registered_ffi_declarations() == baseline

    clear_ffi_registry()
