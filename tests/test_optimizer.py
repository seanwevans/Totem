from totem import (
    TIRProgram,
    common_subexpression_elimination,
    dead_code_elimination,
    evaluate_pure_regions,
    inline_pure_regions,
    schedule_effects,
)


def ids(program):
    return [instr.id for instr in program.instructions]


def test_dead_code_elimination_prunes_unused_pure_ops():
    tir = TIRProgram()
    pure_used = tir.emit("A", "int32", "pure", [], "root", produces="life_a")
    pure_unused = tir.emit("F", "int32", "pure", [], "root", produces="life_unused")
    tir.emit("E", "int32", "pure", [{"target": "life_a"}], "root", produces="life_e")
    stateful = tir.emit(
        "B",
        "int32",
        "state",
        [{"target": "life_a"}],
        "root",
        produces="life_state",
    )

    dead_code_elimination(tir)

    remaining_ids = ids(tir)
    assert pure_unused not in remaining_ids
    assert pure_used in remaining_ids
    assert stateful in remaining_ids  # effectful instructions are preserved


def test_dead_code_elimination_removes_transitive_pure_producers():
    tir = TIRProgram()
    pure_root = tir.emit("A", "int32", "pure", [], "root", produces="life_a")
    pure_mid = tir.emit(
        "B",
        "int32",
        "pure",
        [{"target": "life_a"}],
        "root",
        produces="life_b",
    )
    pure_leaf = tir.emit(
        "C",
        "int32",
        "pure",
        [{"target": "life_b"}],
        "root",
        produces="life_c",
    )

    dead_code_elimination(tir)

    remaining_ids = ids(tir)
    assert pure_leaf not in remaining_ids
    assert pure_mid not in remaining_ids
    assert pure_root not in remaining_ids


def test_common_subexpression_elimination_merges_duplicates():
    tir = TIRProgram()
    base = tir.emit("A", "int32", "pure", [], "root", produces="life_a")
    first = tir.emit(
        "E",
        "int32",
        "pure",
        [{"target": "life_a"}],
        "root",
        produces="life_e1",
    )
    duplicate = tir.emit(
        "E",
        "int32",
        "pure",
        [{"target": "life_a"}],
        "root",
        produces="life_e2",
    )
    consumer = tir.emit(
        "D",
        "int32",
        "pure",
        [{"target": "life_e2"}],
        "root",
        produces="life_d",
    )

    common_subexpression_elimination(tir)

    remaining_ids = ids(tir)
    assert duplicate not in remaining_ids
    assert consumer in remaining_ids
    assert tir.instructions[-1].args[0]["target"] == "life_e1"


def test_evaluate_pure_regions_constant_and_partial():
    tir = TIRProgram()
    tir.emit("A", "int32", "pure", [], "root", produces="life_a")
    tir.emit("D", "int32", "pure", [], "root", produces="life_d")
    mixed = tir.emit(
        "E",
        "int32",
        "pure",
        [
            {"target": "life_a"},
            {"target": "life_d"},
            {"target": "dynamic"},
        ],
        "root",
        produces="life_e",
    )
    only_constants = tir.emit(
        "E",
        "int32",
        "pure",
        [{"target": "life_a"}, {"target": "life_d"}],
        "root",
        produces="life_const",
    )

    evaluate_pure_regions(tir)

    mixed_instr = next(instr for instr in tir.instructions if instr.id == mixed)
    assert mixed_instr.args[0] == {"kind": "const", "value": 3}
    assert mixed_instr.args[1:] == [{"target": "dynamic"}]

    const_instr = next(instr for instr in tir.instructions if instr.id == only_constants)
    assert const_instr.op == "CONST"
    assert getattr(const_instr, "value", None) == 3


def test_inline_pure_regions_promotes_child_scope():
    tir = TIRProgram()
    root_val = tir.emit("A", "int32", "pure", [], "root", produces="life_root")
    child_pure = tir.emit(
        "E",
        "int32",
        "pure",
        [{"target": "life_root"}],
        "root.scope",
        produces="life_child",
    )
    child_const = tir.emit("F", "int32", "pure", [], "root.scope", produces="life_const")
    effect_child = tir.emit(
        "C",
        "int32",
        "io",
        [{"target": "life_child"}],
        "root.io_scope",
        produces="life_io",
    )

    inline_pure_regions(tir)

    updated = {instr.id: instr.scope_path for instr in tir.instructions}
    assert updated[child_pure] == "root"
    assert updated[child_const] == "root"
    assert updated[effect_child] == "root.io_scope"  # unaffected due to effects


def test_schedule_effects_orders_by_grade():
    tir = TIRProgram()
    a = tir.emit("A", "int32", "pure", [], "root", produces="life_a")
    state = tir.emit("B", "int32", "state", [], "root", produces="life_state")
    pure_dep = tir.emit(
        "E",
        "int32",
        "pure",
        [{"target": "life_a"}],
        "root",
        produces="life_pure",
    )
    io_op = tir.emit(
        "G",
        "int32",
        "io",
        [{"target": "life_state"}],
        "root",
        produces="life_io",
    )

    schedule_effects(tir)

    grades = [instr.grade for instr in tir.instructions]
    assert grades.index("state") > grades.index("pure")
    assert grades.index("io") > grades.index("state")

