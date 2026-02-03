"""Regression test for BlockRealize boolean predicate handling.

This test verifies that tir.BlockRealize correctly handles all forms of
boolean predicates by converting them to uint1 IntImm values, which TVM's
C++ runtime can process.

Background:
- TVM C++ runtime rejects "bool" dtype with:
  CHECK(dtype.is_int() || dtype.is_uint()) << "cannot make const for type " << dtype;
- The fix converts any "bool" dtype predicates to IntImm("uint1", value)
- uint1 is TVM's canonical boolean representation and passes is_bool() checks
"""

import pytest
import tvm
from tvm import tir
from tvm.runtime import const


@pytest.mark.parametrize(
    "predicate_input,expected_value",
    [
        # Python bool literals
        (True, 1),
        (False, 0),
        # tir.const with "bool" dtype (legacy API)
        (const(True, "bool"), 1),
        (const(False, "bool"), 0),
        # Direct IntImm with "bool" dtype
        (tir.IntImm("bool", 1), 1),
        (tir.IntImm("bool", 0), 0),
        # Already correct uint1 dtype (should pass through)
        (tir.IntImm("uint1", 1), 1),
        (tir.IntImm("uint1", 0), 0),
    ],
)
def test_block_realize_predicate_conversion(predicate_input, expected_value):
    """Test that BlockRealize converts various boolean predicate forms to uint1."""
    block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="test_block",
        body=tir.Evaluate(0),
    )

    block_realize = tir.BlockRealize(
        iter_values=[],
        predicate=predicate_input,
        block=block,
    )

    # Verify the predicate was converted to uint1
    assert block_realize.predicate.dtype == "uint1", (
        f"Expected uint1 but got {block_realize.predicate.dtype} "
        f"for input {predicate_input}"
    )
    assert isinstance(block_realize.predicate, tir.IntImm)
    assert block_realize.predicate.value == expected_value


@pytest.mark.parametrize(
    "predicate_factory,expected_type,description",
    [
        (lambda: tir.EQ(tir.Var("i", "int32"), tir.Var("j", "int32")), tir.EQ, "EQ(i, j)"),
        (lambda: tir.Not(tir.Var("flag", "uint1")), tir.Not, "Not(flag)"),
        (lambda: tir.Or(tir.Var("a", "uint1"), tir.Var("b", "uint1")), tir.Or, "Or(a, b)"),
        (lambda: tir.And(tir.Var("a", "uint1"), tir.Var("b", "uint1")), tir.And, "And(a, b)"),
    ],
)
def test_block_realize_with_dynamic_predicates(predicate_factory, expected_type, description):
    """Test that dynamic predicates are preserved via Cast, not replaced with constant True.

    This is critical - dynamic predicates like EQ(i, j), Not(flag), Or(a, b) must NOT be
    replaced with constant True. They should be cast to uint1 while preserving their logic.
    """
    dynamic_pred = predicate_factory()
    assert str(dynamic_pred.dtype) == "bool", f"Predicate {description} should have bool dtype"

    block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="guarded_block",
        body=tir.Evaluate(0),
    )

    block_realize = tir.BlockRealize(
        iter_values=[],
        predicate=dynamic_pred,
        block=block,
    )

    # Predicate should be Cast(uint1, <original_expr>), not IntImm(uint1, 1)
    assert block_realize.predicate.dtype == "uint1", f"Failed for {description}"
    assert isinstance(block_realize.predicate, tir.Cast), f"Should be Cast for {description}"
    # The cast's inner value should preserve the original expression type
    assert isinstance(
        block_realize.predicate.value, expected_type
    ), f"Inner expression type mismatch for {description}"


def test_block_realize_in_primfunc_context():
    """Test BlockRealize with boolean predicate inside a PrimFunc.

    This mimics real-world usage in MLC-LLM compiler passes where
    BlockRealize is constructed with various predicate forms.
    """
    block = tir.Block(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=tir.Evaluate(0),
    )

    # Test with tir.const(True, "bool") - common pattern from TVMScript
    block_realize = tir.BlockRealize(
        iter_values=[],
        predicate=const(True, "bool"),
        block=block,
    )

    # Create a minimal PrimFunc - exercises full code path
    func = tir.PrimFunc(
        params=[],
        body=block_realize,
    )

    # The function should be valid and predicate should be uint1
    assert func.body.predicate.dtype == "uint1"
    assert func.body.predicate.value == 1


if __name__ == "__main__":
    pytest.main([__file__])
