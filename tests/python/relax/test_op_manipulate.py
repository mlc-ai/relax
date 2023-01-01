# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    assert relax.op.reshape(x, (4, 5, 3)).op == Op.get("relax.reshape")
    assert relax.op.permute_dims(x).op == Op.get("relax.permute_dims")
    assert relax.op.expand_dims(x, axis=[]).op == Op.get("relax.expand_dims")
    assert relax.op.squeeze(x).op == Op.get("relax.squeeze")
    assert relax.op.flatten(x).op == Op.get("relax.flatten")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_reshape_infer_struct_into():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    s0 = relax.Var("s", R.Shape((3, 8, 5)))
    s1 = relax.Var("s", R.Shape(ndim=3))
    s2 = relax.Var("s", R.Shape())

    _check_inference(
        bb, relax.op.reshape(x0, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x0, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(bb, relax.op.reshape(x0, (-1,)), relax.TensorStructInfo((120,), "float32"))
    _check_inference(
        bb, relax.op.reshape(x1, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x2, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x3, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x3, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x4, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x5, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(bb, relax.op.reshape(x0, s0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s0), relax.TensorStructInfo(s0, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s0), relax.TensorStructInfo(s0, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s0), relax.TensorStructInfo(s0, dtype=""))
    _check_inference(bb, relax.op.reshape(x0, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x0, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s2), relax.TensorStructInfo(s2, dtype=""))


def test_reshape_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))
    s0 = relax.Var("s", R.Shape((c, a, d, b)))
    s1 = relax.Var("s", R.Shape())

    _check_inference(
        bb, relax.op.reshape(x, (c, a, d, b)), relax.TensorStructInfo((c, a, d, b), "float32")
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (d, c, b, -1)),
        relax.TensorStructInfo((d, c, b, tir.floordiv(a * b * c * d, d * c * b)), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (1, -1, 1)),
        relax.TensorStructInfo((1, a * b * c * d, 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (2, -1, a)),
        relax.TensorStructInfo((2, tir.floordiv(a * b * c * d, a * 2), a), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, -1, d, b)),
        relax.TensorStructInfo((c, tir.floordiv(a * b * c * d, c * d * b), d, b), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, a * d, b)),
        relax.TensorStructInfo((c, a * d, b), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, a * b * d, -1)),
        relax.TensorStructInfo(
            (c, a * b * d, tir.floordiv(a * b * c * d, c * (a * b * d))), "float32"
        ),
    )
    _check_inference(bb, relax.op.reshape(x, s0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.reshape(x, s1), relax.TensorStructInfo(s1, "float32"))


def test_reshape_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4, 5)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    ns0 = relax.Var("ns", relax.ShapeStructInfo((3, 8, 5)))
    ns1 = relax.Var("ns", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.reshape(x0, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x0, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(bb, relax.op.reshape(x0, ns0), relax.TensorStructInfo(ns0, "float32"))
    _check_inference(bb, relax.op.reshape(x0, ns1), relax.TensorStructInfo(ns1, "float32"))
    _check_inference(
        bb, relax.op.reshape(x1, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(bb, relax.op.reshape(x1, ns0), relax.TensorStructInfo(ns0, "float32"))
    _check_inference(bb, relax.op.reshape(x1, ns1), relax.TensorStructInfo(ns1, "float32"))
    _check_inference(
        bb, relax.op.reshape(x2, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(bb, relax.op.reshape(x2, ns0), relax.TensorStructInfo(ns0, "float32"))
    _check_inference(bb, relax.op.reshape(x2, ns1), relax.TensorStructInfo(ns1, "float32"))


def test_reshape_infer_struct_into_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.reshape(x0, (120,)), relax.TensorStructInfo((120,), "float16"))
    _check_inference(bb, relax.op.reshape(x1, (120,)), relax.TensorStructInfo((120,), "int8"))


def test_reshape_infer_struct_info_unequal_shape_prod():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo((2, 3, 4, 5)))
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))
    ns = relax.Var("ns", relax.ShapeStructInfo((4, 4, 1, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (4, 4, 1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (4, 4, 1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (4, 4, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (4, 4, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, ns))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, ns))


def test_reshape_infer_struct_info_inference_not_deducible():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor("float32", ndim=4))
    x1 = relax.Var("x", R.Tensor("float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x3, (2, 3, -1)))


def test_reshape_new_shape_not_tuple():
    m = tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        relax.op.reshape(x, 120)
    with pytest.raises(TVMError):
        relax.op.reshape(x, m)


def test_reshape_infer_struct_info_new_shape_not_integer():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2.0, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, 3, -1.0)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, 3, 4.0, -1)))


def test_reshape_infer_struct_info_multiple_dim_inference():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, -1, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (-1, -1, -1, -1)))


def test_reshape_infer_struct_info_non_positive_new_shape():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, 0, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (-2, -3, -4, -5)))


def test_reshape_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    ns = relax.Var("ns", relax.TensorStructInfo((120,), "float32"))
    pv = relax.Var("pv", relax.PrimStructInfo("int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (2, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (2, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, ns))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, [pv]))


def test_permute_dims_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((1, 2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, axes=None), relax.TensorStructInfo((4, 3, 2, 1), "float32")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x0, [-2, -3, 3, -4]),
        relax.TensorStructInfo((3, 2, 4, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 1, 0]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, axes=None), relax.TensorStructInfo(dtype="float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x3, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), dtype="")
    )
    _check_inference(
        bb, relax.op.permute_dims(x3, axes=None), relax.TensorStructInfo((4, 3, 2, 1), dtype="")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x3, [-2, -3, 3, -4]),
        relax.TensorStructInfo((3, 2, 4, 1), dtype=""),
    )
    _check_inference(
        bb, relax.op.permute_dims(x4, [2, 3, 1, 0]), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x4, axes=None), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(bb, relax.op.permute_dims(x5, axes=None), relax.TensorStructInfo(dtype=""))


def test_permute_dims_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(
        bb, relax.op.permute_dims(x, [2, 3, 1, 0]), relax.TensorStructInfo((c, d, b, a), "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x, axes=None), relax.TensorStructInfo((d, c, b, a), "float32")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x, [-2, -3, 3, -4]),
        relax.TensorStructInfo((c, b, d, a), "float32"),
    )


def test_permute_dims_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1, 2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.permute_dims(x0, [0, 1, 2, 3]), relax.TensorStructInfo(s0, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, [-4, -3, -2, -1]), relax.TensorStructInfo(s0, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 0, 1]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [0, 1, 2, 3]), relax.TensorStructInfo(s1, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 0, 1]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, axes=None), relax.TensorStructInfo(dtype="float32")
    )


def test_permute_dims_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((1, 2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((1, 2, 3, 4), "int32"))

    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "float16")
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "int8")
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "int32")
    )


def test_permute_dims_infer_struct_info_unknown_ndim_with_axes():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor("float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [2, 3, 1, 0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [2, 3, 1, 0]))


def test_permute_dims_infer_struct_info_wrong_number_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1, 2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x2, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x2, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x3, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x3, [1, 2, 4, 0, 3]))


def test_permute_dims_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 3, 4, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, -5, 1, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 3, 4, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, -5, 1, 3]))


def test_permute_dims_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, -2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, -2, 1]))


def test_permute_dims_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((1, 2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((1, 2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1))


def test_expand_dims_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "float32")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x0, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((2, 1, 1, 1, 3, 1, 4, 1), "float32"),
    )
    _check_inference(bb, relax.op.expand_dims(x0, []), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.expand_dims(x1, []), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.expand_dims(x2, []), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.expand_dims(x3, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), dtype="")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x3, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((2, 1, 1, 1, 3, 1, 4, 1), dtype=""),
    )
    _check_inference(bb, relax.op.expand_dims(x3, []), relax.TensorStructInfo((2, 3, 4), dtype=""))
    _check_inference(bb, relax.op.expand_dims(x4, [1, 3]), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.expand_dims(x4, []), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.expand_dims(x5, [1, 3]), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.expand_dims(x5, []), relax.TensorStructInfo(dtype=""))


def test_expand_dims_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x = relax.Var("x", R.Tensor((a, 4, b), "float32"))

    _check_inference(
        bb, relax.op.expand_dims(x, [1, 3]), relax.TensorStructInfo((a, 1, 4, 1, b), "float32")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((a, 1, 1, 1, 4, 1, b, 1), "float32"),
    )
    _check_inference(bb, relax.op.expand_dims(x, []), relax.TensorStructInfo((a, 4, b), "float32"))


def test_expand_dims_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.expand_dims(x0, []), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.expand_dims(x1, []), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.expand_dims(x2, []), relax.TensorStructInfo(s2, "float32"))


def test_expand_dims_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "float16")
    )
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "int8")
    )
    _check_inference(
        bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "int32")
    )


def test_expand_dims_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", relax.TensorStructInfo(s0))
    x3 = relax.Var("x", relax.TensorStructInfo(s1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [-6, 1]))


def test_expand_dims_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", relax.TensorStructInfo(s0))
    x3 = relax.Var("x", relax.TensorStructInfo(s1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, -4]))


def test_expand_dims_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, axis=[]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, axis=[]))


def test_squeeze_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=6))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(
        bb, relax.op.squeeze(x0, [1, 4]), relax.TensorStructInfo((2, 3, 1, 4), "float32")
    )
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(
        bb, relax.op.squeeze(x1, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2, [1, 4]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.squeeze(x3, [1, 4]), relax.TensorStructInfo((2, 3, 1, 4), dtype="")
    )
    _check_inference(bb, relax.op.squeeze(x3), relax.TensorStructInfo((2, 3, 4), dtype=""))
    _check_inference(bb, relax.op.squeeze(x4, [1, 4]), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.squeeze(x4), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.squeeze(x5, [1, 4]), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.squeeze(x5), relax.TensorStructInfo(dtype=""))


def test_squeeze_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((a, 1, b), "float32"))
    x1 = relax.Var("x", R.Tensor((a, 1, b)))

    _check_inference(bb, relax.op.squeeze(x0, [1]), relax.TensorStructInfo((a, b), "float32"))
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x1, [1]), relax.TensorStructInfo((a, b), dtype=""))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(dtype=""))


def test_squeeze_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s2 = relax.Var("s", relax.ShapeStructInfo((a, 1, b)))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s4 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s4, "float32"))

    _check_inference(
        bb, relax.op.squeeze(x0, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x0, []), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x1, []), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(s1, dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2, [1]), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.squeeze(x2, []), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.squeeze(x3, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x3, []), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.squeeze(x3), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x4, [1, 4]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x4, []), relax.TensorStructInfo(s4, "float32"))
    _check_inference(bb, relax.op.squeeze(x4), relax.TensorStructInfo(dtype="float32"))


def test_squeeze_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "int32"))

    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo((2, 3, 4), "float16"))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo((2, 3, 4), "int8"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo((2, 3, 4), "int32"))


def test_squeeze_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [-7]))


def test_squeeze_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [1, 1]))


def test_squeeze_infer_struct_info_axis_length_not_one():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo((a, 3, 4)))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((a, 3, 4), "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [0]))


def test_squeeze_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1))


def test_flatten_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor((3,), "float32"))
    x2 = relax.Var("x", R.Tensor((), "float32"))
    x3 = relax.Var("x", R.Tensor("float32", ndim=3))
    x4 = relax.Var("x", R.Tensor("float32", ndim=1))
    x5 = relax.Var("x", R.Tensor("float32", ndim=0))
    x6 = relax.Var("x", R.Tensor("float32"))
    x7 = relax.Var("x", R.Tensor((3, 4, 5)))
    x8 = relax.Var("x", R.Tensor((3,)))
    x9 = relax.Var("x", R.Tensor(()))
    x10 = relax.Var("x", R.Tensor(ndim=3))
    x11 = relax.Var("x", R.Tensor(ndim=1))
    x12 = relax.Var("x", R.Tensor(ndim=0))
    x13 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((60,), "float32"))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((3,), "float32"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x4), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x5), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x6), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x7), relax.TensorStructInfo((60,), dtype=""))
    _check_inference(bb, relax.op.flatten(x8), relax.TensorStructInfo((3,), dtype=""))
    _check_inference(bb, relax.op.flatten(x9), relax.TensorStructInfo((1,), dtype=""))
    _check_inference(bb, relax.op.flatten(x10), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.flatten(x11), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.flatten(x12), relax.TensorStructInfo((1,), dtype=""))
    _check_inference(bb, relax.op.flatten(x13), relax.TensorStructInfo(dtype="", ndim=1))


def test_flatten_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((a, b), "float32"))
    x1 = relax.Var("x", R.Tensor((a, b)))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((a * b,), "float32"))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((a * b,), dtype=""))


def test_flatten_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((3, 4, 5)))
    s1 = relax.Var("s", relax.ShapeStructInfo((3,)))
    s2 = relax.Var("s", relax.ShapeStructInfo(()))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s4 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s5 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    s6 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s4, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s5, "float32"))
    x6 = relax.Var("x", relax.TensorStructInfo(s6, "float32"))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x4), relax.TensorStructInfo(s4, "float32"))
    _check_inference(bb, relax.op.flatten(x5), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x6), relax.TensorStructInfo(dtype="float32", ndim=1))


def test_flatten_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4, 5), "int32"))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((60,), "float16"))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((60,), "int8"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((60,), "int32"))


def test_flatten_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((3, 4, 5), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.flatten(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.flatten(x1))


def test_flatten_wrong_input_number():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TypeError):
        relax.op.flatten(x, y)


if __name__ == "__main__":
    tvm.testing.main()
