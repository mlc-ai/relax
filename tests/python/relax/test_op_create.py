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
    fill_value = relax.Var("fill_value", R.Tensor((), "float32"))
    assert relax.op.full(fill_value, (2, 3)).op == Op.get("relax.full")
    assert relax.op.full_like(x, fill_value).op == Op.get("relax.full_like")
    assert relax.op.ones((2, 3), "float32").op == Op.get("relax.ones")
    assert relax.op.ones_like(x).op == Op.get("relax.ones_like")
    assert relax.op.zeros((2, 3), "float32").op == Op.get("relax.zeros")
    assert relax.op.zeros_like(x).op == Op.get("relax.zeros_like")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_full_infer_struct_info():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.full(v0, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v0, (2, 3)), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full(v0, s0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v0, s0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full(v0, s1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(v0, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full(v0, s2, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(v0, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full(v0, s3, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(v0, s3), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.full(v1, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v1, (2, 3)), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full(v1, s0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v1, s0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full(v1, s1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(v1, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full(v1, s2, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(v1, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full(v1, s3, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(v1, s3), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.full(v2, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v2, (2, 3)), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(
        bb, relax.op.full(v2, s0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v2, s0), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full(v2, s1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(v2, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.full(v2, s2, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(v2, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.full(v2, s3, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(v2, s3), relax.TensorStructInfo(s3, dtype=""))
    _check_inference(
        bb, relax.op.full(v3, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v3, (2, 3)), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(
        bb, relax.op.full(v3, s0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v3, s0), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full(v3, s1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(v3, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.full(v3, s2, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(v3, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.full(v3, s3, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(v3, s3), relax.TensorStructInfo(s3, dtype=""))


def test_full_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))
    s0 = relax.ShapeExpr((a, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((a, 3)))

    _check_inference(
        bb, relax.op.full(v, (a, 3), "float16"), relax.TensorStructInfo((a, 3), "float16")
    )
    _check_inference(bb, relax.op.full(v, (a, 3)), relax.TensorStructInfo((a, 3), "float32"))
    _check_inference(bb, relax.op.full(v, s0, "float16"), relax.TensorStructInfo((a, 3), "float16"))
    _check_inference(bb, relax.op.full(v, s0), relax.TensorStructInfo((a, 3), "float32"))
    _check_inference(bb, relax.op.full(v, s1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(v, s1), relax.TensorStructInfo(s1, "float32"))


def test_full_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(()))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v1 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb, relax.op.full(v0, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full(v1, (2, 3), "float16"), relax.TensorStructInfo((2, 3), "float16")
    )


def test_full_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor((), "int8"))
    v2 = relax.Var("v", R.Tensor((), "int32"))

    _check_inference(
        bb, relax.op.full(v0, (2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.full(v0, (2, 3)), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(
        bb, relax.op.full(v1, (2, 3), "int32"), relax.TensorStructInfo((2, 3), "int32")
    )
    _check_inference(bb, relax.op.full(v1, (2, 3)), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full(v2, (2, 3), "int8"), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full(v2, (2, 3)), relax.TensorStructInfo((2, 3), "int32"))


def test_full_infer_struct_info_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v4 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))
    v5 = relax.Var("v", relax.TensorStructInfo(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v0, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v1, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v2, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v3, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v4, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v5, (2, 3)))


def test_full_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))
    v2 = relax.Var("v", relax.FuncStructInfo([], R.Tensor((), "float32")))
    s = relax.Var("s", R.Tensor((2, 3)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v0, s))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v1, (2, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(v2, (2, 3)))


def test_full_like_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor("float16", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v3), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full_like(x1, v0), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v2), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v3), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v3), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v2), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x4, v0), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v2), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v3), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x5, v0), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v2), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v3), relax.TensorStructInfo(dtype=""))


def test_full_like_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))
    v = relax.Var("v", R.Tensor((), "float16"))

    _check_inference(bb, relax.op.full_like(x0, v), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.full_like(x1, v), relax.TensorStructInfo((m, n), dtype=""))


def test_full_like_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", R.Tensor((2, 3), "float32"))
    sv0 = relax.Var("sv", relax.ShapeStructInfo(()))
    sv1 = relax.Var("sv", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", relax.TensorStructInfo(sv0, "float16"))
    v1 = relax.Var("v", relax.TensorStructInfo(sv1, "float16"))
    v2 = relax.Var("v", R.Tensor((), "float16"))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v2), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorStructInfo((2, 3), "float32"))


def test_full_like_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    v0 = relax.Var("v", R.Tensor((), "int32"))
    v1 = relax.Var("v", R.Tensor((), "float64"))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorStructInfo((2, 3), "int8"))


def test_full_like_infer_struct_info_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v4 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))
    v5 = relax.Var("v", relax.TensorStructInfo(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v5))


def test_full_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3)))
    v0 = relax.Var("v", R.Tensor(()))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x0, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x1, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x2, v1))


def test_ones_zeros_infer_struct_info():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.ones((2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.ones(s2, "float32"), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.ones(s3, "float32"), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.zeros((2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.zeros(s2, "float32"), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.zeros(s3, "float32"), relax.TensorStructInfo(s3, "float32"))


def test_ones_zeros_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    s0 = relax.ShapeExpr((m, n))
    s1 = relax.Var("s", relax.ShapeStructInfo((m, n)))

    _check_inference(
        bb, relax.op.ones((m, n), "float32"), relax.TensorStructInfo((m, n), "float32")
    )
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(
        bb, relax.op.zeros((m, n), "float32"), relax.TensorStructInfo((m, n), "float32")
    )
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorStructInfo(s1, "float32"))


def test_ones_zeros_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    _check_inference(bb, relax.op.ones(s0, "float16"), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.ones(s1, "int8"), relax.TensorStructInfo(s1, "int8"))
    _check_inference(bb, relax.op.zeros(s2, "int32"), relax.TensorStructInfo(s2, "int32"))
    _check_inference(bb, relax.op.zeros(s3, "float64"), relax.TensorStructInfo(s3, "float64"))


def test_ones_zeros_wrong_dtype():
    with pytest.raises(TypeError):
        relax.op.ones((2, 3))
    with pytest.raises(TVMError):
        relax.op.ones((2, 3), "")
    with pytest.raises(TypeError):
        relax.op.zeros((2, 3))
    with pytest.raises(TVMError):
        relax.op.zeros((2, 3), "")


def test_ones_zeros_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", R.Tensor((2, 3)))
    s1 = relax.Var("s", relax.FuncStructInfo([], R.Tensor((2, 3))))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.ones(s0, "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.zeros(s1, "float32"))


def test_ones_like_zeros_like_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.ones_like(x1), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ones_like(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.ones_like(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.ones_like(x4), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.ones_like(x5), relax.TensorStructInfo(dtype=""))


def test_ones_like_zeros_like_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo((m, n), dtype=""))


def test_ones_like_zeros_like_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.zeros_like(x2), relax.TensorStructInfo(s2, "float32"))


def test_ones_like_zeros_like_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo((2, 3), "int8"))


def test_ones_like_zeros_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.ones_like(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.zeros_like(x1))


if __name__ == "__main__":
    tvm.testing.main()
