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
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    assert relax.op.sum(x).op == Op.get("relax.sum")
    assert relax.op.mean(x).op == Op.get("relax.mean")
    assert relax.op.variance(x).op == Op.get("relax.variance")
    assert relax.op.min(x).op == Op.get("relax.min")
    assert relax.op.max(x).op == Op.get("relax.max")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_reduction_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))

    _check_inference(bb, relax.op.sum(x0, axis=[1, 2]), relax.TensorStructInfo((2, 5), "float32"))
    _check_inference(
        bb,
        relax.op.sum(x0, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.sum(x0, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.sum(x0, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.mean(x1, axis=[1, 2]), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb,
        relax.op.mean(x1, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.mean(x1, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.mean(x1, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.variance(x2, axis=[1, 2]), relax.TensorStructInfo(dtype="float32")
    )
    _check_inference(
        bb,
        relax.op.variance(x2, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(bb, relax.op.variance(x2, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.variance(x2, axis=None, keepdims=True),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(bb, relax.op.max(x3, axis=[1, 2]), relax.TensorStructInfo((2, 5), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), dtype=""),
    )
    _check_inference(bb, relax.op.max(x3, axis=None), relax.TensorStructInfo((), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), dtype=""),
    )
    _check_inference(bb, relax.op.sum(x0, axis=[-1, -4]), relax.TensorStructInfo((3, 4), "float32"))
    _check_inference(bb, relax.op.sum(x0, axis=[]), relax.TensorStructInfo((2, 3, 4, 5), "float32"))


def test_reduction_infer_struct_info_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(bb, relax.op.min(x, axis=[1, 2]), relax.TensorStructInfo((a, d), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((a, 1, 1, d), "float32"),
    )
    _check_inference(bb, relax.op.min(x, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )


def test_reduction_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.sum(x0), relax.TensorStructInfo((), "float16"))
    _check_inference(bb, relax.op.sum(x1), relax.TensorStructInfo((), "int8"))


def test_reduction_infer_struct_info_axis_out_of_range_repetitive():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x1, axis=[3, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[-1, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x1, axis=[-4, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[-5]))


def test_dropout_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.variance(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.variance(x1))


if __name__ == "__main__":
    tvm.testing.main()
