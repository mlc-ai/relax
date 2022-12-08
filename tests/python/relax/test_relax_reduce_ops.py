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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import tvm
from tvm import relax
from tvm.error import DiagnosticError
from tvm.script import relax as R
import tvm.testing


def test_sum():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((1, 3), "float32") = R.sum(x, axis=[1, 3])
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.sum(x, axis=[1, 3]))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sum_without_specified_axis():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=0):
        gv: R.Tensor((), "float32") = R.sum(x)
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.sum(x))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sum_keep_dims():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((1, 1, 3, 1), "float32") = R.sum(x, axis=[1, 3], keepdims=True)
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.sum(x, axis=[1, 3], keepdims=True))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sum_duplicate_axis_indices():
    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x]):
            gv = bb.emit(relax.op.reduce.sum(x, axis=[1, -3]))
            bb.emit_func_output(gv)


def test_sum_index_out_of_range():
    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x]):
            gv = bb.emit(relax.op.reduce.sum(x, axis=[1, -6]))
            bb.emit_func_output(gv)


def test_mean():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((1, 3), "float32") = R.mean(x, axis=[1, 3])
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.mean(x, axis=[1, 3]))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_variance():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=1):
        gv: R.Tensor((1,), "float32") = R.variance(x, axis=[-1, -2, -3])
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.variance(x, axis=[-1, -2, -3]))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_max():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((1, 1, 1, 1), "float32") = R.variance(x, axis=[-1, -2, -3], keepdims=True)
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.variance(x, axis=[-1, -2, -3], keepdims=True))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_min():
    @R.function
    def expected(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((1, 3, 4), "float32") = R.min(x, axis=1)
        return gv

    x = relax.Var("x", [1, 2, 3, 4], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.reduce.min(x, axis=1))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    test_sum()
    test_sum_without_specified_axis()
    test_sum_keep_dims()
    test_sum_duplicate_axis_indices()
    test_sum_index_out_of_range()
    test_mean()
    test_variance()
    test_max()
    test_min()
