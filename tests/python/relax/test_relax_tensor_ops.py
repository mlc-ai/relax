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
from tvm import relax
from tvm.error import DiagnosticError
from tvm.script import relax as R


def test_conv2d():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
    ) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, kernel_size=[3, 3])
        return gv

    x = relax.Var("x", [2, 3, 28, 28], relax.DynTensorType(ndim=4, dtype="float32"))
    w = relax.Var("w", [4, 3, 3, 3], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, w]):
        gv = bb.emit(relax.nn.conv2d(x, w, kernel_size=[3, 3]))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_conv2d_with_out_dtype():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 228, 228), "float32"), w: R.Tensor((16, 3, 5, 5), "float32")
    ) -> R.Tensor(None, "float16", ndim=4):
        gv: R.Tensor((2, 16, 224, 224), "float16") = R.nn.conv2d(
            x, w, kernel_size=(5, 5), out_dtype="float16"
        )
        return gv

    x = relax.Var("x", [2, 3, 228, 228], relax.DynTensorType(ndim=4, dtype="float32"))
    w = relax.Var("w", [16, 3, 5, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, w]):
        gv = bb.emit(relax.op.nn.conv2d(x, w, kernel_size=(5, 5), out_dtype="float16"))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_dense():
    @R.function
    def expected(
        x: R.Tensor((4, 8), "float32"), y: R.Tensor((4, 8), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((4, 4), "float32") = R.nn.dense(x, y)
        return gv

    x = relax.Var("x", [4, 8], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [4, 8], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.dense(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_max_pool2d():
    @R.function
    def expected(
        x: R.Tensor((1, 1, 32, 32), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        gv: R.Tensor((1, 1, 30, 30), dtype="float32") = R.nn.max_pool2d(x, pool_size=3)
        return gv

    x = relax.Var("x", [1, 1, 32, 32], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.nn.max_pool2d(x, pool_size=3))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_relu():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.nn.relu(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.nn.relu(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_softmax():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.nn.softmax(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.nn.softmax(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_add():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.add(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.add(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_subtract():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.subtract(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.subtract(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_multiply():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.multiply(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.multiply(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_batch_norm():
    @R.function
    def expected(
        x: R.Tensor((2, 4, 3, 3), dtype="float32"),
        gamma: R.Tensor((4,), dtype="float32"),
        beta: R.Tensor((4,), dtype="float32"),
        moving_mean: R.Tensor((4,), dtype="float32"),
        moving_var: R.Tensor((4,), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor(None, dtype="float32", ndim=4),
        R.Tensor(None, dtype="float32", ndim=1),
        R.Tensor(None, dtype="float32", ndim=1),
    ):
        gv: R.Tuple(
            R.Tensor((2, 4, 3, 3), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
        return gv

    x = relax.Var("x", [2, 4, 3, 3], relax.DynTensorType(ndim=4, dtype="float32"))
    gamma = relax.Var("gamma", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    beta = relax.Var("beta", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    moving_mean = relax.Var("moving_mean", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    moving_var = relax.Var("moving_var", [4], relax.DynTensorType(ndim=1, dtype="float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x, gamma, beta, moving_mean, moving_var]):
        gv = bb.emit(relax.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_gelu():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.nn.gelu(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.nn.gelu(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_silu():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.nn.silu(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.nn.silu(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sin():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.sin(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sin(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_cos():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.cos(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.cos(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_log():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.log(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.log(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_tanh():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.tanh(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.tanh(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_negative():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.negative(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.negative(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_floor_divide():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.floor_divide(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.floor_divide(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_divide():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.divide(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.divide(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_dropout():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")):
        gv = R.nn.dropout(x, rate=0.5)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.nn.dropout(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_layer_norm():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 4, 5), "float32"),
        gamma: R.Tensor((4, 5), "float32"),
        beta: R.Tensor((4, 5), "float32"),
    ) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.layer_norm(x, gamma, beta, axis=[-2, -1])
        return gv

    x = relax.Var("x", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    gamma = relax.Var("gamma", [4, 5], relax.DynTensorType(ndim=2, dtype="float32"))
    beta = relax.Var("beta", [4, 5], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, gamma, beta]):
        gv = bb.emit(relax.op.nn.layer_norm(x, gamma, beta, axis=[-2, -1]))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sqrt():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.sqrt(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sqrt(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_2_2():
    @R.function
    def expected(
        x: R.Tensor((3, 4), "float32"), y: R.Tensor((4, 5), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((3, 5), "float32") = R.nn.matmul(x, y)
        return gv

    x = relax.Var("x", [3, 4], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [4, 5], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_1_1():
    @R.function
    def expected(
        x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")
    ) -> R.Tensor(None, "float32", ndim=0):
        gv: R.Tensor((), "float32") = R.nn.matmul(x, y)
        return gv

    x = relax.Var("x", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    y = relax.Var("y", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_1_4():
    @R.function
    def expected(
        x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")
    ) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((2, 3, 5), "float32") = R.nn.matmul(x, y)
        return gv

    x = relax.Var("x", [4], relax.DynTensorType(ndim=1, dtype="float32"))
    y = relax.Var("y", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_4_1():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")
    ) -> R.Tensor(None, "float32", ndim=3):
        gv: R.Tensor((2, 3, 4), "float32") = R.nn.matmul(x, y)
        return gv

    x = relax.Var("x", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    y = relax.Var("y", [5], relax.DynTensorType(ndim=1, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_4_5():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
    ) -> R.Tensor(None, "float32", ndim=5):
        gv: R.Tensor((6, 2, 3, 4, 7), "float32") = R.nn.matmul(x, y)
        return gv

    x = relax.Var("x", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    y = relax.Var("y", [6, 2, 3, 5, 7], relax.DynTensorType(ndim=5, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_4_5_with_output_dtype():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
    ) -> R.Tensor(None, "float16", ndim=5):
        gv: R.Tensor((6, 2, 3, 4, 7), "float16") = R.nn.matmul(x, y, out_dtype="float16")
        return gv

    x = relax.Var("x", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    y = relax.Var("y", [6, 2, 3, 5, 7], relax.DynTensorType(ndim=5, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.nn.matmul(x, y, out_dtype="float16"))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_matmul_fail_on_incompatible_last_two_dims():
    x = relax.Var("x", [3, 4, 5], relax.DynTensorType(ndim=3, dtype="float32"))
    y = relax.Var("y", [3, 4, 5], relax.DynTensorType(ndim=3, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x, y]):
            gv = bb.emit(relax.op.nn.matmul(x, y))
            bb.emit_func_output(gv)


def test_matmul_fail_on_not_broadcastable():
    x = relax.Var("x", [2, 3, 4, 5], relax.DynTensorType(ndim=4, dtype="float32"))
    y = relax.Var("y", [2, 8, 3, 5, 6], relax.DynTensorType(ndim=5, dtype="float32"))
    bb = relax.BlockBuilder()
    with pytest.raises(DiagnosticError):
        with bb.function("main", [x, y]):
            gv = bb.emit(relax.op.nn.matmul(x, y))
            bb.emit_func_output(gv)


def test_adaptive_avg_pool2d():
    @R.function
    def expected(x: R.Tensor((2, 64, 8, 9), "float32")) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((2, 64, 7, 7), "float32") = R.nn.adaptive_avg_pool2d(x, output_size=[7, 7])
        return gv

    x = relax.Var("x", [2, 64, 8, 9], relax.DynTensorType(ndim=4, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.nn.adaptive_avg_pool2d(x, output_size=(7, 7)))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sigmoid():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv: R.Tensor((2, 3), "float32") = R.sigmoid(x)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sigmoid(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_less():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor(None, "bool", ndim=2):
        gv: R.Tensor((2, 3), "bool") = R.less(x, y)
        return gv

    x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    y = relax.Var("y", [2, 1], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.less(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_cross_entropy():
    @R.function
    def expected(
        predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")
    ) -> R.Tensor(None, "float32", ndim=0):
        gv: R.Tensor((), "float32") = R.nn.cross_entropy(predictions, targets)
        return gv

    predictions = relax.Var("predictions", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    targets = relax.Var("targets", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [predictions, targets]):
        gv = bb.emit(relax.op.nn.cross_entropy(predictions, targets))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_softmax_cross_entropy():
    @R.function
    def expected(
        predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")
    ) -> R.Tensor(None, "float32", ndim=0):
        gv: R.Tensor((), "float32") = R.nn.softmax_cross_entropy(predictions, targets)
        return gv

    predictions = relax.Var("predictions", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    targets = relax.Var("targets", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [predictions, targets]):
        gv = bb.emit(relax.op.nn.softmax_cross_entropy(predictions, targets))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    pytest.main([__file__])
