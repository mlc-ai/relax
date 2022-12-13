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
import numpy as np
import tvm
from tvm import relay, relax
from tvm.error import DiagnosticError
from tvm.relax.testing import transform
from tvm.script import relax as R
import tvm.testing

target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)


def relay_build_and_run(f, inputs):
    mod = tvm.IRModule.from_expr(f)
    with target:
        graph_exec = relay.build_module.create_executor("graph", mod, dev, target).evaluate()
        return graph_exec(*inputs).numpy()


def relax_build_and_run(f, inputs):
    f = f.with_attr("global_symbol", "default")
    mod = tvm.IRModule.from_expr(f)

    with tvm.transform.PassContext(opt_level=3):
        mod = relax.transform.Normalize()(mod)
        mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
        ex = relax.vm.build(mod, target)
        vm = relax.VirtualMachine(ex, dev)
        return vm["default"](*inputs)


@pytest.mark.parametrize("op_name", ["relu", "softmax"])
def test_unary_ops(op_name: str):
    # Set up
    dtype = "float32"
    X_shape = (8, 8)

    data_x = tvm.nd.array(np.random.rand(*X_shape).astype(np.float32), dev)
    inputs = [data_x]

    # Build relay op, run and get the output
    X = relay.var("x", shape=X_shape, dtype=dtype)
    relay_nn = relay.nn
    relay_op = getattr(relay_nn, op_name)

    Z = relay_op(X)
    f = relay.Function([X], Z)
    expected = relay_build_and_run(f, inputs)

    # Relax output
    tensor_type = relax.DynTensorType(2, dtype)
    X = relax.Var("x", X_shape, tensor_type)
    relax_nn = relax.nn
    relax_op = getattr(relax_nn, op_name)
    Z = relax_op(X)
    f = relax.Function([X], Z, ret_type=tensor_type, ret_shape=relax.RuntimeDepShape())
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out.numpy(), expected)


def test_conv2d():
    dtype = "float32"
    kernel_size = 4
    data_shape = (1, 1, 256, 256)
    weight_shape = (1, 1, kernel_size, kernel_size)

    data = tvm.nd.array(np.random.rand(*data_shape).astype(np.float32), dev)
    weight = tvm.nd.array(np.random.rand(*weight_shape).astype(np.float32), dev)
    inputs = [data, weight]

    # Build relay op, run and get the output
    D = relay.var("data", shape=data_shape, dtype=dtype)
    W = relay.var("weight", shape=weight_shape, dtype=dtype)
    Z = relay.nn.conv2d(D, W)
    f = relay.Function([D, W], Z)
    expected = relay_build_and_run(f, inputs)

    # Relax output
    tensor_type = relax.DynTensorType(4, dtype)
    D = relax.Var("data", data_shape, tensor_type)
    W = relax.Var("weight", weight_shape, tensor_type)
    Z = relax.nn.conv2d(D, W, kernel_size=kernel_size)
    f = relax.Function([D, W], Z, ret_type=tensor_type, ret_shape=relax.RuntimeDepShape())
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out.numpy(), expected)


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

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_dense():
    # Set up
    dtype = "float32"
    X_shape = (4, 8)
    Y_shape = (4, 8)

    data_x = tvm.nd.array(np.random.rand(*X_shape).astype(np.float32), dev)
    data_y = tvm.nd.array(np.random.rand(*Y_shape).astype(np.float32), dev)
    inputs = [data_x, data_y]

    # Build relay op, run and get the output
    X = relay.var("x", shape=X_shape, dtype=dtype)
    Y = relay.var("y", shape=Y_shape, dtype=dtype)
    Z = relay.nn.dense(X, Y)
    f = relay.Function([X, Y], Z)
    expected = relay_build_and_run(f, inputs)

    # Relax output
    tensor_type = relax.DynTensorType(2, dtype)

    X = relax.Var("x", X_shape, tensor_type)
    Y = relax.Var("y", Y_shape, tensor_type)
    Z = relax.nn.dense(X, Y)
    f = relax.Function([X, Y], Z, ret_type=tensor_type, ret_shape=relax.RuntimeDepShape())
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out.numpy(), expected)


def test_max_pool2d():
    # Set up
    dtype = "float32"
    X_shape = (1, 1, 32, 32)
    pool_size = 3

    data_x = tvm.nd.array(np.random.rand(*X_shape).astype(np.float32), dev)
    inputs = [data_x]

    # Build relay op, run and get the output
    X = relay.var("x", shape=X_shape, dtype=dtype)
    Z = relay.nn.max_pool2d(X, pool_size=pool_size)
    f = relay.Function([X], Z)
    expected = relay_build_and_run(f, inputs)

    # Relax output
    tensor_type = relax.DynTensorType(4, dtype)

    X = relax.Var("x", X_shape, tensor_type)
    Z = relax.nn.max_pool2d(X, pool_size=pool_size)
    f = relax.Function([X], Z, ret_type=tensor_type, ret_shape=relax.RuntimeDepShape())
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out.numpy(), expected)


def test_add():
    dtype = "float32"
    a_shape = [15, 3, 5]
    b_shape = [15, 1, 5]

    tensor_type = relax.DynTensorType(ndim=3, dtype="float32")
    a = relax.Var("a", a_shape, tensor_type)
    b = relax.Var("b", b_shape, tensor_type)
    c = relax.op.add(a, b)
    f = relax.Function(
        params=[a, b], body=c, ret_type=tensor_type, ret_shape=relax.ShapeExpr(a_shape)
    )

    a_np = np.random.rand(*a_shape).astype(dtype)
    b_np = np.random.rand(*b_shape).astype(dtype)
    a_relax = tvm.nd.array(a_np, dev)
    b_relax = tvm.nd.array(b_np, dev)

    res_np = np.add(a_np, b_np)
    res_relax = relax_build_and_run(f, [a_relax, b_relax])

    tvm.testing.assert_allclose(res_relax.numpy(), res_np)


def test_subtract():
    dtype = "float32"
    a_shape = [15, 3, 5]
    b_shape = [1]

    tensor_type = relax.DynTensorType(ndim=3, dtype="float32")
    a = relax.Var("a", a_shape, tensor_type)
    b = relax.Var("b", b_shape, tensor_type)
    c = relax.op.subtract(a, b)
    f = relax.Function(
        params=[a, b], body=c, ret_type=tensor_type, ret_shape=relax.ShapeExpr(a_shape)
    )

    a_np = np.random.rand(*a_shape).astype(dtype)
    b_np = np.random.rand(*b_shape).astype(dtype)
    a_relax = tvm.nd.array(a_np, dev)
    b_relax = tvm.nd.array(b_np, dev)

    res_np = np.subtract(a_np, b_np)
    res_relax = relax_build_and_run(f, [a_relax, b_relax])

    tvm.testing.assert_allclose(res_relax.numpy(), res_np)


def test_multiply():
    dtype = "float32"
    a_shape = [15, 3, 5]
    b_shape = [3, 1]

    tensor_type = relax.DynTensorType(ndim=3, dtype="float32")
    a = relax.Var("a", a_shape, tensor_type)
    b = relax.Var("b", b_shape, tensor_type)
    c = relax.op.multiply(a, b)
    f = relax.Function(
        params=[a, b], body=c, ret_type=tensor_type, ret_shape=relax.ShapeExpr(a_shape)
    )

    a_np = np.random.rand(*a_shape).astype(dtype)
    b_np = np.random.rand(*b_shape).astype(dtype)
    a_relax = tvm.nd.array(a_np, dev)
    b_relax = tvm.nd.array(b_np, dev)

    res_np = np.multiply(a_np, b_np)
    res_relax = relax_build_and_run(f, [a_relax, b_relax])

    tvm.testing.assert_allclose(res_relax.numpy(), res_np)


def test_batch_norm():
    dtype = "float32"
    input_shape = [2, 4, 3, 3]
    param_shape = [4]

    tensor_type = relax.DynTensorType(ndim=4, dtype="float32")
    param_type = relax.DynTensorType(ndim=1, dtype="float32")

    eps = 1e-5
    x = relax.Var("x", input_shape, tensor_type)
    gamma = relax.Var("gamma", param_shape, param_type)
    beta = relax.Var("beta", param_shape, param_type)
    moving_mean = relax.Var("moving_mean", param_shape, param_type)
    moving_var = relax.Var("moving_var", param_shape, param_type)
    y = relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, epsilon=eps)

    f = relax.Function(
        params=[x, gamma, beta, moving_mean, moving_var],
        body=y,
        ret_type=relax.TupleType([tensor_type, param_type, param_type]),
        ret_shape=relax.ShapeExpr(input_shape),
    )

    x_np = np.random.rand(*input_shape).astype(dtype)
    gamma_np = np.random.rand(1, *param_shape, 1, 1).astype(dtype)
    beta_np = np.random.rand(1, *param_shape, 1, 1).astype(dtype)
    moving_mean_np = np.random.rand(1, *param_shape, 1, 1).astype(dtype)
    moving_var_np = np.random.rand(1, *param_shape, 1, 1).astype(dtype)
    x_relax = tvm.nd.array(x_np, dev)
    gamma_relax = tvm.nd.array(gamma_np.flatten(), dev)
    beta_relax = tvm.nd.array(beta_np.flatten(), dev)
    moving_mean_relax = tvm.nd.array(moving_mean_np.flatten(), dev)
    moving_var_relax = tvm.nd.array(moving_var_np.flatten(), dev)

    res_np = ((x_np - moving_mean_np) / np.sqrt(moving_var_np + eps)) * gamma_np + beta_np
    res_relax = relax_build_and_run(
        f, [x_relax, gamma_relax, beta_relax, moving_mean_relax, moving_var_relax]
    )

    tvm.testing.assert_allclose(res_relax[0].numpy(), res_np)
    tvm.testing.assert_allclose(res_relax[1].numpy(), moving_mean_np.flatten())
    tvm.testing.assert_allclose(res_relax[2].numpy(), moving_var_np.flatten())


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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
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

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    pytest.main([__file__])
