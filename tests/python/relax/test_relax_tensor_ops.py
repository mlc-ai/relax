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
from tvm.relax.testing import transform
from tvm.script import relax as R, tir as T
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
        return vm["default"](*inputs).numpy()


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
    f = relax.Function([X], Z, ret_type=tensor_type)
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out, expected)


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
    f = relax.Function([D, W], Z, ret_type=tensor_type)
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out, expected)


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
    f = relax.Function([X, Y], Z, ret_type=tensor_type)
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out, expected)


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
    f = relax.Function([X], Z, ret_type=tensor_type)
    out = relax_build_and_run(f, inputs)

    tvm.testing.assert_allclose(out, expected)


if __name__ == "__main__":
    pytest.main([__file__])
