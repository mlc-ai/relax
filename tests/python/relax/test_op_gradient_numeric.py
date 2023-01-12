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
import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.transform.op_legalizer import OperatorLegalizer
from tvm.testing.utils import check_numerical_grads
from tvm.ir.op import Op
import tvm.relax.op.gradient
from typing import Callable, Union, Tuple, List


def relax_check_gradients(
    op_func: Callable,
    op_name: str,
    input_numpy: np.array,
    target: Union[str, tvm.target.Target],
    dev: tvm._ffi.runtime_ctypes.Device,
    output_shape: Union[Tuple, List],
    **kwargs,  # attr for operators
):
    """Generate module and run it to check numberic gradients."""

    func_name = "main"
    param_vars = []
    input_ndarray = []

    # prepare input
    for i in range(len(input_numpy)):
        param_vars.append(
            relax.Var(
                "x_" + str(i),
                relax.TensorStructInfo(input_numpy[i].shape, "float32"),
            )
        )
        input_ndarray.append(tvm.nd.array(input_numpy[i]))
    grad_var = relax.Var("grad", relax.TensorStructInfo(output_shape, "float32"))

    # get gradient
    op = Op.get(op_name)
    op_grad_func = op.get_attr("FPrimalGradient")
    call = op_func(*param_vars, **kwargs)
    grad_call = relax.Tuple(op_grad_func(call, grad_var))

    bb = relax.BlockBuilder()
    with bb.function(func_name, param_vars):
        with bb.dataflow():
            out = bb.emit_output(call)
        bb.emit_func_output(out)
    mod = bb.get()
    lower_mod = OperatorLegalizer(mod).transform()
    ex_0 = relax.vm.build(lower_mod, target)
    vm_0 = relax.VirtualMachine(ex_0, dev)

    def forward(*inputs):
        result = vm_0[func_name](*[tvm.nd.array(i) for i in inputs])
        return np.sum(result.numpy())

    bb1 = relax.BlockBuilder()
    with bb1.function(func_name, param_vars + [grad_var]):
        with bb1.dataflow():
            out = bb1.emit_output(grad_call)
        bb1.emit_func_output(out)
    grad_mod = bb1.get()
    lower_grad_mod = OperatorLegalizer(grad_mod).transform()

    ex_1 = relax.vm.build(lower_grad_mod, target)
    vm_1 = relax.VirtualMachine(ex_1, dev)
    result = vm_1[func_name](
        *[tvm.nd.array(i) for i in input_numpy],
        tvm.nd.array(np.ones(output_shape).astype(np.float32)),
    )

    check_numerical_grads(forward, input_numpy, [i.numpy() for i in result])


@tvm.testing.parametrize_targets("llvm")
def test_add(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.add, "relax.add", [data1_numpy, data2_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_subtract(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.subtract, "relax.subtract", [data1_numpy, data2_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_multiply(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.multiply, "relax.multiply", [data1_numpy, data2_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_permute_dims(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.permute_dims, "relax.permute_dims", [data1_numpy], target, dev, (4, 3, 2)
    )


@tvm.testing.parametrize_targets("llvm")
def test_permute_dims_with_axes(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.permute_dims,
        "relax.permute_dims",
        [data1_numpy],
        target,
        dev,
        (2, 4, 3),
        axes=(0, 2, 1),
    )


@tvm.testing.parametrize_targets("llvm")
def test_relu(target, dev):
    data1_numpy = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.nn.relu, "relax.nn.relu", [data1_numpy], target, dev, (3, 3))


@tvm.testing.parametrize_targets("llvm")
def test_matmul_2_2(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul, "relax.matmul", [data1_numpy, data2_numpy], target, dev, (2, 4)
    )


@tvm.testing.parametrize_targets("llvm")
def test_matmul_1_1(target, dev):
    data1_numpy = np.random.randint(0, 16, (4,)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (4,)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul, "relax.matmul", [data1_numpy, data2_numpy], target, dev, ()
    )


@tvm.testing.parametrize_targets("llvm")
def test_matmul_1_4(target, dev):
    data1_numpy = np.random.randint(0, 16, (4,)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul, "relax.matmul", [data1_numpy, data2_numpy], target, dev, (2, 3, 5)
    )


@tvm.testing.parametrize_targets("llvm")
def test_matmul_4_1(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 4, 5)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (5,)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul, "relax.matmul", [data1_numpy, data2_numpy], target, dev, (2, 3, 4)
    )


@tvm.testing.parametrize_targets("llvm")
def test_matmul_5_4(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 1, 4, 5)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 2, 5, 4)).astype(np.float32)
    relax_check_gradients(
        relax.op.matmul,
        "relax.matmul",
        [data1_numpy, data2_numpy],
        target,
        dev,
        (2, 3, 2, 4, 4),
    )


@tvm.testing.parametrize_targets("llvm")
def test_softmax(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.softmax, "relax.nn.softmax", [data1_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_sum(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.sum, "relax.sum", [data1_numpy], target, dev, ())


@tvm.testing.parametrize_targets("llvm")
def test_sum_with_axis(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.sum, "relax.sum", [data1_numpy], target, dev, (2, 4), axis=[1, 3]
    )


@tvm.testing.parametrize_targets("llvm")
def test_sum_keepdims(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.sum, "relax.sum", [data1_numpy], target, dev, (3, 1), keepdims=True, axis=1
    )


@tvm.testing.parametrize_targets("llvm")
def test_softmax_with_axis(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.softmax, "relax.nn.softmax", [data1_numpy], target, dev, (3, 3), axis=1
    )


@tvm.testing.parametrize_targets("llvm")
def test_sigmoid(target, dev):
    data_numpy = np.random.randint(1, 16, (3,)).astype(np.float32)
    relax_check_gradients(relax.op.sigmoid, "relax.sigmoid", [data_numpy], target, dev, (3,))


@tvm.testing.parametrize_targets("llvm")
def test_tanh(target, dev):
    data_numpy = np.random.randint(1, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(relax.op.tanh, "relax.tanh", [data_numpy], target, dev, (3, 3))


if __name__ == "__main__":
    pytest.main([__file__])
