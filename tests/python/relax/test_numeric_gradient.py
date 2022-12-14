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


def relax_check_gradients(op_func, op_name, input_numpy, target, dev, output_shape):
    """
        Generate module and run it to check numberic gradients.
    """

    func_name = "main"
    param_vars = []
    input_ndarray = []

    # prepare input
    for i in range(len(input_numpy)):
        param_vars.append(
            relax.Var("x_" + str(i), input_numpy[i].shape, relax.DynTensorType(ndim=len(input_numpy[i].shape), dtype="float32"))
        )
        input_ndarray.append(tvm.nd.array(input_numpy[i]))
    grad_var = relax.Var("grad", output_shape, relax.DynTensorType(ndim=len(output_shape), dtype="float32"))

    # get gradient
    op = Op.get(op_name)
    op_grad_func = op.get_attr("FPrimalGradient")
    call = op_func(*param_vars)
    grad_call = relax.Tuple(op_grad_func(call, grad_var))

    bb = relax.BlockBuilder()
    with bb.function(func_name, param_vars):
        with bb.dataflow():
            out = bb.emit_output(call)
        bb.emit_func_output(out)
    mod = bb.get()
    lower_mod = OperatorLegalizer(mod).transform()

    def forward(*inputs):
        ex_0 = relax.vm.build(lower_mod, target)
        vm_0 = relax.VirtualMachine(ex_0, dev)
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
    result = vm_1[func_name](*[tvm.nd.array(i) for i in input_numpy], tvm.nd.array(np.ones(output_shape).astype(np.float32)))

    check_numerical_grads(forward, input_numpy, [i.numpy() for i in result])


@tvm.testing.parametrize_targets("llvm")
def test_add(target, dev):
    data1_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.add, "relax.add", [data1_numpy, data2_numpy], target, dev, (16, 16))


@tvm.testing.parametrize_targets("llvm")
def test_subtract(target, dev):
    data1_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.subtract, "relax.subtract", [data1_numpy, data2_numpy], target, dev, (16, 16))


@tvm.testing.parametrize_targets("llvm")
def test_transpose(target, dev):
    data1_numpy = np.random.randint(0, 16, (5, 10)).astype(np.float32)
    relax_check_gradients(relax.op.transpose, "relax.transpose",  [data1_numpy], target, dev, (10, 5))


@tvm.testing.parametrize_targets("llvm")
def test_relu(target, dev):
    data1_numpy = np.random.uniform(-1, 1, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.nn.relu, "relax.nn.relu", [data1_numpy], target, dev, (16, 16))


@tvm.testing.parametrize_targets("llvm")
def test_matmul(target, dev):
    data1_numpy = np.random.randint(0, 16, (7, 8)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (8, 10)).astype(np.float32)
    relax_check_gradients(relax.op.nn.matmul, "relax.nn.matmul", [data1_numpy, data2_numpy], target, dev, (7, 10))


@tvm.testing.parametrize_targets("llvm")
def test_softmax(target, dev):
    data1_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.nn.softmax, "relax.nn.softmax",
        [data1_numpy], target, dev, (16, 16))


@tvm.testing.parametrize_targets("llvm")
def test_cross_entropy(target, dev):
    data1_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    relax_check_gradients(relax.op.nn.cross_entropy, "relax.nn.cross_entropy",
        [data1_numpy, data2_numpy], target, dev, ())


@tvm.testing.parametrize_targets("llvm")
def test_softmax_cross_entropy(target, dev):
    data1_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy /= np.sum(data2_numpy)
    relax_check_gradients(relax.op.nn.softmax_cross_entropy, "relax.nn.softmax_cross_entropy",
        [data1_numpy, data2_numpy], target, dev, ())


@tvm.testing.parametrize_targets("llvm")
def test_sigmoid(target, dev):
    data_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    relax_check_gradients(relax.op.sigmoid, "relax.sigmoid", [data_numpy], target, dev, (10,))


@tvm.testing.parametrize_targets("llvm")
def test_tanh(target, dev):
    data_numpy = np.random.randint(1, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.tanh, "relax.tanh", [data_numpy], target, dev, (16, 16))


if __name__ == "__main__":
    pytest.main([__file__])
