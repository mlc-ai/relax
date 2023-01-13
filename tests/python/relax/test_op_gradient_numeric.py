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
    inputs_numpy: np.array,
    target: Union[str, tvm.target.Target],
    dev: tvm._ffi.runtime_ctypes.Device,
    output_shape: Union[Tuple, List[Tuple]],
    tuple_input: bool = False,
    **kwargs,  # attr for operators
):
    """Generate module and run it to check numberic gradients."""

    func_name = "main"

    # prepare input
    def _numpy_to_var(data, var_name):
        if isinstance(data, list):
            struct_infos = []
            for i in range(len(data)):
                tvm_var = _numpy_to_var(data[i], "")
                struct_infos.append(tvm_var.struct_info)
            return relax.Var(var_name, relax.TupleStructInfo(struct_infos))
        return relax.Var(var_name, relax.TensorStructInfo(data.shape, "float32"))

    def _numpy_to_tvm(data):
        if isinstance(data, list):
            ret_data = []
            for i in range(len(data)):
                tvm_data = _numpy_to_tvm(data[i])
                ret_data.append(tvm_data)
            return tvm.runtime.container.ADT(0, ret_data)
        return tvm.nd.array(data)

    def _tvm_to_numpy(data):
        if isinstance(data, tvm.runtime.container.ADT):
            return [_tvm_to_numpy(i) for i in data]
        return data.numpy()

    def _gen_weights(shape):
        if isinstance(shape, list):
            ret = []
            for s in shape:
                ret.append(_gen_weights(s))
            return ret
        else:
            return np.random.uniform(size=shape).astype(np.float32)

    param_vars = [_numpy_to_var(inputs_numpy[i], "x_" + str(i)) for i in range(len(inputs_numpy))]
    weights = _gen_weights(output_shape)
    grad_var = _numpy_to_var(weights, "grad")

    # get gradient
    op = Op.get(op_name)
    op_grad_func = op.get_attr("FPrimalGradient")
    if tuple_input:
        t = relax.Tuple(param_vars)
        call = op_func(t, **kwargs)
    else:
        call = op_func(*param_vars, **kwargs)

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
        inputs_tvm = [_numpy_to_tvm(i) for i in inputs]
        result = vm_0[func_name](*inputs_tvm)
        result_numpy = _tvm_to_numpy(result)
        if isinstance(result_numpy, list):
            assert isinstance(weights, list)
            assert len(weights) == len(result_numpy)
            ret = 0
            for i in range(len(weights)):
                ret += np.sum(weights[i] * result_numpy[i])
            return ret
        return np.sum(weights * result_numpy)

    grad_call = relax.Tuple(op_grad_func(call, grad_var))

    bb1 = relax.BlockBuilder()
    with bb1.function(func_name, param_vars + [grad_var]):
        with bb1.dataflow():
            if tuple_input:
                adjoints = bb1.emit(grad_call)
                out = bb1.emit_output(relax.TupleGetItem(adjoints, 0))
            else:
                out = bb1.emit_output(grad_call)
        bb1.emit_func_output(out)
    grad_mod = bb1.get()
    lower_grad_mod = OperatorLegalizer(grad_mod).transform()

    ex_1 = relax.vm.build(lower_grad_mod, target)
    vm_1 = relax.VirtualMachine(ex_1, dev)
    inputs_tvm = [_numpy_to_tvm(i) for i in inputs_numpy]
    weights_tvm = _numpy_to_tvm(weights)
    result = vm_1[func_name](*inputs_tvm, weights_tvm)

    check_numerical_grads(forward, inputs_numpy, _tvm_to_numpy(result))


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
    data1_numpy = np.random.randint(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.permute_dims, "relax.permute_dims", [data1_numpy], target, dev, (5, 4, 3, 2)
    )


@tvm.testing.parametrize_targets("llvm")
def test_permute_dims_with_axes(target, dev):
    data1_numpy = np.random.randint(0, 16, (2, 3, 4, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.permute_dims,
        "relax.permute_dims",
        [data1_numpy],
        target,
        dev,
        (2, 5, 3, 4),
        axes=(0, 3, 1, 2),
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


@tvm.testing.parametrize_targets("llvm")
def test_concat(target, dev):
    data_numpy1 = np.random.randint(1, 16, (3, 3)).astype(np.float32)
    data_numpy2 = np.random.randint(1, 16, (3, 4)).astype(np.float32)
    data_numpy3 = np.random.randint(1, 16, (3, 5)).astype(np.float32)
    relax_check_gradients(
        relax.op.concat,
        "relax.concat",
        [data_numpy1, data_numpy2, data_numpy3],
        target,
        dev,
        (3, 12),
        tuple_input=True,
        axis=1,
    )


@tvm.testing.parametrize_targets("llvm")
def test_split(target, dev):
    data_numpy = np.random.randint(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(
        relax.op.split,
        "relax.split",
        [data_numpy],
        target,
        dev,
        [(3, 3), (3, 4), (3, 5)],
        indices_or_sections=[3, 7],
        axis=1,
    )


if __name__ == "__main__":
    pytest.main([__file__])
