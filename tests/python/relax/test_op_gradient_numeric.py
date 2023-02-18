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
from typing import Callable, Union, Tuple, List

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.transform import LegalizeOps
from tvm.testing.utils import check_numerical_grads
from tvm.ir.op import Op


def relax_check_gradients(
    op_func: Callable,
    op_name: str,
    inputs_numpy: np.array,
    target: Union[str, tvm.target.Target],
    dev: tvm._ffi.runtime_ctypes.Device,
    output_shape: Union[Tuple, List[Tuple]],
    tuple_input: bool = False,
    ignore_grads: List[int] = [],
    **kwargs,  # attr for operators
):
    """Generate module and run it to check numberic gradients."""

    func_name = "main"

    # prepare input
    def _numpy_to_var(data, var_name):
        if isinstance(data, list):
            struct_infos = []
            for _data in data:
                tvm_var = _numpy_to_var(_data, "")
                struct_infos.append(tvm_var.struct_info)
            return relax.Var(var_name, relax.TupleStructInfo(struct_infos))
        return relax.Var(var_name, relax.TensorStructInfo(data.shape, str(data.dtype)))

    def _numpy_to_tvm(data):
        if isinstance(data, list):
            ret_data = []
            for _data in data:
                tvm_data = _numpy_to_tvm(_data)
                ret_data.append(tvm_data)
            return ret_data
        return tvm.nd.array(data)

    def _tvm_to_numpy(data):
        if isinstance(data, tvm.ir.Array):
            return [_tvm_to_numpy(i) for i in data]
        return data.numpy()

    def _gen_weights(shape, dtype):
        if isinstance(shape, list):
            ret = []
            for s in shape:
                ret.append(_gen_weights(s, dtype))
            return ret
        else:
            return np.random.uniform(1, 2, size=shape).astype(dtype)

    param_vars = [
        _numpy_to_var(input_numpy, "x_" + str(i)) for i, input_numpy in enumerate(inputs_numpy)
    ]
    weights = _gen_weights(output_shape, inputs_numpy[0].dtype)
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
    lower_mod = LegalizeOps()(mod)
    ex_0 = relax.vm.build(lower_mod, target)
    vm_0 = relax.VirtualMachine(ex_0, dev)

    def forward(*inputs):
        inputs_iter = iter(inputs)
        inputs_tvm = [
            _numpy_to_tvm(next(inputs_iter))
            if i not in ignore_grads
            else _numpy_to_tvm(inputs_numpy[i])
            for i in range(len(inputs_numpy))
        ]
        result = vm_0[func_name](*inputs_tvm)
        result_numpy = _tvm_to_numpy(result)
        if isinstance(result_numpy, list):
            assert isinstance(weights, list)
            assert len(weights) == len(result_numpy)
            ret = 0
            for i, weight in enumerate(weights):
                ret += np.sum(weight * result_numpy[i])
            return ret
        return np.sum(weights * result_numpy)

    bb1 = relax.BlockBuilder()
    with bb1.function(func_name, param_vars + [grad_var]):
        with bb1.dataflow():
            orig_var = bb1.emit(call)
            grad_call = relax.Tuple(op_grad_func(orig_var, call, grad_var, bb1))
            if tuple_input:
                adjoints = bb1.emit(grad_call)
                out = bb1.emit_output(relax.TupleGetItem(adjoints, 0))
            else:
                out = bb1.emit_output(grad_call)
        bb1.emit_func_output(out)
    grad_mod = bb1.get()
    lower_grad_mod = LegalizeOps()(grad_mod)

    ex_1 = relax.vm.build(lower_grad_mod, target)
    vm_1 = relax.VirtualMachine(ex_1, dev)
    inputs_tvm = [_numpy_to_tvm(i) for i in inputs_numpy]
    weights_tvm = _numpy_to_tvm(weights)
    result = _tvm_to_numpy(vm_1[func_name](*inputs_tvm, weights_tvm))
    result_filtered = [result[i] for i in range(len(result)) if i not in ignore_grads]

    check_numerical_grads(forward, inputs_numpy, result_filtered)


##################### Binary #####################

binary_arith_op_func, binary_arith_op_name = tvm.testing.parameters(
    (relax.op.add, "relax.add"),
    (relax.op.subtract, "relax.subtract"),
    (relax.op.multiply, "relax.multiply"),
    (relax.op.divide, "relax.divide"),
)


@tvm.testing.parametrize_targets("llvm")
def test_binary_arith(target, dev, binary_arith_op_func, binary_arith_op_name):
    data1_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    data2_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    relax_check_gradients(
        binary_arith_op_func, binary_arith_op_name, [data1_numpy, data2_numpy], target, dev, (3, 3)
    )


binary_cmp_op_func, binary_cmp_op_name = tvm.testing.parameters(
    (relax.op.equal, "relax.equal"),
    (relax.op.greater, "relax.greater"),
    (relax.op.greater_equal, "relax.greater_equal"),
    (relax.op.less, "relax.less"),
    (relax.op.less_equal, "relax.less_equal"),
    (relax.op.not_equal, "relax.not_equal"),
)


@tvm.testing.parametrize_targets("llvm")
def test_binary_cmp(target, dev, binary_cmp_op_func, binary_cmp_op_name):
    # We must assure data1_numpy[i] != data2_numpy[i] for all possible i-s
    # If data1_numpy[i] == data2_numpy[i], the operator is not differentiable w.r.t. place i
    data1_numpy = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    delta = np.random.uniform(1, 2, (3, 3)).astype(np.float32)
    sign = np.random.randint(0, 2, (3, 3)).astype(np.float32) * 2 - 1
    data2_numpy = data1_numpy + delta * sign
    relax_check_gradients(
        binary_cmp_op_func, binary_cmp_op_name, [data1_numpy, data2_numpy], target, dev, (3, 3)
    )


##################### Unary #####################

unary_op_func, unary_op_name, can_be_neg = tvm.testing.parameters(
    (relax.op.abs, "relax.abs", True),
    (relax.op.cos, "relax.cos", True),
    (relax.op.exp, "relax.exp", True),
    (relax.op.log, "relax.log", False),
    (relax.op.negative, "relax.negative", True),
    (relax.op.sigmoid, "relax.sigmoid", True),
    (relax.op.sin, "relax.sin", True),
    (relax.op.sqrt, "relax.sqrt", False),
    (relax.op.tanh, "relax.tanh", True),
)


@tvm.testing.parametrize_targets("llvm")
def test_unary(target, dev, unary_op_func, unary_op_name, can_be_neg):
    (low, high) = (-1, 1) if can_be_neg else (0.1, 1)
    data_numpy = np.random.uniform(low, high, (3, 3)).astype(np.float32)
    relax_check_gradients(unary_op_func, unary_op_name, [data_numpy], target, dev, (3, 3))


##################### Create #####################


like_op_func, like_op_name, is_full = tvm.testing.parameters(
    (relax.op.zeros_like, "relax.zeros_like", False),
    (relax.op.ones_like, "relax.ones_like", False),
    (relax.op.full_like, "relax.full_like", True),
)


@tvm.testing.parametrize_targets("llvm")
def test_create_like(target, dev, like_op_func, like_op_name, is_full):
    data_numpy = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)
    if is_full:
        full_value = np.random.uniform(-1, 1, ()).astype(np.float32)
        relax_check_gradients(
            like_op_func, like_op_name, [data_numpy, full_value], target, dev, (3, 3)
        )
    else:
        relax_check_gradients(like_op_func, like_op_name, [data_numpy], target, dev, (3, 3))


##################### Statistical #####################


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


##################### Manipulate #####################


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
def test_split_indices(target, dev):
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


@tvm.testing.parametrize_targets("llvm")
def test_split_section(target, dev):
    data_numpy = np.random.randint(1, 16, (3, 12)).astype(np.float32)
    relax_check_gradients(
        relax.op.split,
        "relax.split",
        [data_numpy],
        target,
        dev,
        [(3, 4), (3, 4), (3, 4)],
        indices_or_sections=3,
        axis=1,
    )


##################### Linear Algebra #####################


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


##################### Neural network #####################


@tvm.testing.parametrize_targets("llvm")
def test_relu(target, dev):
    data1_numpy = np.random.uniform(0.2, 1, (3, 3)).astype(np.float32)
    sign = np.random.randint(0, 2, (3, 3)).astype(np.float32) * 2 - 1
    data1_numpy *= sign
    relax_check_gradients(relax.op.nn.relu, "relax.nn.relu", [data1_numpy], target, dev, (3, 3))


@tvm.testing.parametrize_targets("llvm")
def test_softmax(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.softmax, "relax.nn.softmax", [data1_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_softmax_with_axis(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.softmax, "relax.nn.softmax", [data1_numpy], target, dev, (3, 3), axis=1
    )


@tvm.testing.parametrize_targets("llvm")
def test_log_softmax(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.log_softmax, "relax.nn.log_softmax", [data1_numpy], target, dev, (3, 3)
    )


@tvm.testing.parametrize_targets("llvm")
def test_log_softmax_with_axis(target, dev):
    data1_numpy = np.random.randint(0, 16, (3, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.log_softmax, "relax.nn.log_softmax", [data1_numpy], target, dev, (3, 3), axis=1
    )


@tvm.testing.parametrize_targets("llvm")
def test_cross_entropy_with_logits(target, dev):
    data_numpy1 = np.random.randint(1, 16, (3,)).astype(np.float32)
    data_numpy2 = np.random.randint(1, 16, (3,)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.cross_entropy_with_logits,
        "relax.nn.cross_entropy_with_logits",
        [data_numpy1, data_numpy2],
        target,
        dev,
        (),
    )


@tvm.testing.parametrize_targets("llvm")
def test_cross_entropy_with_logits_batch(target, dev):
    data_numpy1 = np.random.randint(1, 16, (2, 3)).astype(np.float32)
    data_numpy2 = np.random.randint(1, 16, (2, 3)).astype(np.float32)
    relax_check_gradients(
        relax.op.nn.cross_entropy_with_logits,
        "relax.nn.cross_entropy_with_logits",
        [data_numpy1, data_numpy2],
        target,
        dev,
        (),
    )


if __name__ == "__main__":
    tvm.testing.main()
