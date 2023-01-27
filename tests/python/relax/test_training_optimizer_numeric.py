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
"""Numeric tests for relax optimizer APIs."""
from typing import Callable, List

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm import IRModule
from tvm.relax.training.optimizer import Adam, SGD, MomentumSGD
from tvm.script.parser import relax as R
from tvm.testing import assert_allclose
from tvm.runtime.container import tuple_object
from tvm.relax.transform import LegalizeOps


def _legalize_and_build(mod: IRModule):
    lowered_mod = LegalizeOps()(mod)
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm


def _numpy_to_tvm(data):
    if isinstance(data, (list, tuple)):
        return tuple_object([_numpy_to_tvm(_data) for _data in data])
    return tvm.nd.array(data)


def _tvm_to_numpy(data):
    if isinstance(data, (list, tuple, tvm.runtime.container.ADT)):
        return [_tvm_to_numpy(_data) for _data in data]
    return data.numpy()


def _assert_allclose_nested(data1, data2):
    if isinstance(data1, (list, tuple)):
        assert isinstance(data2, (list, tuple))
        assert len(data1) == len(data2)
        for x, y in zip(data1, data2):
            _assert_allclose_nested(x, y)
    else:
        assert_allclose(data1, data2)


def _assert_run_result_same(tvm_func: Callable, np_func: Callable, np_inputs: List):
    result = _tvm_to_numpy(tvm_func(*[_numpy_to_tvm(i) for i in np_inputs]))
    expected = np_func(*np_inputs)
    _assert_allclose_nested(result, expected)


def _test_optimizer(np_func, opt_type, *args, **kwargs):
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    opt = opt_type([x, y], *args, **kwargs)
    mod = IRModule.from_expr(opt.get_function())
    tvm_func = _legalize_and_build(mod)["main"]

    x_arr = [np.random.rand(3, 3).astype(np.float32), np.random.rand(3).astype(np.float32)]
    x_grad_arr = [np.random.rand(3, 3).astype(np.float32), np.random.rand(3).astype(np.float32)]
    state_arr = _tvm_to_numpy(opt.state)

    _assert_run_result_same(tvm_func, np_func, [x_arr, x_grad_arr, state_arr])


def test_sgd():
    def np_func(param_tuple, grad_tuple, state_tuple):
        num_steps = state_tuple[0]
        state_tuple[0] = num_steps + 1
        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            param_tuple[i] = param - lr * (grad + weight_decay * param)
        return param_tuple, state_tuple

    lr, weight_decay = 0.01, 0
    _test_optimizer(np_func, SGD, lr)
    lr, weight_decay = 0.01, 0.02
    _test_optimizer(np_func, SGD, lr, weight_decay)


def test_momentum_sgd():
    def np_func(param_tuple, grad_tuple, state_tuple):
        num_steps = state_tuple[0]
        state_tuple[0] = num_steps + 1

        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            velocity = state_tuple[i + 1]
            grad = param * weight_decay + grad
            velocity = momentum * velocity + grad * (1 - dampening)
            if nesterov:
                param = param - (grad + momentum * velocity) * lr
            else:
                param = param - velocity * lr
            param_tuple[i] = param
            state_tuple[i + 1] = velocity

        return param_tuple, state_tuple

    lr, momentum, dampening, weight_decay, nesterov = 0.01, 0.9, 0, 0, False
    _test_optimizer(np_func, MomentumSGD, lr, momentum, dampening, weight_decay, nesterov)
    lr, momentum, dampening, weight_decay, nesterov = 0.01, 0.9, 0.85, 0.02, False
    _test_optimizer(np_func, MomentumSGD, lr, momentum, dampening, weight_decay, nesterov)
    lr, momentum, dampening, weight_decay, nesterov = 0.01, 0.9, 0.85, 0.02, True
    _test_optimizer(np_func, MomentumSGD, lr, momentum, dampening, weight_decay, nesterov)


def test_adam():
    def np_func(param_tuple, grad_tuple, state_tuple):
        state_tuple[0] += 1
        state_tuple[1] *= betas[0]
        state_tuple[2] *= betas[1]
        num_steps = state_tuple[0]

        for i in range(len(param_tuple)):
            param = param_tuple[i]
            grad = grad_tuple[i]
            m = state_tuple[i + 3]
            v = state_tuple[i + 3 + len(param_tuple)]
            grad = grad + weight_decay * param
            m = betas[0] * m + (1 - betas[0]) * grad
            v = betas[1] * v + (1 - betas[1]) * grad * grad
            m_hat = m / (1 - betas[0] ** num_steps)
            v_hat = v / (1 - betas[1] ** num_steps)
            param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
            param_tuple[i] = param
            state_tuple[i + 3] = m
            state_tuple[i + 3 + len(param_tuple)] = v

        return param_tuple, state_tuple

    lr, betas, eps, weight_decay = 0.01, (0.9, 0.999), 1e-08, 0
    _test_optimizer(np_func, Adam, lr, betas, eps, weight_decay)
    lr, betas, eps, weight_decay = 0.01, (0.8, 0.85), 1e-07, 0.1
    _test_optimizer(np_func, Adam, lr, betas, eps, weight_decay)


if __name__ == "__main__":
    tvm.testing.main()
