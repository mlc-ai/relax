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
from typing import Callable, List  # must import to defer parsing of annotations

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm import relax as rx
from tvm import IRModule
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.ir.op import Op
from tvm.relax.training import SGD, MomentumSGD
from tvm.relay.testing import rand
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
    def np_func(x_arr, x_grad_arr, state_arr):
        num_steps = state_arr[0]
        state_arr[0] = num_steps + 1
        for i in range(len(x_arr)):
            x = x_arr[i]
            x_grad = x_grad_arr[i]
            x_arr[i] = x - lr * (x_grad + weight_decay * x)
        return x_arr, state_arr

    lr, weight_decay = 0.01, 0
    _test_optimizer(np_func, SGD, lr)
    lr, weight_decay = 0.01, 0.02
    _test_optimizer(np_func, SGD, lr, weight_decay)


def test_momentum_sgd():
    def np_func(x_arr, x_grad_arr, state_arr):
        num_steps = state_arr[0]
        state_arr[0] = num_steps + 1

        for i in range(len(x_arr)):
            x = x_arr[i]
            x_grad = x_grad_arr[i]
            x_velocity = state_arr[i + 1]
            dp = x * wd + x_grad
            x_velocity = mom * x_velocity + dp * (1 - damp)
            if nest:
                x = x - (dp + mom * x_velocity) * lr
            else:
                x = x - x_velocity * lr
            x_arr[i] = x
            state_arr[i + 1] = x_velocity

        return x_arr, state_arr

    lr, mom, damp, wd, nest = 0.01, 0.9, 0, 0, False
    _test_optimizer(np_func, MomentumSGD, lr, mom, damp, wd, nest)
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, False
    _test_optimizer(np_func, MomentumSGD, lr, mom, damp, wd, nest)
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, True
    _test_optimizer(np_func, MomentumSGD, lr, mom, damp, wd, nest)


if __name__ == "__main__":
    tvm.testing.main()
