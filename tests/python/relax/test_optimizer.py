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

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm import relax as rx
from tvm import IRModule
from tvm.script._parser import ir as I, relax as R, tir as T
from tvm.ir.op import Op
from tvm.relax.training import SGD, MomentumSGD
from tvm.relay.testing import rand
from tvm.testing import assert_allclose
from tvm.runtime.container import tuple_object
from tvm.relax.transform import OperatorLegalizer

import tvm.relax.training.legalizer_update


def _execute_mod(mod, func_name, *args):
    lowered_mod = OperatorLegalizer(mod).transform()
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*args)


def _check_shape_equal(shape1, shape2):
    try:
        val = int(shape1)
        return val == int(shape2)
    except TypeError:
        assert len(shape1) == len(shape2)
        for x, y in zip(shape1, shape2):
            _check_shape_equal(x, y)


def _check_optimizer_shape(opt, args):
    shape = tuple(x.shape for x in args)
    f = opt.get_function()
    _check_shape_equal(f.params[0].shape, shape)
    _check_shape_equal(f.params[1].shape, shape)
    _check_shape_equal(f.body.body[0].shape, shape)


def test_shape():
    x = relax.Var("x", (3, 3), relax.DynTensorType(dtype="float32"))
    y = relax.Var("y", (2,), relax.DynTensorType(dtype="float32"))
    sgd1 = SGD(x, 0.01)
    sgd2 = SGD([x, y], 0.01)
    msgd = MomentumSGD([x, y], 0.01, 0.9, 0.1, 0.001, True)

    _check_optimizer_shape(sgd1, [x])
    _check_optimizer_shape(sgd2, [x, y])
    _check_optimizer_shape(msgd, [x, y])


def test_sgd_numeric():
    x = relax.Var("x", (3, 3), relax.DynTensorType(dtype="float32"))
    sgd = SGD(x, 0.01, 0.02)
    mod = IRModule.from_expr(sgd.get_function())

    x_np = np.random.rand(3, 3).astype(np.float32)
    x_ad_np = np.random.rand(3, 3).astype(np.float32)

    param = tuple_object([tvm.nd.array(x_np)])
    grad = tuple_object([tvm.nd.array(x_ad_np)])
    new_param, sgd.state = _execute_mod(mod, "SGD", param, grad, sgd.state)

    new_param_np = x_np - 0.01 * (x_ad_np + 0.02 * x_np)
    assert_allclose(new_param[0].numpy(), new_param_np)


def test_momentum_sgd_numeric():
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, False

    x = relax.Var("x", (3, 3), relax.DynTensorType(dtype="float32"))
    msgd = MomentumSGD(x, lr, mom, damp, wd, nest)
    mod = IRModule.from_expr(msgd.get_function())
    mod.show()

    x_np = np.random.rand(3, 3).astype(np.float32)
    x_ad_np = np.random.rand(3, 3).astype(np.float32)

    param = tuple_object([tvm.nd.array(x_np)])
    grad = tuple_object([tvm.nd.array(x_ad_np)])
    for _ in range(3):
        param, msgd.state = _execute_mod(mod, "MomentumSGD", param, grad, msgd.state)

    # numpy implementation
    b = np.zeros(x_np.shape).astype(np.float32)
    for _ in range(3):
        g = x_ad_np
        if wd:
            g = g + wd * x_np
        if mom:
            b = mom * b + (1 - damp) * g
            if nest:
                g = g + mom * b
            else:
                g = b
        x_np -= lr * g

    assert_allclose(param[0].numpy(), x_np)

if __name__ == "__main__":
    pytest.main([__file__])
