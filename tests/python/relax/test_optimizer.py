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
"""Unit tests for relax optimizer APIs."""
import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.relax.training.optimizer import SGD, MomentumSGD, Adam
from tvm.script.parser import relax as R


def test_optimizer_wrong_param():
    x1 = relax.Var("x1", R.Tensor((3, 3), "float32"))
    x2 = relax.Var("x1", R.Tensor((3, 3), "bfloat16"))
    x3 = relax.Var("x2", R.Tuple([R.Tensor((3, 3), "float32")]))
    x4 = relax.Var("x3", R.Tensor((3, 3), "int64"))
    x5 = relax.Tuple([x1])

    SGD(x1, 0.01) # fine
    SGD([x1], 0.01) # fine
    assert SGD([x2], 0.01)._dtype == "bfloat16"

    with pytest.raises(AssertionError, match="Not every parameter is Var."):
        SGD(x5, 0.01)
    with pytest.raises(AssertionError, match="Not every parameter is Tensor Var"):
        SGD(x3, 0.01)
    with pytest.raises(AssertionError, match="Pamameters must be of float dtype"):
        SGD(x4, 0.01)
    with pytest.raises(AssertionError, match="All parameters should have the same dtype"):
        SGD([x1, x2], 0.01)


def test_sgd_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    sgd = SGD([x, y], 0.01).get_function()

    # fmt: off
    @R.function
    def sgd_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv6: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv2)
            lv7: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv6)
            lv8: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv3)
            lv9: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv8)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv7, lv9)
            gv1: R.Tuple(R.Tensor((), dtype="int64")) = (lv5,)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(sgd, sgd_expected)


def test_sgd_complex():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    sgd = SGD([x, y], 0.01, 0.02).get_function()

    # fmt: off
    @R.function
    def sgd_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv6: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv)
            lv7: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv2)
            lv8: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv7)
            lv9: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv8)
            lv10: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv1)
            lv11: R.Tensor((3,), dtype="float32") = R.add(lv10, lv3)
            lv12: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv11)
            lv13: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv12)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv9, lv13)
            gv1: R.Tuple(R.Tensor((), dtype="int64")) = (lv5,)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(sgd, sgd_expected)


def test_momentum_sgd_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD([x, y], 0.01, 0.9).get_function()

    # fmt: off
    @R.function
    def msgd_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((3, 3), dtype="float32") = optim_states[1]
            lv6: R.Tensor((3,), dtype="float32") = optim_states[2]
            lv7: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv8: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv5)
            lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv2)
            lv10: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv9)
            lv11: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv10)
            lv12: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv6)
            lv13: R.Tensor((3,), dtype="float32") = R.add(lv12, lv3)
            lv14: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv13)
            lv15: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv14)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv11, lv15)
            gv1: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv7, lv9, lv13)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(msgd, msgd_expected)


def test_momentum_sgd_complex():
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, False

    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD([x, y], lr, mom, damp, wd, nest).get_function()

    # fmt: off
    @R.function
    def msgd_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((3, 3), dtype="float32") = optim_states[1]
            lv6: R.Tensor((3,), dtype="float32") = optim_states[2]
            lv7: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv8: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv)
            lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv2)
            lv10: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv5)
            lv11: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv9)
            lv12: R.Tensor((3, 3), dtype="float32") = R.add(lv10, lv11)
            lv13: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv12)
            lv14: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv13)
            lv15: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv1)
            lv16: R.Tensor((3,), dtype="float32") = R.add(lv15, lv3)
            lv17: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv6)
            lv18: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv16)
            lv19: R.Tensor((3,), dtype="float32") = R.add(lv17, lv18)
            lv20: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv19)
            lv21: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv20)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv14, lv21)
            gv1: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv7, lv12, lv19)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(msgd, msgd_expected)


def test_momentum_sgd_nesterov():
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, True

    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD([x, y], lr, mom, damp, wd, nest).get_function()

    # fmt: off
    @R.function
    def msgd_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((3, 3), dtype="float32") = optim_states[1]
            lv6: R.Tensor((3,), dtype="float32") = optim_states[2]
            lv7: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv8: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv)
            lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv2)
            lv10: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv5)
            lv11: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv9)
            lv12: R.Tensor((3, 3), dtype="float32") = R.add(lv10, lv11)
            lv13: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv12)
            lv14: R.Tensor((3, 3), dtype="float32") = R.add(lv9, lv13)
            lv15: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv14)
            lv16: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv15)
            lv17: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.02, "float32"), lv1)
            lv18: R.Tensor((3,), dtype="float32") = R.add(lv17, lv3)
            lv19: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv6)
            lv20: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv18)
            lv21: R.Tensor((3,), dtype="float32") = R.add(lv19, lv20)
            lv22: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv21)
            lv23: R.Tensor((3,), dtype="float32") = R.add(lv18, lv22)
            lv24: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv23)
            lv25: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv24)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv16, lv25)
            gv1: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv7, lv12, lv21)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(msgd, msgd_expected)


def test_adam_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    adam = Adam([x, y], 0.01).get_function()

    # fmt: off
    @R.function
    def adam_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((), dtype="float32") = optim_states[1]
            lv6: R.Tensor((), dtype="float32") = optim_states[2]
            lv7: R.Tensor((3, 3), dtype="float32") = optim_states[3]
            lv8: R.Tensor((3,), dtype="float32") = optim_states[4]
            lv9: R.Tensor((3, 3), dtype="float32") = optim_states[5]
            lv10: R.Tensor((3,), dtype="float32") = optim_states[6]
            lv11: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv12: R.Tensor((), dtype="float32") = R.multiply(lv5, R.const(0.9, "float32"))
            lv13: R.Tensor((), dtype="float32") = R.multiply(lv6, R.const(0.999, "float32"))
            lv14: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv7)
            lv15: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv2)
            lv16: R.Tensor((3, 3), dtype="float32") = R.add(lv14, lv15)
            lv17: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.999, "float32"), lv9)
            lv18: R.Tensor((3, 3), dtype="float32") = R.multiply(lv2, lv2)
            lv19: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.001, "float32"), lv18)
            lv20: R.Tensor((3, 3), dtype="float32") = R.add(lv17, lv19)
            lv21: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv12)
            lv22: R.Tensor((3, 3), dtype="float32") = R.divide(lv16, lv21)
            lv23: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv13)
            lv24: R.Tensor((3, 3), dtype="float32") = R.divide(lv20, lv23)
            lv25: R.Tensor((3, 3), dtype="float32") = R.sqrt(lv24)
            lv26: R.Tensor((3, 3), dtype="float32") = R.add(lv25, R.const(1e-08, "float32"))
            lv27: R.Tensor((3, 3), dtype="float32") = R.divide(lv22, lv26)
            lv28: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv27)
            lv29: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv28)
            lv30: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.9, "float32"), lv8)
            lv31: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv3)
            lv32: R.Tensor((3,), dtype="float32") = R.add(lv30, lv31)
            lv33: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.999, "float32"), lv10)
            lv34: R.Tensor((3,), dtype="float32") = R.multiply(lv3, lv3)
            lv35: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.001, "float32"), lv34)
            lv36: R.Tensor((3,), dtype="float32") = R.add(lv33, lv35)
            lv37: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv12)
            lv38: R.Tensor((3,), dtype="float32") = R.divide(lv32, lv37)
            lv39: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv13)
            lv40: R.Tensor((3,), dtype="float32") = R.divide(lv36, lv39)
            lv41: R.Tensor((3,), dtype="float32") = R.sqrt(lv40)
            lv42: R.Tensor((3,), dtype="float32") = R.add(lv41, R.const(1e-08, "float32"))
            lv43: R.Tensor((3,), dtype="float32") = R.divide(lv38, lv42)
            lv44: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv43)
            lv45: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv44)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv29, lv45)
            gv1: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (lv11, lv12, lv13, lv16, lv32, lv20, lv36)
            R.output(gv, gv1)
        return (gv, gv1)
    # fmt: on

    assert_structural_equal(adam, adam_expected)


def test_adam_complex():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    adam = Adam([x, y], 0.01, (0.8, 0.85), 1e-7, 0.1).get_function()

    @R.function
    def adam_expected(
        params: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
        gradients: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
        optim_states: R.Tuple(
            R.Tensor((), dtype="int64"),
            R.Tensor((), dtype="float32"),
            R.Tensor((), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")),
        R.Tuple(
            R.Tensor((), dtype="int64"),
            R.Tensor((), dtype="float32"),
            R.Tensor((), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
            R.Tensor((3, 3), dtype="float32"),
            R.Tensor((3,), dtype="float32"),
        ),
    ):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float32") = params[0]
            lv1: R.Tensor((3,), dtype="float32") = params[1]
            lv2: R.Tensor((3, 3), dtype="float32") = gradients[0]
            lv3: R.Tensor((3,), dtype="float32") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((), dtype="float32") = optim_states[1]
            lv6: R.Tensor((), dtype="float32") = optim_states[2]
            lv7: R.Tensor((3, 3), dtype="float32") = optim_states[3]
            lv8: R.Tensor((3,), dtype="float32") = optim_states[4]
            lv9: R.Tensor((3, 3), dtype="float32") = optim_states[5]
            lv10: R.Tensor((3,), dtype="float32") = optim_states[6]
            lv11: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv12: R.Tensor((), dtype="float32") = R.multiply(lv5, R.const(0.8, "float32"))
            lv13: R.Tensor((), dtype="float32") = R.multiply(lv6, R.const(0.85, "float32"))
            lv14: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv)
            lv15: R.Tensor((3, 3), dtype="float32") = R.add(lv14, lv2)
            lv16: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.8, "float32"), lv7)
            lv17: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.2, "float32"), lv15)
            lv18: R.Tensor((3, 3), dtype="float32") = R.add(lv16, lv17)
            lv19: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.85, "float32"), lv9)
            lv20: R.Tensor((3, 3), dtype="float32") = R.multiply(lv15, lv15)
            lv21: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv20)
            lv22: R.Tensor((3, 3), dtype="float32") = R.add(lv19, lv21)
            lv23: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv12)
            lv24: R.Tensor((3, 3), dtype="float32") = R.divide(lv18, lv23)
            lv25: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv13)
            lv26: R.Tensor((3, 3), dtype="float32") = R.divide(lv22, lv25)
            lv27: R.Tensor((3, 3), dtype="float32") = R.sqrt(lv26)
            lv28: R.Tensor((3, 3), dtype="float32") = R.add(lv27, R.const(1e-07, "float32"))
            lv29: R.Tensor((3, 3), dtype="float32") = R.divide(lv24, lv28)
            lv30: R.Tensor((3, 3), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv29)
            lv31: R.Tensor((3, 3), dtype="float32") = R.subtract(lv, lv30)
            lv32: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.1, "float32"), lv1)
            lv33: R.Tensor((3,), dtype="float32") = R.add(lv32, lv3)
            lv34: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.8, "float32"), lv8)
            lv35: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.2, "float32"), lv33)
            lv36: R.Tensor((3,), dtype="float32") = R.add(lv34, lv35)
            lv37: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.85, "float32"), lv10)
            lv38: R.Tensor((3,), dtype="float32") = R.multiply(lv33, lv33)
            lv39: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.15, "float32"), lv38)
            lv40: R.Tensor((3,), dtype="float32") = R.add(lv37, lv39)
            lv41: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv12)
            lv42: R.Tensor((3,), dtype="float32") = R.divide(lv36, lv41)
            lv43: R.Tensor((), dtype="float32") = R.subtract(R.const(1, "float32"), lv13)
            lv44: R.Tensor((3,), dtype="float32") = R.divide(lv40, lv43)
            lv45: R.Tensor((3,), dtype="float32") = R.sqrt(lv44)
            lv46: R.Tensor((3,), dtype="float32") = R.add(lv45, R.const(1e-07, "float32"))
            lv47: R.Tensor((3,), dtype="float32") = R.divide(lv42, lv46)
            lv48: R.Tensor((3,), dtype="float32") = R.multiply(R.const(0.01, "float32"), lv47)
            lv49: R.Tensor((3,), dtype="float32") = R.subtract(lv1, lv48)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3,), dtype="float32")) = (
                lv31,
                lv49,
            )
            gv1: R.Tuple(
                R.Tensor((), dtype="int64"),
                R.Tensor((), dtype="float32"),
                R.Tensor((), dtype="float32"),
                R.Tensor((3, 3), dtype="float32"),
                R.Tensor((3,), dtype="float32"),
                R.Tensor((3, 3), dtype="float32"),
                R.Tensor((3,), dtype="float32"),
            ) = (lv11, lv12, lv13, lv18, lv36, lv22, lv40)
            R.output(gv, gv1)
        return (gv, gv1)

    assert_structural_equal(adam, adam_expected)


def test_adam_float64():
    x = relax.Var("x", R.Tensor((3, 3), "float64"))
    y = relax.Var("y", R.Tensor((3,), "float64"))
    adam = Adam([x, y], 0.01, (0.8, 0.85), 1e-7, 0.1).get_function()

    @R.function
    def adam_expected(params: R.Tuple(R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64")), gradients: R.Tuple(R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float64"), R.Tensor((), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64"))) -> R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float64"), R.Tensor((), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64"))):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 3), dtype="float64") = params[0]
            lv1: R.Tensor((3,), dtype="float64") = params[1]
            lv2: R.Tensor((3, 3), dtype="float64") = gradients[0]
            lv3: R.Tensor((3,), dtype="float64") = gradients[1]
            lv4: R.Tensor((), dtype="int64") = optim_states[0]
            lv5: R.Tensor((), dtype="float64") = optim_states[1]
            lv6: R.Tensor((), dtype="float64") = optim_states[2]
            lv7: R.Tensor((3, 3), dtype="float64") = optim_states[3]
            lv8: R.Tensor((3,), dtype="float64") = optim_states[4]
            lv9: R.Tensor((3, 3), dtype="float64") = optim_states[5]
            lv10: R.Tensor((3,), dtype="float64") = optim_states[6]
            lv11: R.Tensor((), dtype="int64") = R.add(lv4, R.const(1, "int64"))
            lv12: R.Tensor((), dtype="float64") = R.multiply(lv5, R.const(0.8, "float64"))
            lv13: R.Tensor((), dtype="float64") = R.multiply(lv6, R.const(0.85, "float64"))
            lv14: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.1, "float64"), lv)
            lv15: R.Tensor((3, 3), dtype="float64") = R.add(lv14, lv2)
            lv16: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.8, "float64"), lv7)
            lv17: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.2, "float64"), lv15)
            lv18: R.Tensor((3, 3), dtype="float64") = R.add(lv16, lv17)
            lv19: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.85, "float64"), lv9)
            lv20: R.Tensor((3, 3), dtype="float64") = R.multiply(lv15, lv15)
            lv21: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.15, "float64"), lv20)
            lv22: R.Tensor((3, 3), dtype="float64") = R.add(lv19, lv21)
            lv23: R.Tensor((), dtype="float64") = R.subtract(R.const(1, "float64"), lv12)
            lv24: R.Tensor((3, 3), dtype="float64") = R.divide(lv18, lv23)
            lv25: R.Tensor((), dtype="float64") = R.subtract(R.const(1, "float64"), lv13)
            lv26: R.Tensor((3, 3), dtype="float64") = R.divide(lv22, lv25)
            lv27: R.Tensor((3, 3), dtype="float64") = R.sqrt(lv26)
            lv28: R.Tensor((3, 3), dtype="float64") = R.add(lv27, R.const(1e-07, "float64"))
            lv29: R.Tensor((3, 3), dtype="float64") = R.divide(lv24, lv28)
            lv30: R.Tensor((3, 3), dtype="float64") = R.multiply(R.const(0.01, "float64"), lv29)
            lv31: R.Tensor((3, 3), dtype="float64") = R.subtract(lv, lv30)
            lv32: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.1, "float64"), lv1)
            lv33: R.Tensor((3,), dtype="float64") = R.add(lv32, lv3)
            lv34: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.8, "float64"), lv8)
            lv35: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.2, "float64"), lv33)
            lv36: R.Tensor((3,), dtype="float64") = R.add(lv34, lv35)
            lv37: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.85, "float64"), lv10)
            lv38: R.Tensor((3,), dtype="float64") = R.multiply(lv33, lv33)
            lv39: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.15, "float64"), lv38)
            lv40: R.Tensor((3,), dtype="float64") = R.add(lv37, lv39)
            lv41: R.Tensor((), dtype="float64") = R.subtract(R.const(1, "float64"), lv12)
            lv42: R.Tensor((3,), dtype="float64") = R.divide(lv36, lv41)
            lv43: R.Tensor((), dtype="float64") = R.subtract(R.const(1, "float64"), lv13)
            lv44: R.Tensor((3,), dtype="float64") = R.divide(lv40, lv43)
            lv45: R.Tensor((3,), dtype="float64") = R.sqrt(lv44)
            lv46: R.Tensor((3,), dtype="float64") = R.add(lv45, R.const(1e-07, "float64"))
            lv47: R.Tensor((3,), dtype="float64") = R.divide(lv42, lv46)
            lv48: R.Tensor((3,), dtype="float64") = R.multiply(R.const(0.01, "float64"), lv47)
            lv49: R.Tensor((3,), dtype="float64") = R.subtract(lv1, lv48)
            gv: R.Tuple(R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64")) = (lv31, lv49)
            gv1: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((), dtype="float64"), R.Tensor((), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64"), R.Tensor((3, 3), dtype="float64"), R.Tensor((3,), dtype="float64")) = (lv11, lv12, lv13, lv18, lv36, lv22, lv40)
            R.output(gv, gv1)
        return (gv, gv1)

    assert_structural_equal(adam, adam_expected)

test_adam_float64()
# if __name__ == "__main__":
#     tvm.testing.main()
