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
import tvm
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.relax.training import SGD, MomentumSGD
from tvm.script.parser import relax as R


def test_sgd_simple():
    pass


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


def test_momentum_sgd():
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


if __name__ == "__main__":
    tvm.testing.main()
