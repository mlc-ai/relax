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

from typing import Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.relax import Function
from tvm.script import relax as R, ir as I


def _check(before: IRModule, expected: IRModule, mode: str):
    if isinstance(before, Function):
        before = IRModule({"main": before})
    if isinstance(expected, Function):
        expected = IRModule({"main": expected})
    after = relax.transform.SimplifyNorm("main", mode)(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_batch_norm_simple():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                gv = bn[0]
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tensor((1, 64, 112, 112), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_mean, axis=[0, 2, 3]
                )
                lv1: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_var, axis=[0, 2, 3]
                )
                lv3: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                lv4: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(gamma, axis=[0, 2, 3])
                lv7: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv7, lv8)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv9, moving_mean, moving_var)
                gv: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                R.output(gv)
            return gv

    _check(Before, Expected, "eval")


def test_batch_norm_complex():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                )
                gv0 = bn[0]
                gv1 = bn[1]
                R.output(gv0, gv1)
            return gv0, gv1

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((1, 64, 112, 112), dtype="float32"), R.Tensor((64,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_mean, axis=[0, 2, 3]
                )
                lv1: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(
                    moving_var, axis=[0, 2, 3]
                )
                lv3: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv2, R.const(9.9999997473787516e-06, "float32")
                )
                lv4: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(gamma, axis=[0, 2, 3])
                lv7: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv7, lv8)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv9, moving_mean, moving_var)
                gv0: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                gv1: R.Tensor((64,), dtype="float32") = bn[1]
                R.output(gv0, gv1)
            return (gv0, gv1)

    _check(Before, Expected, "eval")


def test_batch_norm_training():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), "float32"),
            gamma: R.Tensor((64,), "float32"),
            beta: R.Tensor((64,), "float32"),
            moving_mean: R.Tensor((64,), "float32"),
            moving_var: R.Tensor((64,), "float32"),
        ):
            with R.dataflow():
                bn = R.nn.batch_norm(
                    x,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    momentum=0.1,
                )
                gv0 = bn[0]
                gv1 = bn[1]
                gv2 = bn[2]
                R.output(gv0, gv1, gv2)
            return gv0, gv1, gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 64, 112, 112), dtype="float32"),
            gamma: R.Tensor((64,), dtype="float32"),
            beta: R.Tensor((64,), dtype="float32"),
            moving_mean: R.Tensor((64,), dtype="float32"),
            moving_var: R.Tensor((64,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((1, 64, 112, 112), dtype="float32"),
            R.Tensor((64,), dtype="float32"),
            R.Tensor((64,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((64,), dtype="float32") = R.mean(x, axis=[0, 2, 3], keepdims=False)
                lv1: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(lv, axis=[0, 2, 3])
                lv2: R.Tensor((1, 64, 112, 112), dtype="float32") = R.subtract(x, lv1)
                lv3: R.Tensor((64,), dtype="float32") = R.variance(
                    x, axis=[0, 2, 3], keepdims=False
                )
                lv4: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(lv3, axis=[0, 2, 3])
                lv5: R.Tensor((1, 64, 1, 1), dtype="float32") = R.add(
                    lv4, R.const(9.9999997473787516e-06, "float32")
                )
                lv6: R.Tensor((1, 64, 1, 1), dtype="float32") = R.sqrt(lv5)
                lv7: R.Tensor((1, 64, 112, 112), dtype="float32") = R.divide(lv2, lv6)
                lv8: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(gamma, axis=[0, 2, 3])
                lv9: R.Tensor((1, 64, 112, 112), dtype="float32") = R.multiply(lv7, lv8)
                lv10: R.Tensor((1, 64, 1, 1), dtype="float32") = R.expand_dims(beta, axis=[0, 2, 3])
                lv11: R.Tensor((1, 64, 112, 112), dtype="float32") = R.add(lv9, lv10)
                lv12: R.Tensor((64,), dtype="float32") = R.multiply(
                    R.const(0.89999997615814209, "float32"), moving_mean
                )
                lv13: R.Tensor((64,), dtype="float32") = R.multiply(
                    R.const(0.10000000149011612, "float32"), lv
                )
                lv14: R.Tensor((64,), dtype="float32") = R.add(lv12, lv13)
                lv15: R.Tensor((64,), dtype="float32") = R.multiply(
                    R.const(0.89999997615814209, "float32"), moving_var
                )
                lv16: R.Tensor((64,), dtype="float32") = R.multiply(
                    R.const(0.10000000149011612, "float32"), lv3
                )
                lv17: R.Tensor((64,), dtype="float32") = R.add(lv15, lv16)
                bn: R.Tuple(
                    R.Tensor((1, 64, 112, 112), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                    R.Tensor((64,), dtype="float32"),
                ) = (lv11, lv14, lv17)
                gv0: R.Tensor((1, 64, 112, 112), dtype="float32") = bn[0]
                gv1: R.Tensor((64,), dtype="float32") = bn[1]
                gv2: R.Tensor((64,), dtype="float32") = bn[2]
                R.output(gv0, gv1, gv2)
            return (gv0, gv1, gv2)

    _check(Before, Expected, "training")


if __name__ == "__main__":
    test_batch_norm_training()
