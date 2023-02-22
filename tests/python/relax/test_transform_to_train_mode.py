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
from tvm.relax.transform import ToTrainMode
from tvm.script import relax as R, tir as T, ir as I
from tvm.ir.base import assert_structural_equal
import tvm.testing


def test_rewrite_batch_norm():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2, 4, 3, 3), dtype="float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 3, 3), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            gv: R.Tuple(
                R.Tensor((2, 4, 3, 3), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
            ) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 4, 3, 3), dtype="float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 3, 3), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            gv: R.Tuple(
                R.Tensor((2, 4, 3, 3), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
            ) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1, training=True)
            return gv

    After = ToTrainMode()(Before)
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
