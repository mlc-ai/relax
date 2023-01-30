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
import pytest
import tvm.testing
from tvm import relax, TVMError
from tvm.ir.base import assert_structural_equal
from tvm.script import relax as R


def test_append_loss_basic_extend():
    @R.function
    def orig(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")):
        with R.dataflow():
            gv0 = R.sum(x)
            gv1 = R.sum(y)
            R.output(gv0, gv1)
        return gv0, gv1

    @R.function
    def loss(arg1: R.Tensor((), dtype="float32"), arg2: R.Tensor((), dtype="float32")):
        with R.dataflow():
            gv0 = R.add(arg1, arg2)
            R.output(gv0)
        return gv0

    @R.function
    def expected(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=0):
        # block 0
        with R.dataflow():
            gv0: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            gv01: R.Tensor((), dtype="float32") = R.add(gv0, gv1)
            R.output(gv01)
        return gv01

    after = relax.training.utils.append_loss(orig, loss)
    assert_structural_equal(after, expected)


def test_append_loss_extra_params():
    @R.function
    def orig(x: R.Tensor((3, 3), dtype="float32")):
        with R.dataflow():
            gv0 = R.sum(x)
            gv1 = R.add(x, x)
            R.output(gv0, gv1)
        return gv0, gv1

    @R.function
    def loss(
        arg1: R.Tensor((), dtype="float32"),
        arg2: R.Tensor((3, 3), dtype="float32"),
        arg3: R.Tensor((3, 3), dtype="float32"),
    ):
        with R.dataflow():
            gv0 = R.add(arg2, arg3)
            R.output(gv0)
        return gv0

    @R.function
    def expected(
        x: R.Tensor((3, 3), dtype="float32"), arg3: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=2):
        with R.dataflow():
            gv0: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
            gv01: R.Tensor((3, 3), dtype="float32") = R.add(gv1, arg3)
            R.output(gv01)
        return gv01

    after = relax.training.utils.append_loss(orig, loss)
    assert_structural_equal(after, expected)


def test_append_loss_nested_tuple():
    @R.function
    def orig(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")):
        with R.dataflow():
            gv0 = R.add(x, x)
            gv1 = R.sum(y)
            gv2 = R.add(x, y)
            R.output(gv0, gv1, gv2)
        return (gv0, gv1), gv2

    @R.function
    def loss(
        arg1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32")),
        arg2: R.Tensor((3, 3), dtype="float32"),
    ):
        with R.dataflow():
            arg10 = arg1[0]
            gv0 = R.add(arg10, arg2)
            R.output(gv0)
        return gv0

    @R.function
    def expected(
        x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
    ) -> R.Tensor((3, 3), dtype="float32"):
        # block 0
        with R.dataflow():
            gv0: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
            gv1: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
            gv2: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            ret_0: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((), dtype="float32")) = (
                gv0,
                gv1,
            )
            arg10: R.Tensor((3, 3), dtype="float32") = ret_0[0]
            gv01: R.Tensor((3, 3), dtype="float32") = R.add(arg10, gv2)
            R.output(gv01)
        return gv01

    after = relax.training.utils.append_loss(orig, loss)
    assert_structural_equal(after, expected)


def test_append_loss_wrong_struct_info():
    @R.function
    def orig(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")):
        with R.dataflow():
            gv0 = R.sum(x)
            gv1 = R.sum(y)
            R.output(gv0, gv1)
        return gv0, gv1

    @R.function
    def loss(arg1: R.Tensor((), dtype="float64"), arg2: R.Tensor((), dtype="float64")):
        with R.dataflow():
            gv0 = R.add(arg1, arg2)
            R.output(gv0)
        return gv0

    with pytest.raises(TVMError):
        after = relax.training.utils.append_loss(orig, loss)


if __name__ == "__main__":
    tvm.testing.main()
