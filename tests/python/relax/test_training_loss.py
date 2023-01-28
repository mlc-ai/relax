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
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R


@R.function
def forward(
    x: R.Tensor((2, 4), dtype="float32"),
    w: R.Tensor((4, 4), dtype="float32"),
    b: R.Tensor((2, 4), dtype="float32"),
) -> R.Tensor((2, 4), dtype="float32"):
    with R.dataflow():
        lv: R.Tensor((2, 4), dtype="float32") = R.matmul(x, w)
        out: R.Tensor((2, 4), dtype="float32") = R.add(lv, b)
        R.output(out)
    return out


def test_l1_loss():
    N = 3
    C = 5
    predictions = relax.TensorStructInfo((N, C), "float32")
    targets = relax.TensorStructInfo((N, C), "float32")
    l1_loss = relax.training.L1Loss()

    # fmt: off
    @R.function
    def expected(predictions: R.Tensor((3, 5), dtype="float32"), targets: R.Tensor((3, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        R.func_attr({"global_symbol": "l1_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.subtract(predictions, targets)
            lv1: R.Tensor((3, 5), dtype="float32") = R.abs(lv)
            gv: R.Tensor((), dtype="float32") = R.mean(lv1, axis=None, keepdims=False)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(l1_loss(predictions, targets), expected)


def test_l1_loss_append():
    s = forward.ret_struct_info
    l1_loss = relax.training.L1Loss(reduction="sum")
    forward_with_loss = relax.training.utils.append_loss(forward, l1_loss(s, s))

    # fmt: off
    @R.function
    def expected(x: R.Tensor((2, 4), dtype="float32"), w: R.Tensor((4, 4), dtype="float32"), b: R.Tensor((2, 4), dtype="float32"), targets: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((2, 4), dtype="float32") = R.matmul(x, w, out_dtype="")
            out: R.Tensor((2, 4), dtype="float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), dtype="float32") = R.subtract(out, targets)
            lv11: R.Tensor((2, 4), dtype="float32") = R.abs(lv1)
            gv: R.Tensor((), dtype="float32") = R.sum(lv11, axis=None, keepdims=False)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(forward_with_loss, expected)


def test_mse_loss():
    N = 3
    C = 5
    predictions = relax.TensorStructInfo((N, C), "float32")
    targets = relax.TensorStructInfo((N, C), "float32")
    mse_loss = relax.training.MSELoss()

    # fmt: off
    @R.function
    def expected(predictions: R.Tensor((3, 5), dtype="float32"), targets: R.Tensor((3, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # function attr dict
        R.func_attr({"global_symbol": "mse_loss"})
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.subtract(predictions, targets)
            lv1: R.Tensor((3, 5), dtype="float32") = R.multiply(lv, lv)
            gv: R.Tensor((), dtype="float32") = R.mean(lv1, axis=None, keepdims=False)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(mse_loss(predictions, targets), expected)


def test_mse_loss_append():
    s = forward.ret_struct_info
    mse_loss = relax.training.MSELoss(reduction="sum")
    forward_with_loss = relax.training.utils.append_loss(forward, mse_loss(s, s))

    # fmt: off
    @R.function
    def expected(x: R.Tensor((2, 4), dtype="float32"), w: R.Tensor((4, 4), dtype="float32"), b: R.Tensor((2, 4), dtype="float32"), targets: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((2, 4), dtype="float32") = R.matmul(x, w, out_dtype="")
            out: R.Tensor((2, 4), dtype="float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), dtype="float32") = R.subtract(out, targets)
            lv11: R.Tensor((2, 4), dtype="float32") = R.multiply(lv1, lv1)
            gv: R.Tensor((), dtype="float32") = R.sum(lv11, axis=None, keepdims=False)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(forward_with_loss, expected)


def test_cross_entropy_loss():
    N = 3
    C = 5
    predictions = relax.TensorStructInfo((N, C), "float32")
    targets = relax.TensorStructInfo((N,), "int64")
    weights = relax.TensorStructInfo((C,), "float32")
    cross_entropy_loss = relax.training.CrossEntropyLoss(reduction="sum", ignore_index=1)

    # fmt: off
    @R.function
    def expected(predictions: R.Tensor((3, 5), dtype="float32"), targets: R.Tensor((3,), dtype="int64"), weights: R.Tensor((5,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # function attr dict
        R.func_attr({"global_symbol": "cross_entropy_loss"})
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.nn.log_softmax(predictions, axis=-1)
            gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(lv, targets, weights, reduction="sum", ignore_index=1)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(cross_entropy_loss(predictions, targets, weights), expected)


def test_cross_entropy_loss_without_weights():
    N = 3
    C = 5
    predictions = relax.TensorStructInfo((N, C), "float32")
    targets = relax.TensorStructInfo((N,), "int64")
    cross_entropy_loss = relax.training.CrossEntropyLoss()

    # fmt: off
    @R.function
    def expected(predictions: R.Tensor((3, 5), dtype="float32"), targets: R.Tensor((3,), dtype="int64")) -> R.Tensor((), dtype="float32"):
        # function attr dict
        R.func_attr({"global_symbol": "cross_entropy_loss"})
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.nn.log_softmax(predictions, axis=-1)
            gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(lv, targets, reduction="mean", ignore_index=-100)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(cross_entropy_loss(predictions, targets), expected)


def test_cross_entropy_loss_append():
    s = forward.ret_struct_info
    N = s.shape[0]
    C = s.shape[1]
    targets = relax.TensorStructInfo((N,), "int64")
    weights = relax.TensorStructInfo((C,), "float32")
    cross_entropy_loss = relax.training.CrossEntropyLoss(reduction="sum", ignore_index=1)
    forward_with_loss = relax.training.utils.append_loss(
        forward, cross_entropy_loss(s, targets, weights)
    )

    # fmt: off
    @R.function
    def expected(x: R.Tensor((2, 4), dtype="float32"), w: R.Tensor((4, 4), dtype="float32"), b: R.Tensor((2, 4), dtype="float32"), targets: R.Tensor((2,), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((2, 4), dtype="float32") = R.matmul(x, w, out_dtype="")
            out: R.Tensor((2, 4), dtype="float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), dtype="float32") = R.nn.log_softmax(out, axis=-1)
            gv: R.Tensor((), dtype="float32") = R.nn.nll_loss(lv1, targets, weights, reduction="sum", ignore_index=1)
            R.output(gv)
        return gv
    # fmt: on

    assert_structural_equal(forward_with_loss, expected)


if __name__ == "__main__":
    tvm.testing.main()
