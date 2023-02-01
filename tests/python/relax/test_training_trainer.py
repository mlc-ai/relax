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
import numpy as np

from tvm import relax, TVMError
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import ir as I, relax as R
from tvm.relax.training.optimizer import SGD
from tvm.relax.training.loss import MSELoss
from tvm.relax.training.trainer import Trainer


def _get_trainer() -> Trainer:
    @I.ir_module
    class MLP:
        @R.function
        def main(
            x: R.Tensor((1, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    trainer = Trainer(MLP, [1, 2])
    return trainer


def _make_dataset():
    N = 100
    return [
        [
            np.random.uniform(size=(1, 10)).astype(np.float32),
            np.array([[0, 0, 1, 0, 0]]).astype(np.float32),
        ]
        for _ in range(N)
    ]


@tvm.testing.parametrize_targets("llvm")
def test_execute(target, dev):
    trainer = _get_trainer()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    trainer.set_vm_config(target="llvm")
    trainer.set_optimizer(optim_type=SGD, lr=0.001).setup().rand_init_params()

    dataset = _make_dataset()
    last_loss = np.inf
    for epoch in range(5):
        for i, data in enumerate(dataset):
            loss = trainer.backward(data[0], data[1])


@tvm.testing.parametrize_targets("llvm")
def test_load_export_params(target, dev):
    trainer = _get_trainer()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    trainer.set_vm_config(target="llvm")
    trainer.set_optimizer(optim_type=SGD, lr=0.001).setup().rand_init_params()

    dataset = _make_dataset()
    for i, data in enumerate(dataset):
        loss = trainer.backward(data[0], data[1])

    param_dict = trainer.export_params()
    assert "w0" in param_dict
    assert "b0" in param_dict

    trainer1 = _get_trainer()
    trainer1.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    trainer1.set_vm_config(target="llvm")
    trainer1.set_optimizer(optim_type=SGD, lr=0.001).setup().load_params(param_dict)

    x_sample = dataset[np.random.randint(len(dataset))][0]
    tvm.testing.assert_allclose(
        trainer.forward(x_sample).numpy(), trainer1.forward(x_sample).numpy()
    )


@tvm.testing.parametrize_targets("llvm")
def test_get_mod(target, dev):
    trainer = _get_trainer()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    trainer.set_vm_config(target="llvm")
    trainer.set_optimizer(optim_type=SGD, lr=0.001).setup()

    # fmt: off
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32")) -> R.Tensor((1, 5), dtype="float32"):
            with R.dataflow():
                lv0: R.Tensor((1, 5), dtype="float32") = R.matmul(x, w0, out_dtype="")
                lv1: R.Tensor((1, 5), dtype="float32") = R.add(lv0, b0)
                out: R.Tensor((1, 5), dtype="float32") = R.nn.relu(lv1)
                R.output(out)
            return out

        @R.function
        def main_loss(x: R.Tensor((1, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32"), targets: R.Tensor((1, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            with R.dataflow():
                lv0: R.Tensor((1, 5), dtype="float32") = R.matmul(x, w0, out_dtype="")
                lv1: R.Tensor((1, 5), dtype="float32") = R.add(lv0, b0)
                out: R.Tensor((1, 5), dtype="float32") = R.nn.relu(lv1)
                lv: R.Tensor((1, 5), dtype="float32") = R.subtract(out, targets)
                lv11: R.Tensor((1, 5), dtype="float32") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float32") = R.sum(lv11, axis=None, keepdims=False)
                R.output(gv)
            return gv

        @R.function
        def main_loss_adjoint(x: R.Tensor((1, 10), dtype="float32"), w0: R.Tensor((10, 5), dtype="float32"), b0: R.Tensor((5,), dtype="float32"), targets: R.Tensor((1, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((5,), dtype="float32"))):
            with R.dataflow():
                lv0: R.Tensor((1, 5), dtype="float32") = R.matmul(x, w0, out_dtype="")
                lv1: R.Tensor((1, 5), dtype="float32") = R.add(lv0, b0)
                out: R.Tensor((1, 5), dtype="float32") = R.nn.relu(lv1)
                lv: R.Tensor((1, 5), dtype="float32") = R.subtract(out, targets)
                lv11: R.Tensor((1, 5), dtype="float32") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float32") = R.sum(lv11, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv1_adjoint: R.Tensor((1, 5), dtype="float32") = R.broadcast_to(gv_adjoint, (1, 5))
                lv2: R.Tensor((1, 5), dtype="float32") = R.multiply(lv1_adjoint, lv)
                lv12: R.Tensor((1, 5), dtype="float32") = R.multiply(lv1_adjoint, lv)
                lv_adjoint: R.Tensor((1, 5), dtype="float32") = R.add(lv2, lv12)
                out_adjoint: R.Tensor((1, 5), dtype="float32") = lv_adjoint
                lv21: R.Tensor((1, 5), dtype="float32") = R.zeros((1, 5), dtype="float32")
                lv3: R.Tensor((1, 5), dtype="bool") = R.less(lv1, lv21)
                lv4: R.Tensor((1, 5), dtype="float32") = R.ones((1, 5), dtype="float32")
                lv5: R.Tensor((1, 5), dtype="float32") = R.multiply(lv4, out_adjoint)
                lv1_adjoint1: R.Tensor((1, 5), dtype="float32") = R.where(lv3, lv21, lv5)
                lv0_adjoint: R.Tensor((1, 5), dtype="float32") = lv1_adjoint1
                lv6: R.Tensor((10, 1), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                lv7: R.Tensor((10, 5), dtype="float32") = R.matmul(lv6, lv0_adjoint, out_dtype="")
                w0_adjoint: R.Tensor((10, 5), dtype="float32") = R.collapse_sum_to(lv7, (10, 5))
                b0_adjoint: R.Tensor((5,), dtype="float32") = R.collapse_sum_to(lv1_adjoint1, (5,))
                R.output(gv, w0_adjoint, b0_adjoint)
            return (gv, (w0_adjoint, b0_adjoint))
    # fmt: on

    mod = trainer.mod
    assert_structural_equal(Expected["main_loss"], trainer.mod["main_loss"])
    assert_structural_equal(Expected["main_loss_adjoint"], trainer.mod["main_loss_adjoint"])


@tvm.testing.parametrize_targets("llvm")
def test_setting_error(target, dev):
    trainer = _get_trainer()

    with pytest.raises(TVMError):
        trainer.mod
    with pytest.raises(TVMError):
        trainer.vm
    with pytest.raises(TVMError):
        trainer.optimizer
    with pytest.raises(TVMError):
        trainer.rand_init_params()
    with pytest.raises(TVMError):
        trainer.load_params({})

    with pytest.raises(TVMError):
        trainer.setup()

    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    with pytest.raises(TVMError):
        trainer.setup()

    trainer.set_vm_config(target="llvm")
    with pytest.raises(TVMError):
        trainer.setup()
    trainer.set_optimizer(optim_type=SGD, lr=0.001)
    trainer.setup()


def test_invalid_mod():
    @I.ir_module
    class InvalidMLP:
        @R.function
        def main(
            x: R.Tensor((1, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                gv = R.add(lv0, b0)
                out = R.nn.relu(gv)
                R.output(gv, out)
            return gv, out

    with pytest.raises(ValueError):
        trainer = Trainer(InvalidMLP, [1, 2])


if __name__ == "__main__":
    tvm.testing.main()
