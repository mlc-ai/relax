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
import numpy as np
from tvm import relax
from tvm.script.parser import ir as I, relax as R
from tvm.relax.training.optimizer import SGD
from tvm.relax.training.loss import MSELoss
from tvm.relax.training.trainer import Trainer


def test_basic(target, dev):
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

    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    trainer = Trainer(MLP, [1, 2])
    trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    trainer.set_vm_config(target="llvm")
    trainer.set_optimizer(optim_type=SGD, lr=0.001).setup()

    N = 100
    dataset = [
        [
            np.random.uniform(size=(1, 10)).astype(np.float32),
            np.array([[0, 0, 1, 0, 0]]).astype(np.float32),
        ]
        for _ in range(N)
    ]

    last_loss = np.inf
    for epoch in range(5):
        for i, data in enumerate(dataset):
            loss = trainer.backward(data[0], data[1])
        print(f"Epoch #{epoch}. Loss = {loss}.")
        assert last_loss > loss
        last_loss = loss


test_basic("llvm", tvm.cpu(0))
