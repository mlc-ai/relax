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
# pylint: disable=not-callable, unused-argument
"""Setup Trainer Pass."""
from typing import List

import tvm
from tvm.ir.module import IRModule

from ..transform import LegalizeOps, Gradient
from .loss import Loss
from .utils import append_loss
from .optimizer import Optimizer
from ..struct_info import TensorStructInfo
from ..analysis import well_formed


@tvm.transform.module_pass(opt_level=0, name="SetupTrainer")
class SetupTrainer:
    """Transform a NN backbone module to a complete, legalized trainer module.

    The transformed module will contains the following functions:
    - `predict`: Predicts the result. It is provided in the input module.
    - `loss`: Calculates the specified loss between the prediction results and the ground truth.
    - `loss_adjoint`: Calculates the loss and the adjoints of parameters.
    - `update_params`: Takes the parameters, the adjoints of parameters and the optimizer state as
    the inputs and returns updated parameters and new optimizer state. It contains a func attr named
    `optim_state` which is the initial state of the specified optimizer.

    Parameters
    ----------
    loss : Loss
        The loss function. It will be appended to the backbone function using
        relax.training.utils.append_loss.

    optimizer : Optimizer
        The optimizer. It will be put as the `update_params` function of the transformed module. The
        initial state will be put in the func attr `optim_state` of this function.

    loss_args : List[TensorStructInfo]
        The arguments to call the loss function.

    Notes
    -----
    This transform requires the input module to have the following properties:
    - The module should contain a function named `predict`.
    - This prediction function should only return a single Tensor which is the prediction
    result of the backbone.
    - This prediction function should have a func attr `params_num`, which indicates that
    the first `params_num` inputs are the parameters (which will be gradiented and trained)
    of the NN module.
    """

    PREDICT_FUNC_NAME: str = "predict"
    LOSS_FUNC_NAME: str = "loss"
    ADJOINT_FUNC_NAME: str = "loss_adjoint"
    UPDATE_PARAMS_FUNC_NAME: str = "update_params"

    PARAMS_NUM_ATTR_KEY: str = "params_num"
    OPTIM_STATE_ATTR_KEY: str = "optim_state"

    def __init__(self, loss: Loss, optimizer: Optimizer, loss_args: List[TensorStructInfo]):
        self._loss = loss
        self._optimizer = optimizer
        self._loss_args = loss_args

    @classmethod
    def _check_backbone_validity(cls, mod: IRModule):
        if not well_formed(mod):
            raise ValueError("Backbone Invalid: The backbone is not well formed.")
        ret_sinfo = mod[cls.PREDICT_FUNC_NAME].struct_info.ret
        if not isinstance(ret_sinfo, TensorStructInfo):
            raise ValueError(
                "Backbone Invalid: The predict function is expected to have a single Tensor "
                "return value, which serves as the prediction result of the module. But got ",
                ret_sinfo,
            )

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        """Setup the trainer in 3 steps.
        1. Prepare Loss.
        2. Gradient pass.
        3. Add Optimizer function.
        4. Legalize operators.
        """

        self._check_backbone_validity(mod)
        # Step 1: Prepare Loss.
        predict_with_loss = append_loss(
            mod[self.PREDICT_FUNC_NAME],
            self._loss(*self._loss_args),  # type: ignore
        )
        mod[self.LOSS_FUNC_NAME] = predict_with_loss

        # Step 2: Gradient pass.
        params_num = int(mod[self.PREDICT_FUNC_NAME].attrs[self.PARAMS_NUM_ATTR_KEY])
        require_grads = mod[self.LOSS_FUNC_NAME].params[:params_num]
        mod = Gradient(self.LOSS_FUNC_NAME, require_grads=require_grads)(mod)

        # Step 3: Build Optimizer.
        self._optimizer.init(list(mod[self.ADJOINT_FUNC_NAME].params)[:params_num])
        mod[self.UPDATE_PARAMS_FUNC_NAME] = self._optimizer.get_function().with_attr(
            self.OPTIM_STATE_ATTR_KEY, self._optimizer.state
        )

        # Step 4: Legalize operators.
        return LegalizeOps()(mod)  # type: ignore
