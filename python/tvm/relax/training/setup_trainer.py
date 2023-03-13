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
from tvm._ffi.base import TVMError
from tvm.ir.module import IRModule
from tvm.tir.expr import IntImm

from ..analysis import well_formed
from ..expr import Tuple
from ..struct_info import TensorStructInfo
from ..training.utils import AppendLoss
from ..transform import LegalizeOps, Gradient
from .loss import Loss
from .optimizer import Optimizer


@tvm.transform.module_pass(opt_level=0, name="SetupTrainer")
class SetupTrainer:
    """Transform a NN backbone module to a complete, legalized trainer module.

    The transformed module will contains the following functions:
    - `predict`: Predicts the result. It is provided in the input module.
    - `loss`: Calculates the specified loss between the prediction result and the ground truth.
    - `loss_adjoint`: Calculates the loss and the adjoints of parameters.
    - `update_params`: Takes the parameters, the adjoints of parameters and the optimizer state as
    the inputs and returns updated parameters and new optimizer state. It contains a func attr named
    `optim_state` which is the initial state of the specified optimizer.

    Parameters
    ----------
    loss : Loss
        The loss function. It will be appended to the backbone function using
        relax.transform.AppendLoss.

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
    LOSS_FUNC_NAME: str = "predict_loss"
    ADJOINT_FUNC_NAME: str = "predict_loss_adjoint"
    UPDATE_PARAMS_FUNC_NAME: str = "update_params"

    PARAMS_NUM_ATTR_KEY: str = "params_num"
    STATES_NUM_ATTR_KEY: str = "states_num"

    def __init__(self, loss: Loss, optimizer: Optimizer, loss_args: List[TensorStructInfo]):
        self._loss = loss
        self._optimizer = optimizer
        self._loss_args = loss_args

    def _check_mod(self, mod: IRModule):
        if not well_formed(mod):
            raise ValueError("SetupTrainer: The backbone module is not well formed.")
        try:
            func = mod[self.PREDICT_FUNC_NAME]
        except TVMError as exc:
            raise ValueError(
                f"SetupTrainer: The backbone module does not contain a function named "
                f"{self.PREDICT_FUNC_NAME}"
            ) from exc

        # Check function attrs
        if not self.PARAMS_NUM_ATTR_KEY in mod.attrs or not isinstance(
            mod.attrs[self.PARAMS_NUM_ATTR_KEY], IntImm
        ):
            raise ValueError(
                f"SetupTrainer: The backbone module should has integer attribute "
                f"{self.PARAMS_NUM_ATTR_KEY}"
            )
        if not self.STATES_NUM_ATTR_KEY in mod.attrs or not isinstance(
            mod.attrs[self.STATES_NUM_ATTR_KEY], IntImm
        ):
            raise ValueError(
                f"SetupTrainer: The backbone module should has integer attribute "
                f"{self.STATES_NUM_ATTR_KEY}"
            )

        nparam = int(mod.attrs[self.PARAMS_NUM_ATTR_KEY])
        nstate = int(mod.attrs[self.STATES_NUM_ATTR_KEY])

        # Check parameters and return values
        if len(func.params) < nparam + nstate:
            raise ValueError(
                "SetupTrainer: The number of parameters of the predict function should be no less "
                "than the number of parameters and states"
            )

        if nstate > 0:
            if not isinstance(func.body.body, Tuple) or len(func.body.body) <= nstate:
                raise ValueError(
                    "SetupTrainer: When model state exists, the predict function should return a "
                    "tuple of length more than the number of states"
                )

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        """Setup the trainer in 3 steps.
        1. Prepare Loss.
        2. Gradient pass.
        3. Add Optimizer function.
        4. Legalize operators.
        """
        self._check_mod(mod)

        # AppendLoss pass.
        mod = AppendLoss(
            self.PREDICT_FUNC_NAME,
            self._loss(*self._loss_args),  # type: ignore
            self._loss.num_backbone_outputs,
            self.LOSS_FUNC_NAME,
        )(mod)

        # Gradient pass.
        params_num = int(mod.attrs[self.PARAMS_NUM_ATTR_KEY])
        states_num = int(mod.attrs[self.STATES_NUM_ATTR_KEY])
        inputs_num = len(mod[self.PREDICT_FUNC_NAME].params) - params_num - states_num
        params = mod[self.LOSS_FUNC_NAME].params[inputs_num : inputs_num + params_num]
        mod = Gradient(self.LOSS_FUNC_NAME, require_grads=params, target_index=0)(mod)

        # Build Optimizer.
        self._optimizer.init(params)
        mod[self.UPDATE_PARAMS_FUNC_NAME] = self._optimizer.get_function()

        # Module attrs
        mod = mod.with_attrs(
            {
                # inputs_num
                "inputs_num": inputs_num,
                # function names
                "predict_fn": self.PREDICT_FUNC_NAME,
                "loss_fn": self.LOSS_FUNC_NAME,
                "adjoint_fn": self.ADJOINT_FUNC_NAME,
                "update_params_fn": self.UPDATE_PARAMS_FUNC_NAME,
                # optimizer states
                "optim_state": self._optimizer.state,
            }
        )

        # Legalize operators.
        return LegalizeOps()(mod)
