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
# pylint: disable=not-callable,invalid-name,unused-argument
"""Unified Trainer API for relax training."""

from typing import Union, List, Type, Optional, Any, Dict
import numpy as np  # type: ignore

import tvm
from tvm import TVMError
from tvm.ir.module import IRModule
from tvm.runtime.container import tuple_object
from tvm.runtime.ndarray import NDArray

from ..transform import LegalizeOps, Gradient
from .loss import Loss
from .utils import append_loss
from .optimizer import Optimizer
from ..struct_info import TensorStructInfo
from ..vm import VirtualMachine, build
from ..analysis import well_formed


@tvm.transform.module_pass(opt_level=3, name="SetupTrainer")
class SetupTrainer:
    """Setup a module."""

    PREDICT_FUNC_NAME: str = "predict"
    LOSS_FUNC_NAME: str = "loss"
    ADJOINT_FUNC_NAME: str = "loss_adjoint"
    UPDATE_PARAMS_FUNC_NAME: str = "update_params"

    PARAMS_NUM_ATTR_KEY: str = "params_num"
    OPTIM_STATE_ATTR_KEY: str = "optim_state"

    def __init__(self):
        self._loss_config: Optional[Dict[str, Any]] = None
        self._optim_config: Optional[Dict[str, Any]] = None

    @classmethod
    def _check_backbone_validity(cls, mod: IRModule):
        if not well_formed(mod):
            raise ValueError("Backbone Invalid: The backbone is not well formed.")
        ret_sinfo = mod[cls.PREDICT_FUNC_NAME].body.body.struct_info
        if not isinstance(ret_sinfo, TensorStructInfo):
            raise ValueError(
                "Backbone Invalid: The predict function is expected to have a single Tensor "
                "return value, which serves as the prediction result of the module. But got ",
                ret_sinfo,
            )

    def set_loss(self, loss: Loss, *call_args: TensorStructInfo) -> "SetupTrainer":
        """Specify the loss function.

        Parameters
        ----------
        loss : Loss
            The loss function. It will be appended to the backbone function using
            relax.training.utils.append_loss.

        call_args : TensorStructInfo
            The struct info a.k.a. the arguments to call the loss function.

        Returns
        -------
        self : SetupTrainer
            Return itself to support fluent interface style.
        """
        self._loss_config = {"loss": loss, "call_args": call_args}
        return self

    def set_optimizer(
        self, optim_type: Type[Optimizer], *init_args: Any, **init_kwargs: Any
    ) -> "SetupTrainer":
        """Specify the optimizer for training.

        Parameters
        ----------
        optim_type : Type[Optimizer]
            The specified optimizer class.

        init_args : Any
            Positional arguments passed to the optimize constructor.

        init_kwargs : Any
            Keyword arguments passed to the optimize constructor.

        Returns
        -------
        self : SetupTrainer
            Return itself to support fluent interface style.

        Notes
        -----
        SetupTrainer will set param_list of the optimzer automatically. So you only need
        to pass the rest initial arguments to it.
        """
        self._optim_config = {
            "optim_type": optim_type,
            "init_args": init_args,
            "init_kwargs": init_kwargs,
        }
        return self

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        """Setup the trainer in 3 steps.
        1. Prepare Loss.
        2. Gradient pass.
        3. Build Optimizer.
        4. Legalize operators.
        """

        self._check_backbone_validity(mod)
        # Step 1: Prepare Loss.
        if self._loss_config is None:
            raise TVMError("Setup Error: Please call 'set_loss' first before you transform.")

        predict_with_loss = append_loss(
            mod[self.PREDICT_FUNC_NAME],
            self._loss_config["loss"](*self._loss_config["call_args"]),
        )
        mod[self.LOSS_FUNC_NAME] = predict_with_loss

        # Step 2: Gradient pass.
        params_num = int(mod[self.PREDICT_FUNC_NAME].attrs[self.PARAMS_NUM_ATTR_KEY])
        require_grads = mod[self.LOSS_FUNC_NAME].params[:params_num]
        mod = Gradient(mod.get_global_var(self.LOSS_FUNC_NAME), require_grads=require_grads)(mod)

        # Step 3: Build Optimizer.
        if self._optim_config is None:
            raise TVMError("Setup Error: Please call `set_optimizer` first before you setup.")
        optimizer = self._optim_config["optim_type"](
            param_list=list(mod[self.ADJOINT_FUNC_NAME].params)[:params_num],
            *self._optim_config["init_args"],
            **self._optim_config["init_kwargs"],
        )
        mod[self.UPDATE_PARAMS_FUNC_NAME] = optimizer.get_function().with_attr(
            self.OPTIM_STATE_ATTR_KEY, optimizer.state
        )

        # Step 4: Legalize operators.
        return LegalizeOps()(mod)  # type: ignore


class Trainer:
    r"""Unified wrapper for relax training.

    Usage: Initialize it first, do some settings and then train.

    Parameters
    ----------
    backbone : IRModule
        Backbone of the module to be trained. It should be a relax module with a function
        whose name is `func_name`.

    parameters_indices : Union[int, List[int]],
        The indices of parameters in the input list of the function.

    func_name : str
        The name of the forward function. The function should return a single Tensor
        which is the output of the module.

    Examples
    --------

    >>> trainer = Trainer(MLP, [1, 2], "main")
    >>> trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    >>> trainer.set_vm_config(target="llvm")
    >>> trainer.set_optimizer(optim_type=SGD, lr=0.001).setup()
    >>> trainer.setup()
    >>> trainer.rand_init_params()
    >>> trainer.forward(*fwd_inputs)
    >>> trainer.backward(*bwd_inputs)
    """

    def __init__(
        self,
        backbone: IRModule,
        params_num: int,
        setup_trainer_pass: SetupTrainer,
    ) -> None:
        # Function Names
        self._predict_fn = setup_trainer_pass.PREDICT_FUNC_NAME
        self._adjoint_fn = setup_trainer_pass.ADJOINT_FUNC_NAME
        self._update_params_fn = setup_trainer_pass.UPDATE_PARAMS_FUNC_NAME

        # Compile & Build
        backbone[self._predict_fn] = backbone[self._predict_fn].with_attr(
            setup_trainer_pass.PARAMS_NUM_ATTR_KEY, params_num
        )
        self._mod = setup_trainer_pass(backbone)  # type: ignore
        self._vm: Optional[VirtualMachine] = None
        self._optim_state = self._mod[self._update_params_fn].attrs[
            setup_trainer_pass.OPTIM_STATE_ATTR_KEY
        ]

        # Allocate Parameters
        self._parameters_buffer: List[NDArray] = []
        self._parameters_name_to_pos: Dict[str, int] = {}

        def _convert_from_tvm_shape(tvm_shape):
            return [int(dim) for dim in tvm_shape]

        self._parameters_buffer = []
        for i, param in enumerate(self._mod[setup_trainer_pass.ADJOINT_FUNC_NAME].params):
            if i < params_num:
                self._parameters_buffer.append(
                    tvm.nd.array(
                        np.zeros(
                            shape=_convert_from_tvm_shape(param.struct_info.shape),
                            dtype=np.dtype(param.struct_info.dtype),
                        )
                    )
                )
                self._parameters_name_to_pos[param.name_hint] = i

    def build(
        self,
        target: Union[str, tvm.target.Target],
        device: Union[tvm.runtime.Device, List[tvm.runtime.Device]] = tvm.cpu(0),
        memory_cfg: Optional[str] = None,
    ):
        """Specify the following vm config: target, device, memory_cfg.

        Parameters
        ----------
        target : Union[str, tvm.target.Target]
            The target to run all modules on.

        device : Union[Device, List[Device]]
            The device, or devices on which to execute the VM code.

        memory_cfg : Optional[str]
            The allocator behavior to use for the VM.
        """
        ex = build(self._mod, target=target)
        self._vm = VirtualMachine(ex, device=device, memory_cfg=memory_cfg)

    def _check_build(self):
        if self._vm is None:
            raise TVMError("Please build the trainer first. Use `trainer.build(...)`.")

    @property
    def mod(self) -> IRModule:
        """Return the differentiated IRModule.

        Returns
        -------
        ret : IRModule
            The differentiated IRModule.
        """
        return self._mod

    @property
    def vm(self) -> VirtualMachine:
        """Return the relax virtual machine of the module.

        Returns
        -------
        ret : VirtualMachine
            The relax virtual machine.
        """
        self._check_build()
        return self._vm

    def rand_init_params(self) -> "Trainer":
        """Randomly initialize parameters using np.random.uniform.

        Returns
        -------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._parameters_buffer = [
            tvm.nd.array(
                np.sqrt(6.0 / np.sum(v.shape))
                * np.random.uniform(-1.0, 1.0, v.shape).astype(np.dtype(v.dtype))
            )
            for v in self._parameters_buffer
        ]
        return self

    def load_params(self, extern_param_dict: Dict[str, Union[np.ndarray, NDArray]]) -> "Trainer":
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.

        Parameters
        ----------
        extern_param_dict : Dict[str, Union[np.ndarray, NDArray]]
            The external parameters dict: param_name -> param_value. The param name should
            be the same as the name hint of corresponding relax Var.

        Returns
        -------
        self : Trainer
            Return itself to support fluent interface style.
        """
        for key, val in extern_param_dict.items():
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(val)
        return self

    def export_params(self) -> Dict[str, NDArray]:
        """Export parameters to a dict (parameter name -> NDArray).

        Returns
        -------
        exported_dict : Dict[str, NDArray]
            The exported dictionary of parameters.
        """
        ret: Dict[str, NDArray] = {}
        for key, pos in self._parameters_name_to_pos.items():
            ret[key] = self._parameters_buffer[pos]
        return ret

    def _prepare_inputs(self, func_name, inputs):
        input_len = len(self._mod[func_name].params)
        param_len = len(self._parameters_buffer)
        assert len(inputs) + param_len == input_len
        to_vm = []
        for i in range(param_len):
            to_vm.append(self._parameters_buffer[i])
        for i in range(input_len - param_len):
            to_vm.append(tvm.nd.array(inputs[i]))
        return to_vm

    def predict(self, *inputs: List[Union[np.ndarray, NDArray]]) -> NDArray:
        """Predict.

        Parameters
        ----------
        inputs: List[Union[np.ndarray, NDArray]]
            The necessary inputs of the module.

        Returns
        -------
        output : NDArray
            The output result of the forward process. Only support single return value now.

        Notes
        -----
        You don't need to provide parameters of the module in `inputs`. Instead, the trainer
        will maintain them interally. The inputs should be given in the order of the function
        argument list with skipping all parameters.
        """
        self._check_build()
        return self._vm[self._predict_fn](*self._prepare_inputs(self._predict_fn, inputs))

    def update_params(self, *inputs: List[Union[np.ndarray, NDArray]]) -> NDArray:
        """Calculate loss and update parameters. It will calculate the gradient of each
        parameter and update them using optimizer.

        Parameters
        ----------
        inputs: List[Union[np.ndarray, NDArray]]
            The necessary inputs of the module.

        Returns
        -------
        loss : NDArray
            The loss stored in Numpy array.

        Notes
        -----
        You don't need to provide parameters of the module in `inputs`. Instead, the trainer
        will maintain them interally. The inputs should be given in the order of the function
        argument list with skipping all parameters.
        """
        self._check_build()
        loss, grads = self._vm[self._adjoint_fn](*self._prepare_inputs(self._adjoint_fn, inputs))
        new_params, self._optim_state = self._vm[self._update_params_fn](
            tuple_object(self._parameters_buffer), grads, self._optim_state
        )
        self._parameters_buffer = [new_params[i] for i in range(len(new_params))]
        return loss
