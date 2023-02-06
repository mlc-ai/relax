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
# pylint: disable=invalid-name
"""Unified Trainer API for relax training."""

from typing import Union, List, Optional, Dict
import numpy as np  # type: ignore

import tvm
from tvm import TVMError
from tvm.ir.module import IRModule
from tvm.runtime.container import tuple_object
from tvm.runtime.ndarray import NDArray

from .setup_trainer import SetupTrainer
from ..vm import VirtualMachine, build


class Trainer:
    r"""Unified wrapper for relax training. It uses SetupTrainer to setup the backbone and
    then compiles/builds/runs the module. Also it maintains the parameters internally.

    Parameters
    ----------
    backbone : IRModule
        Backbone of the module to be trained. It should be a relax module with a function
        which has name `predict` and returns a single Tensor.

    params_num : int,
        The numer of parameters. The first `params_num` inputs of the prediction function
        will be regarded as the parameters. It will be annotated into the prediction function
        as the value of func attr `update_params`.

    setup_trainer_pass : SetupTrainer
        The configured SetupTrainer pass.

    Examples
    --------
    >>> setup_trainer = SetupTrainer()
    >>> setup_trainer.set_loss(MSELoss(reduction="sum"), pred_sinfo, pred_sinfo)
    >>> setup_trainer.set_optimizer(optim_type=SGD, lr=0.001)
    >>> trainer = Trainer(MLP, 2, "main")
    >>> trainer.build(target="llvm")
    >>> trainer.xaiver_uniform_init_params()
    >>> trainer.predict(*predict_inputs)
    >>> trainer.update_params(*update_params_inputs)

    Notes
    -----
    The user should first build it then execute it.
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
        """Build and compile the module.

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
        """Return the IRModule transformed by the setup trainer pass.

        Returns
        -------
        ret : IRModule
            The IRModule transformed by the setup trainer pass.
        """
        return self._mod

    @property
    def vm(self) -> VirtualMachine:
        """Return the relax virtual machine of the module. Should first build the trainer.

        Returns
        -------
        ret : VirtualMachine
            The relax virtual machine.
        """
        self._check_build()
        return self._vm

    def xaiver_uniform_init_params(self):
        """Randomly initialize parameters using the method described in `Understanding the difficulty
        of training deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010)."""
        self._parameters_buffer = [
            tvm.nd.array(
                np.sqrt(6.0 / np.sum(v.shape))
                * np.random.uniform(-1.0, 1.0, v.shape).astype(np.dtype(v.dtype))
            )
            for v in self._parameters_buffer
        ]

    def load_params(self, extern_param_dict: Dict[str, Union[np.ndarray, NDArray]]):
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.

        Parameters
        ----------
        extern_param_dict : Dict[str, Union[np.ndarray, NDArray]]
            The external parameters dict: param_name -> param_value. The param name should
            be the same as the name hint of corresponding relax Var.
        """
        for key, val in extern_param_dict.items():
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(val)

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

    def predict(self, *inputs: Union[np.ndarray, NDArray]) -> NDArray:
        """Call the `predict` function and return the prediction result of the backbone.

        Parameters
        ----------
        *inputs : Union[np.ndarray, NDArray]
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

    def update_params(self, *inputs: Union[np.ndarray, NDArray]) -> NDArray:
        """Calculate loss and update parameters. It will calculate the gradients of parameters
        and update them using `update_params` function.

        Parameters
        ----------
        *inputs : Union[np.ndarray, NDArray]
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
