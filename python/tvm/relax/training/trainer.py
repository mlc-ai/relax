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
from tvm import relax, TVMError
from tvm.ir.module import IRModule
from tvm.runtime.ndarray import NDArray


class Trainer:
    r"""Unified wrapper for relax training. It uses SetupTrainer to setup the backbone and then
    builds and runs the module. Also it maintains the parameters and the model states internally.

    Parameters
    ----------
    backbone : IRModule
        Backbone of the module to be trained. It should be a relax module with a function which has
        name `predict` and returns a single Tensor.

    setup_trainer_pass : SetupTrainer
        The configured SetupTrainer pass.

    Examples
    --------
    .. code-block:: python
        setup_trainer = SetupTrainer(
            MSELoss(reduction="sum"),
            SGD(0.001),
            [pred_sinfo, target_sinfo],
        )

        trainer = Trainer(MLP, 2, setup_trainer)
        trainer.build(target="llvm")
        trainer.xaiver_uniform_init_params()
        trainer.predict(*predict_inputs)
        trainer.update_params(*update_params_inputs)

    Notes
    -----
    The user should first build it then execute it.
    """

    def __init__(
        self,
        train_mod: IRModule,
        zero_init_param_state: bool = True,
    ) -> None:
        # Function Handles
        self._predict_fn = train_mod.attrs["predict_fn"]
        self._adjoint_fn = train_mod.attrs["adjoint_fn"]
        self._update_params_fn = train_mod.attrs["update_params_fn"]

        # Compile & Build
        self._mod = train_mod  # type: ignore
        self._vm: Optional[relax.VirtualMachine] = None

        # Runtime values
        self._optim_state = self._mod.attrs["optim_state"]

        self._inputs_num = int(self._mod.attrs["inputs_num"])
        self._params_num = int(self._mod.attrs["params_num"])
        self._states_num = int(self._mod.attrs["states_num"])

        # used to initialize params and states
        self._param_vars = self._mod[self._adjoint_fn].params[
            self._inputs_num : self._inputs_num + self._params_num
        ]
        self._state_vars = self._mod[self._adjoint_fn].params[
            self._inputs_num
            + self._params_num : self._inputs_num
            + self._params_num
            + self._states_num
        ]

        self._params: List[Optional[NDArray]] = [None] * self._params_num
        self._param_name_to_pos: Dict[str, int] = {
            p.name_hint: i for i, p in enumerate(self._param_vars)
        }

        self._states: List[Optional[NDArray]] = [None] * self._states_num
        self._state_name_to_pos: Dict[str, int] = {
            s.name_hint: i for i, s in enumerate(self._state_vars)
        }

        if zero_init_param_state:
            self.zero_init_params()
            self.zero_init_states()

    @staticmethod
    def _shape_expr_to_int_list(tvm_shape):
        return [int(dim) for dim in tvm_shape]

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
        ex = relax.build(self._mod, target=target)
        self._vm = relax.VirtualMachine(ex, device=device, memory_cfg=memory_cfg)

    def _check_build(self):
        if self._vm is None:
            raise TVMError("Please build the trainer first. Use `trainer.build(...)`.")

        idx_not_inited_param = next((i for i, p in enumerate(self._params) if p is None), -1)
        if idx_not_inited_param != -1:
            raise TVMError(
                f"The {idx_not_inited_param}-th parameter is not initialized before training or "
                "inference."
            )

        idx_not_inited_state = next((i for i, s in enumerate(self._states) if s is None), -1)
        if idx_not_inited_state != -1:
            raise TVMError(
                f"The {idx_not_inited_state}-th model state is not initialized before training or "
                "inference."
            )

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
    def vm(self) -> relax.VirtualMachine:
        """Return the relax virtual machine of the module. Should first build the trainer.

        Returns
        -------
        ret : VirtualMachine
            The relax virtual machine.
        """
        if self._vm is None:
            raise TVMError("Please build the trainer first. Use `trainer.build(...)`.")
        return self._vm

    def xaiver_uniform_init_params(self):
        """Randomly initialize parameters using the method described in `Understanding the
        difficulty of training deep feedforward neural networks` - Glorot, X. & Bengio, Y.
        (2010)."""
        self._params = [
            tvm.nd.array(
                (np.sqrt(6.0 / np.sum(p.shape)) * np.random.uniform(-1.0, 1.0, p.shape)).astype(
                    p.dtype
                )
            )
            for p in self._param_vars
        ]

    def zero_init_params(self):
        self._params = [
            tvm.nd.array(
                np.zeros(self._shape_expr_to_int_list(p.struct_info.shape), p.struct_info.dtype)
            )
            for p in self._param_vars
        ]

    def zero_init_states(self):
        self._states = [
            tvm.nd.array(
                np.zeros(self._shape_expr_to_int_list(s.struct_info.shape), s.struct_info.dtype)
            )
            for s in self._state_vars
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
            self._params[self._param_name_to_pos[key]] = tvm.nd.array(val)

    def load_states(self, extern_state_dict: Dict[str, Union[np.ndarray, NDArray]]):
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.

        Parameters
        ----------
        extern_param_dict : Dict[str, Union[np.ndarray, NDArray]]
            The external parameters dict: param_name -> param_value. The param name should
            be the same as the name hint of corresponding relax Var.
        """
        for key, val in extern_state_dict.items():
            self._states[self._state_name_to_pos[key]] = tvm.nd.array(val)

    def export_params(self) -> Dict[str, NDArray]:
        """Export parameters to a dict (parameter name -> NDArray).

        Returns
        -------
        exported_dict : Dict[str, NDArray]
            The exported dictionary of parameters.
        """
        return {key: self._params[pos] for key, pos in self._param_name_to_pos.items()}

    def export_states(self) -> Dict[str, NDArray]:
        """Export parameters to a dict (parameter name -> NDArray).

        Returns
        -------
        exported_dict : Dict[str, NDArray]
            The exported dictionary of parameters.
        """
        return {key: self._states[pos] for key, pos in self._state_name_to_pos.items()}

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
        if len(inputs) != self._inputs_num:
            raise ValueError("The length of the input does not match the backbone")
        all_inputs = [tvm.nd.array(i) for i in inputs] + self._params + self._states
        return self._vm[self._predict_fn](*all_inputs)

    def update_params(
        self, inputs: List[Union[np.ndarray, NDArray]], targets: List[Union[np.ndarray, NDArray]]
    ) -> NDArray:
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

        # handle inputs
        if len(inputs) != self._inputs_num:
            raise ValueError("The length of the input does not match the backbone")
        all_inputs = (
            [tvm.nd.array(i) for i in inputs]
            + self._params
            + self._states
            + [tvm.nd.array(i) for i in targets]
        )
        ret, grads = self._vm[self._adjoint_fn](*all_inputs)

        # update model states
        if self._states_num != 0:
            self._states = list(ret[1:])
            ret = ret[0]

        # update params
        new_params, self._optim_state = self._vm[self._update_params_fn](
            self._params, grads, self._optim_state
        )
        self._params = list(new_params)

        # return the loss
        return ret
