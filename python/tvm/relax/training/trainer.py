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

from typing import Union, List, Type, Optional

import tvm
from tvm import relax, TVMError
from tvm.ir.module import IRModule
from tvm.ir.op import Op
from tvm.runtime.container import tuple_object


import numpy as np
import math

from ..transform import LegalizeOps, Gradient
from .loss import _Loss
from .utils import append_loss
from .optimizer import Optimizer
from ..struct_info import TensorStructInfo
from ..vm import VirtualMachine, build
from ..analysis import well_formed


class Trainer:
    """Simple wrapper for relax training.

    Examples
    --------
    Initialize it first, do some settings and then train.

    .. code-block:: python
        trainer = Trainer(backbone=MLP, func_name="main", parameters_indices=range(1, 5))
        trainer.prepare("relax.nn.softmax_cross_entropy", SGD(None, 0.01))
        trainer.set_vm_config(target="llvm", device=tvm.cpu())
        trainer.setup()
        trainer.rand_init_params()
        trainer.train(epoch=10, loader=loader, data_hook=_hook, show_detail=True)
    """

    def __init__(
        self,
        backbone: IRModule,
        parameters_indices: Union[int, List[int]],
        func_name: str = "main",
    ) -> None:
        """Default initializer for relax.training.Trainer.

        Parameters
        ----------
        backbone: IRModule
            Backbone of the training module. It should be a relax module with a function
            whose name is `func_name`

        parameters_indices:
            The indices of parameters in the input list of the function.

        func_name: str
            The name of the target function. The function should return the output of the module.
        """

        # Parameter
        if isinstance(parameters_indices, int):
            parameters_indices = [parameters_indices]
        self._parameters_indices = list(parameters_indices)
        self._parameters_buffer = []
        self._parameters_name_to_pos = {}

        # Configs
        self._loss_config = None
        self._optim_config = None
        self._vm_config = None

        # Constructor
        self._backbone = backbone
        self._func_name = func_name

        # Build
        self._train_func_name = ""
        self._optimizer = None
        self._vm = None
        self._mod = None

    def set_loss(self, loss: _Loss, *call_args) -> "Trainer":
        """Specify the loss function.

        Parameters
        ----------
        loss : _Loss
            The loss function. It will be extended to the backbone using
            relax.training.utils.append_loss automatically.

        *call_args:
            The struct info of all arguments of loss function.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._loss_config = {"loss": loss, "call_args": call_args}
        return self

    def set_optimizer(self, optim_type: Type[Optimizer], *args, **kwargs) -> "Trainer":
        """Specify the optimizer for training.

        Parameters
        ----------
        optim_type: Type[Optimizer]
            The specified optimizer class.

        *args, **kwargs:
            Arguments passed to the optimize constructor.

        Note
        ----------
        The Trainer will set param_list and dtype of the optimzer automatically. So you only need
        to pass the rest arguments to it.


        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._optim_config = {"optim_type": optim_type, "args": args, "kwargs": kwargs}
        return self

    def set_vm_config(
        self,
        target: Union[str, tvm.target.Target],
        device: Union[tvm.runtime.Device, List[tvm.runtime.Device]] = tvm.cpu(0),
        memory_cfg: Optional[str] = None,
    ) -> "Trainer":
        """Specify the following vm config: target, device, memory_cfg.

        Parameters
        ----------
        target : Union[str, tvm.target.Target]
            The target to run all modules on.

        device: Union[Device, List[Device]]
            The device, or devices on which to execute the VM code.

        memory_cfg: Optional[str]
            The allocator behavior to use for the VM.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._vm_config = {"target": target, "device": device, "memory_cfg": memory_cfg}
        return self

    def setup(self) -> "Trainer":
        """Setup the trainer in 4 steps.
        1. Prepare Loss.
        2. Gradient pass and Legalize.
        3. Allocate buffers for parameters.
        4. Build Optimizer and VM.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """

        # Step 1: Prepare Loss.
        if self._loss_config is None:
            raise TVMError("Setup Error: Please call 'set_loss' first before you setup.")

        forward_with_loss = append_loss(
            self._backbone[self._func_name],
            self._loss_config["loss"](*self._loss_config["call_args"]),
        )
        forward_with_loss_name = self._func_name + "_loss"
        self._backbone[forward_with_loss_name] = forward_with_loss

        # Step 2: Gradient pass and Legalize.
        require_grads = [
            self._backbone[forward_with_loss_name].params[index]
            for index in self._parameters_indices
        ]
        self._mod = Gradient(
            self._backbone.get_global_var(forward_with_loss_name),
            require_grads=require_grads,
        )(self._backbone)
        self._train_func_name = forward_with_loss_name + "_adjoint"

        lowered_mod = LegalizeOps()(self._mod)

        # Step 3: Allocate Buffer for Parameters.
        def _convert_from_tvm_shape(tvm_shape):
            return [int(dim) for dim in tvm_shape]

        param_list = []
        self._parameters_buffer = []
        for i, param in enumerate(self._mod[self._train_func_name].params):
            if i in self._parameters_indices:
                param_list.append(param)
                self._parameters_buffer.append(
                    tvm.nd.array(
                        np.zeros(
                            shape=_convert_from_tvm_shape(param.struct_info.shape),
                            dtype=np.dtype(param.struct_info.dtype),
                        )
                    )
                )
                self._parameters_name_to_pos[param.name_hint] = len(self._parameters_buffer) - 1

        # Step 4: Build Optimizer and VM
        if self._vm_config is None:
            raise TVMError("Setup Error: Please call 'set_vm_config' first before you setup.")
        ex = build(lowered_mod, target=self._vm_config["target"])
        self._vm = VirtualMachine(
            ex, device=self._vm_config["device"], memory_cfg=self._vm_config["memory_cfg"]
        )

        if self._optim_config is None:
            raise TVMError("Setup Error: Please call `set_optimizer` first before you setup.")
        self._optimizer = self._optim_config["optim_type"](
            param_list=param_list,
            *self._optim_config["args"],
            **self._optim_config["kwargs"],
        )
        self._optimizer.set_vm_config(self._vm_config["target"], self._vm_config["device"])

        # End.
        return self

    def _check_backbone_validity(self):
        if not well_formed(self._backbone):
            raise ValueError("Backbone Invalid: The backbone is not well formed.")
        ret_sinfo = self._backbone[self._func_name].body.body.struct_info
        if not isintance(ret_sinfo, TensorStructInfo):
            raise ValueError(
                "Backbone Invalid: The backbone function is expected to have a single Tensor \
                return value, which serves as the prediction result of the module. But got ",
                ret_sinfo,
            )

    def _check_setup(self):
        if self._vm is None:
            raise TVMError("Please setup the trainer first. Use `trainer.setup()`.")

    def rand_init_params(self) -> "Trainer":
        """Randomly initialize parameters using np.random.uniform.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._check_setup()
        self._parameters_buffer = [
            tvm.nd.array(
                math.sqrt(6.0 / np.sum(v.shape))
                * np.random.uniform(-1.0, 1.0, v.shape).astype(np.dtype(v.dtype))
            )
            for v in self._parameters_buffer
        ]

    def load_params(self, extern_param_dict: dict) -> "Trainer":
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.

        Parameters
        ----------
        extern_param_dict : dict
            The external parameters dict: param_name -> param_value. The param name should
            be the same as the name hint of corresponding relax Var.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._check_setup()
        for key in extern_param_dict:
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(
                extern_param_dict[key]
            )
        return self

    def _prepare_inputs(self, func_name, inputs):
        ptr_inputs = 0
        ptr_params = 0
        input_len = len(self._mod[func_name].params)
        assert len(inputs) + len(self._parameters_buffer) == input_len
        to_vm = []
        for i in range(input_len):
            if i in self._parameters_indices:
                to_vm.append(self._parameters_buffer[ptr_params])
                ptr_params += 1
            else:
                to_vm.append(tvm.nd.array(inputs[ptr_inputs]))
                ptr_inputs += 1
        return to_vm

    def forward(self, *inputs: "Unpack[np.array]"):
        """Forward process.

        Parameters
        ----------
        *inputs: Unpack[np.array]
            The necessary inputs of the module.

        Returns
        ----------
        output : np.array
            The output result of the forward process. Only support single return value now.

        Note
        ----------
        You don't need to provide parameters of the module in `inputs`. Instead, the trainer
        will maintain them interally. The inputs should be given in the order of the function
        argument list with skipping all parameters.
        """
        self._check_setup()
        return self._vm[self._func_name](*self._prepare_inputs(self._func_name, inputs)).numpy()

    def backward(self, *inputs: "Unpack[np.array]"):
        """Backward. It will calculate the gradient of each parameter and
        update them using optimizer.

        Parameters
        ----------
        *inputs: Unpack[np.array]
            The necessary inputs of the module.

        Returns
        ----------
        loss : np.array
            The loss.

        Note
        ----------
        You don't need to provide parameters of the module in `inputs`. Instead, the trainer
        will maintain them interally. The inputs should be given in the order of the function
        argument list with skipping all parameters.
        """
        self._check_setup()
        loss, grads = self._vm[self._train_func_name](
            *self._prepare_inputs(self._train_func_name, inputs)
        )
        if len(grads) != len(self._parameters_buffer):
            raise TVMError(
                "Internal error: the number of gradients is not matched with the number of parameters."
            )
        new_params = self._optimizer(tuple_object(self._parameters_buffer), grads)
        if len(new_params) != len(self._parameters_buffer):
            raise TVMError(
                "Internal error: the number of new params is not matched with the number of parameters."
            )
        self._parameters_buffer = [new_params[i] for i in range(len(new_params))]
        return loss.numpy()
