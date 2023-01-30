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

from typing import Union, List, Type
import tvm
from tvm import relax
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
        func_name: str,
        parameters_indices: Union[int, List[int]],
        dtype: str = "float32",
    ) -> None:
        """Default initializer for relax.training.Trainer.

        Parameters
        ----------
        backbone: IRModule
            Backbone of the training module. It should be a relax module with a function
            whose name is `func_name`.

        func_name: str
            The name of the target function. The function should return the output of the module.

        parameters_indices:
            The indices of parameters in the input list of the target function.

        dtype: str
            The dtype of all data here. For simplicity, we suppose all data uses the same dtype. It should be
            a string which can be used to initialize both numpy dtype and relax DynTensorType.
            By default it is float32.
        """

        # should be config after
        if isinstance(parameters_indices, int):
            parameters_indices = [parameters_indices]
        self._parameters_indices = list(parameters_indices)

        self._loss_config = None
        self._optim_config = None
        self._vm_config = None

        self._backbone = backbone
        self._func_name = func_name
        self._dtype = dtype

        self._parameters_buffer = []
        self._parameters_name_to_pos = {}

        self._train_func_name = ""
        self._optimizer = None
        self._vm = None
        self._mod = None

    def set_loss(
        self, loss: _Loss, loss_inputs: Union[TensorStructInfo, List[TensorStructInfo]]
    ) -> Trainer:
        """Specify the loss function.

        Parameters
        ----------
        loss : _Loss
            The loss function. It will be extended to the backbone using
            relax.training.utils.append_loss automatically.

        loss_inputs: Union[TensorStructInfo, List[TensorStructInfo]]
            The struct info of all arguments of loss function.

        Returns
        ----------
        self : Trainer
            Return itself to support fluent interface style.
        """
        self._loss_config = {"loss": loss, "loss_inputs": loss_inputs}
        return self

    def set_optimizer(self, optim_type: Type[Optimizer], *args, **kwargs) -> Trainer:
        """Specify the optimizer for training.

        Parameters
        ----------
        optim_type: Type[Optimizer]
            The specified optimizer class.

        args, kwargs:
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
        device: Union[tvm.runtime.Device, List[tvm.runtime.Device]] = tvm.cpu(),
        memory_cfg: Optional[str] = None,
    ) -> Trainer:
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

    def setup(self) -> None:
        """Setup the trainer in 4 steps.
        1. Prepare Loss.
        2. Gradient pass and Legalize.
        3. Allocate buffers for parameters.
        4. Build Optimizer and VM.
        """

        # Step 1: Prepare Loss.
        loss = relax.Var("loss", TensorStructInfo((), self._dtype))

        if self._loss_config is None:
            raise TVMError("Trainer Error: Please call 'set_loss' first before you setup")

        forward_with_loss = append_loss(
            self._backbone[self._func_name],
            self._loss_config["loss"](*self._loss_config["loss_inputs"]),
        )
        forward_with_loss_name = self._func_name + "_loss"
        self._backbone[forward_with_loss_name] = forward_with_loss

        # Step 2: Gradient pass and Legalize.
        require_grads = [
            self._backbone[forward_with_loss_name].params[index]
            for index in self._parameters_indices
        ]
        self._mod = Gradient(
            func=self._backbone.get_global_var(forward_with_loss_name),
            require_grads=require_grads,
        )(self._backbone)
        self._train_func_name = forward_with_loss_name + "_adjoint"

        lowered_mod = LegalizeOps()(self._mod)

        # Step 3: Allocate Buffer for Parameters.
        def _convert_from_tvm_shape(tvm_shape):
            return [int(dim) for dim in tvm_shape]

        param_list = []
        self._parameters_buffer = []
        for i in range(len(self._mod[self._train_func_name].params)):
            if i in self._parameters_indices:
                param = self._mod[self._train_func_name].params[i]
                param_list.append(param)
                self._parameters_buffer.append(
                    tvm.nd.array(
                        np.zeros(
                            shape=_convert_from_tvm_shape(param.shape),
                            dtype=np.dtype(self._dtype),
                        )
                    )
                )
                self._parameters_name_to_pos[param.name_hint] = len(self._parameters_buffer) - 1

        # Step 4: Build Optimizer and VM
        if self._optim_config is None:
            raise TVMError("Trainer Error: Please call 'set_optimizer' first before you setup")
        self._optimizer = self._optim_config["optim_type"](
            param_list=param_list,
            dtype=self._dtype,
            *self._optim_config["args"],
            **self._optim_config["kwargs"],
        )

        if self._vm_config is None:
            raise TVMError("Trainer Error: Please set vm_config first before you setup")
        ex = build(lowered_mod, target=self._vm_config["target"])
        self._vm = VirtualMachine(
            ex, device=self._vm_config["device"], memory_cfg=self._vm_config["memory_cfg"]
        )

    def _check_setup(self):
        if self._vm is None:
            raise TVMError("Trainer Error: Please setup first.")

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

    def forward(self, *inputs):
        """Forward. Return output in numpy."""
        self._check_setup()
        return self._vm[self._func_name](*self._prepare_inputs(self._func_name, inputs)).numpy()

    def backward(self, *inputs):
        """Backward. Return loss in numpy."""
        self._check_setup()
        loss, grads = self._vm[self._train_func_name](
            *self._prepare_inputs(self._train_func_name, inputs)
        )
        assert len(grads) == len(self._parameters_buffer)
        new_params, self._optimizer.state = self._vm[self._optimizer.__class__.__name__](
            tuple_object(self._parameters_buffer), grads, self._optimizer.state
        )
        assert len(new_params) == len(self._parameters_buffer)
        self._parameters_buffer = new_params
        return loss.numpy()

    def train(self, epoch: int, loader, data_hook=lambda x: x, show_detail=False):
        """A simple wrapper for the training loop.

        Parameters
        ----------
        epoch: int
            The number of the training epochs.

        loader:
            The data loader. It should be a iterable object with the input data returned every iteration.

        data_hook:
            A hook function which takes the return value of the loader iteration as input, and return things
            that you want to feed to the module.
            It is used to preprocess the input data. By default it is an identity function.

        show_detail: boolean
            Whether to show some information about training.
        """
        self._check_setup()
        for i in range(epoch):
            loss_buffer = []
            for dataline in loader:
                loss = self.backward(*data_hook(dataline))
                loss_buffer.append(loss)
            if show_detail:
                print(f"Train Epoch #{i}, Loss = {np.mean(loss_buffer)}")

    def rand_init_params(self):
        """Randomly initialize parameters using np.random.uniform."""
        self._check_setup()
        self._parameters_buffer = [
            tvm.nd.array(
                math.sqrt(6.0 / np.sum(v.shape))
                * np.random.uniform(-1.0, 1.0, v.shape).astype(np.float32)
            )
            for v in self._parameters_buffer
        ]

    def load_params(self, extern_param_dict: dict):
        """Load parameters from a dict.
        The key of the dict should be the same with the parameter name in backbone.
        """
        self._check_setup()
        for key in extern_param_dict:
            self._parameters_buffer[self._parameters_name_to_pos[key]] = tvm.nd.array(
                extern_param_dict[key]
            )
