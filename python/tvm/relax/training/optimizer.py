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

from __future__ import annotations

import numpy as np

import tvm
import tvm.ir
from tvm import relax as rx
from tvm.runtime.container import tuple_object
from tvm.relax.op import add, subtract, multiply
from typing import List, Union
from tvm.relax import Var, Function


class Optimizer:
    """Relax training optimizer.

    Examples
    --------
    This is an example that uses SGD to optimize parameters.

    .. code-block:: python
        optimizer = Optimizer(None)
        params, gradient = None, None
        func = None
        params, optimizer.args = func(params, gradient, optimizer.args)
    """

    def __init__(self, params: Union[Var, List[Var]]) -> None:
        """Default initializer for rx.training.Optimizer.

        Parameters
        ----------
        params: Union[Var, list[Var]]
            The parameter or the list of parameters to optimize.

            If params is None, it indicates params will be added later using add_params.
        """
        if params is None:
            params = []
        elif not isinstance(params, list):
            params = [params]
        if not all(isinstance(x, Var) for x in params):
            raise ValueError("Not all elements in argument params is Var")
        self._param_list = params
        self._state = None

    def add_params(self, params: Union[Var, list[Var]]):
        """Add one parameter or a list of new parameters.

        Parameters
        ----------
        params: Union[Var, list[Var]]
            The parameter or the list of parameters to optimize.
        """
        assert self._state is None
        if not isinstance(params, list):
            params = [params]
        if not all(isinstance(x, Var) for x in params):
            raise ValueError("Not all elements in argument params is Var")
        self._param_list += params

    @property
    def state(self):
        """Return the state of the optimizer. This should be used the last argument of the function
        that `get_function()` returns, and updated by the last return value of that function.

        Here state is a property because self._state is constructed when its first used. Before that,
        you can freely add parameters using `add_params()`.

        See also `rx.training.Optimizer`.
        """
        return self._state

    @state.setter
    def state(self, value):
        """Setter of state.

        Setter of state must be defined in any optimizer inheriting the Optimizer class to ensure
        the state can be updated.
        """
        self._state = value

    def get_function(self) -> Function:
        """Use blockbuilder to build a new function that executes parameter and optimizer state update.

        Returns
        -------
        func: Function

        Examples
        --------
        This is an example for the returned relax function.

        .. code-block:: python
            # You could assume adjoint function be like:
            # See also `rx.transform.SimpleAD`.
            @R.function
            def main_adjoint(arg1: R.Tensor((1, 10), "float32"), arg2: R.Tensor((1, 10), "float32")):
                # some calculation...
                return (loss, (arg1_adjoint, arg2_adjoint))

            # The returned function should be like:
            @R.function
            def optimizer(params: R.Tuple(R.Tensor((1, 10), "float32"), R.Tensor((1, 10), "float32")),
                          gradients: R.Tuple(R.Tensor((1, 10), "float32"), R.Tensor((1, 10), "float32")),
                          optimizer_args):
                # some calculation...
                return (new_params, new_optimizer_args)
        """
        raise NotImplementedError()


def _get_shape_list(var):
    """
    var should be DynTensorType.
    var.shape should be a ShapeExpr whose all fields are IntImm.
    """
    return [int(val) for val in var.struct_info.shape]


class SGD(Optimizer):
    def __init__(self, param_list, lr, weight_decay=0):
        super().__init__(param_list)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def state(self):
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps = 0
                    tvm.nd.array(np.zeros((), "int64")),
                )
            )
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def get_function(self) -> Function:
        plist = self._param_list
        len_param = len(plist)

        # input variables
        param_var = Var("params", rx.TupleStructInfo([p.struct_info for p in plist]))
        grad_var = Var("gradients", rx.TupleStructInfo([p.struct_info for p in plist]))
        state_var = Var("optim_states", rx.TupleStructInfo([rx.TensorStructInfo((), "int64")]))

        # constants
        lr = rx.const(self.lr)
        weight_decay = rx.const(self.weight_decay)
        one = rx.const(1, "int64")

        bb = rx.BlockBuilder()
        with bb.function("SGD", [param_var, grad_var, state_var]):
            with bb.dataflow():
                # get variables in tuples
                param_var_list = [
                    bb.emit(rx.TupleGetItem(param_var, i))
                    for i in range(len_param)
                ]
                grad_var_list = [
                    bb.emit(rx.TupleGetItem(grad_var, i))
                    for i in range(len_param)
                ]
                state_var_list = [bb.emit(rx.TupleGetItem(state_var, 0))]

                # computation logic
                state_var_list[0] = bb.emit(add(state_var_list[0], one))
                for i in range(len(self._param_list)):
                    p, g = param_var_list[i], grad_var_list[i]
                    dp = bb.emit(add(multiply(weight_decay, p), g)) if self.weight_decay else g
                    p = bb.emit(subtract(p, multiply(lr, dp)))
                    param_var_list[i] = p

                # handle return values
                param_var_tuple = rx.Tuple(param_var_list)
                state_var_tuple = rx.Tuple(state_var_list)
                gv0, gv1 = bb.emit_output(param_var_tuple), bb.emit_output(state_var_tuple)
            bb.emit_func_output((gv0, gv1))
        return bb.get()["SGD"]


class MomentumSGD(Optimizer):
    def __init__(self, param_list, lr, momentum, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(param_list)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.dampening = float(dampening)
        self.nesterov = float(nesterov)

    @property
    def state(self):
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps = 0
                    tvm.nd.array(np.zeros((), "int64")),
                    # v_{param} is initialized to all zeros
                    *(
                        tvm.nd.array(np.zeros(_get_shape_list(p), "float32"))
                        for p in self._param_list
                    ),
                )
            )
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def get_function(self) -> Function:
        plist = self._param_list
        len_param = len(plist)

        # input variables
        param_var = Var("params", rx.TupleStructInfo([p.struct_info for p in plist]))
        grad_var = Var("gradients", rx.TupleStructInfo([p.struct_info for p in plist]))
        state_var = Var("optim_states", rx.TupleStructInfo([rx.TensorStructInfo((), "int64"), *(p.struct_info for p in plist)]))

        # constants
        lr = rx.const(self.lr)
        momentum = rx.const(self.momentum)
        weight_decay = rx.const(self.weight_decay)
        dampening_inv = rx.const(1 - self.dampening)
        one = rx.const(1, "int64")

        bb = rx.BlockBuilder()
        with bb.function("MomentumSGD", [param_var, grad_var, state_var]):
            with bb.dataflow():
                # get variables in tuples
                param_var_list = [
                    bb.emit(rx.TupleGetItem(param_var, i))
                    for i in range(len_param)
                ]
                grad_var_list = [
                    bb.emit(rx.TupleGetItem(grad_var, i))
                    for i in range(len_param)
                ]
                state_var_list = [
                    bb.emit(rx.TupleGetItem(state_var, 0)),
                    *(
                        bb.emit(rx.TupleGetItem(state_var, i + 1))
                        for i in range(len_param)
                    ),
                ]

                # computation logic:
                # num_steps = state_var_list[0]
                # num_steps += 1
                # for p, g, v in zip(param_var_list, grad_var_list, state_var_list[1:]):
                #     dp = p * weight_decay + g
                #     v = momentum * v + dp * (1 - dampening)
                #     if nesterov:
                #         p = p - (dp + momentum * v) * lr
                #     else:
                #         p = p - v * lr
                #     update p, v in var lists
                state_var_list[0] = bb.emit(add(state_var_list[0], one))
                for i in range(len(self._param_list)):
                    p, g, v = param_var_list[i], grad_var_list[i], state_var_list[i + 1]
                    dp = bb.emit(add(multiply(weight_decay, p), g)) if self.weight_decay else g
                    ddp = multiply(dampening_inv, dp) if self.dampening else dp
                    v = bb.emit(add(multiply(momentum, v), ddp))
                    g_new = bb.emit(add(dp, multiply(momentum, v))) if self.nesterov else v
                    p = bb.emit(subtract(p, multiply(lr, g_new)))
                    param_var_list[i] = p
                    state_var_list[i + 1] = v

                # handle return values
                param_var_tuple = rx.Tuple(param_var_list)
                state_var_tuple = rx.Tuple(state_var_list)
                gv0, gv1 = bb.emit_output(param_var_tuple), bb.emit_output(state_var_tuple)
            bb.emit_func_output((gv0, gv1))
        return bb.get()["MomentumSGD"]


class Adam(Optimizer):
    def __init__(self, param_list, lr, beta1, beta2, eps, normalize: bool):
        pass
