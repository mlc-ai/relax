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
from tvm import relax
from tvm import relax as rx
from tvm.runtime.container import ADT, tuple_object
from tvm.relax.op import add, subtract, multiply
from typing import Callable, Dict, List, Optional, Union
from tvm.relax import Var


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
        """Default initializer for relax.training.Optimizer.

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

        See also `relax.training.Optimizer`.
        """
        return self._state

    @state.setter
    def state(self, value):
        """Setter of state.

        Setter of state must be defined in any optimizer inheriting the Optimizer class to ensure
        the state can be updated.
        """
        self._state = value

    def get_function(self) -> relax.Function:
        """Use blockbuilder to build a new function that executes parameter and optimizer state update.

        Returns
        -------
        func: relax.Function

        Examples
        --------
        This is an example for the returned relax function.

        .. code-block:: python
            # You could assume adjoint function be like:
            # See also `relax.transform.SimpleAD`.
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


def _get_var_shape_list(var):
    """
    var should be DynTensorType.
    var.shape should be a ShapeExpr whose all fields are IntImm.
    """
    return [int(val) for val in var.shape]


def _float2constant(*f_list):
    return [relax.Constant(tvm.nd.array(np.array(f).astype(np.float32))) for f in f_list]


def _copy_var(var, name_suffix="", is_dataflow=True):
    if is_dataflow:
        return relax.DataflowVar(var.name_hint + name_suffix, var.shape, var.checked_type)
    else:
        return Var(var.name_hint + name_suffix, var.shape, var.checked_type)


class SGD(Optimizer):
    def __init__(self, param_list, lr, weight_decay=0):
        super().__init__(param_list)
        self.lr = lr
        self.weight_decay = weight_decay

    @property
    def state(self):
        if self._state is None:
            self._state = tuple_object((
                # num_steps = 0
                tvm.nd.array(np.zeros(()).astype(np.float32)),
            ))
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def get_function(self) -> relax.Function:
        var_len = len(self._param_list)

        # input variables
        param_var = Var("params", rx.Tuple([p.shape for p in self._param_list]),
            rx.TupleType([p.checked_type for p in self._param_list]))
        grad_var = Var("gradients", rx.Tuple([p.shape for p in self._param_list]),
            rx.TupleType([p.checked_type for p in self._param_list]))
        state_var = Var("optim_states", rx.Tuple([relax.ShapeExpr([])]),
            rx.TupleType([rx.DynTensorType(0, "float32")]))

        # constants
        lr, weight_decay, one = \
            _float2constant(self.lr, self.weight_decay, 1.0)

        bb = rx.BlockBuilder()
        with bb.function("SGD", [param_var, grad_var, state_var]):
            with bb.dataflow():
                # get variables in tuples
                param_var_list = [
                    bb.emit_var_binding(VarBinding(_copy_var(self._param_list[i]),
                                                         relax.TupleGetItem(param_var, i)))
                    for i in range(var_len)
                ]
                grad_var_list = [
                    bb.emit_var_binding(VarBinding(_copy_var(self._param_list[i], "_adjoint"),
                                                         relax.TupleGetItem(grad_var, i)))
                    for i in range(var_len)
                ]
                num_steps_var = relax.DataflowVar("num_steps", [], rx.DynTensorType(0, "float32"))
                state_var_list = [
                    bb.emit_var_binding(VarBinding(num_steps_var,
                                                         relax.TupleGetItem(state_var, 0)))
                ]

                # computation logic
                state_var_list[0] = bb.emit(add(state_var_list[0], one))
                for i in range(len(self._param_list)):
                    p, g = param_var_list[i], grad_var_list[i]
                    dp = bb.emit(add(multiply(weight_decay, p), g)) if self.weight_decay else g
                    p = bb.emit(subtract(p, multiply(lr, dp)))
                    param_var_list[i] = p

                # handle return values
                param_var_tuple = relax.Tuple(param_var_list)
                state_var_tuple = relax.Tuple(state_var_list)
                gv0, gv1 = bb.emit_output(param_var_tuple), bb.emit_output(state_var_tuple)
            bb.emit_func_output((gv0, gv1))
        return bb.get()["SGD"]


class MomentumSGD(Optimizer):
    def __init__(self, param_list, lr, momentum, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(param_list)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    @property
    def state(self):
        if self._state is None:
            self._state = tuple_object((
                # num_steps = 0
                tvm.nd.array(np.zeros(()).astype(np.float32)),
                # v_{param} is initialized to all zeros
                *(tvm.nd.array(np.zeros(_get_var_shape_list(p)).astype(np.float32)) for p in self._param_list)
            ))
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def get_function(self) -> relax.Function:
        var_len = len(self._param_list)

        # input variables
        param_var = Var("params", rx.Tuple([p.shape for p in self._param_list]),
            rx.TupleType([p.checked_type for p in self._param_list]))
        grad_var = Var("gradients", rx.Tuple([p.shape for p in self._param_list]),
            rx.TupleType([p.checked_type for p in self._param_list]))
        state_var = Var("optim_states",
            rx.Tuple([relax.ShapeExpr([])] + [p.shape for p in self._param_list]),
            rx.TupleType([rx.DynTensorType(0, "float32")] + [p.checked_type for p in self._param_list]))

        lr, momentum, weight_decay, dampening_inv, one = \
            _float2constant(self.lr, self.momentum, self.weight_decay, 1 - self.dampening, 1)

        bb = rx.BlockBuilder()
        with bb.function("MomentumSGD", [param_var, grad_var, state_var]):
            with bb.dataflow():
                # get variables in tuples
                param_var_list = [
                    bb.emit_var_binding(VarBinding(_copy_var(self._param_list[i]),
                                                         relax.TupleGetItem(param_var, i)))
                    for i in range(var_len)
                ]
                grad_var_list = [
                    bb.emit_var_binding(VarBinding(_copy_var(self._param_list[i], "_adjoint"),
                                                         relax.TupleGetItem(grad_var, i)))
                    for i in range(var_len)
                ]
                num_steps_var = relax.DataflowVar("num_steps", [], rx.DynTensorType(0, "float32"))
                state_var_list = \
                    [bb.emit_var_binding(VarBinding(num_steps_var,
                                                          relax.TupleGetItem(state_var, 0)))] + \
                    [bb.emit_var_binding(VarBinding(_copy_var(self._param_list[i], "_velocity"),
                                         relax.TupleGetItem(state_var, i + 1)))
                     for i in range(var_len)]

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
                param_var_tuple = relax.Tuple(param_var_list)
                state_var_tuple = relax.Tuple(state_var_list)
                gv0, gv1 = bb.emit_output(param_var_tuple), bb.emit_output(state_var_tuple)
            bb.emit_func_output((gv0, gv1))
        return bb.get()["MomentumSGD"]


class Adam(Optimizer):
    def __init__(self, param_list, lr, beta1, beta2, eps, normalize: bool):
        pass
