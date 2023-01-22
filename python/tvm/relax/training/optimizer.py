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
"""Provide abstraction for defining optimizers and a set of common optimizers."""

from typing import List, Union

import numpy as np

import tvm
from tvm import relax as rx
from tvm.runtime.container import tuple_object
from tvm.relax.op import add, subtract, multiply, divide, sqrt
from tvm.relax import Var, Function


class Optimizer:
    """Relax training optimizer. This class could generate relax Functions for optimizing specified
    parameters, and store the states used in the optimization process, such as momentum.

    Parameters
    ----------
    params : Union[Var, list[Var]]
        The parameter or the list of parameters to optimize.

        If params is None, it indicates params will be added later using add_params.
    """

    def __init__(self, params: Union[Var, List[Var]]) -> None:
        if params is None:
            params = []
        elif not isinstance(params, list):
            params = [params]
        if not all(isinstance(x, Var) for x in params):
            raise ValueError("Not all elements in argument params is Var")
        self._param_list = params
        self._state = None

    def add_params(self, params: Union[Var, list[Var]]):
        """Append one parameter or a list of new parameters to the optimizer.

        Parameters
        ----------
        params : Union[Var, list[Var]]
            The parameter or the list of parameters to append.

        Note
        ----
        This method can only be called before `opt.state` or `opt.get_function()`.
        """
        assert self._state is None, "Add parameter after the state is acquired"
        if not isinstance(params, list):
            params = [params]
        assert all(isinstance(x, Var) for x in params), "Not all elements in params are Vars"
        self._param_list += params

    @property
    def state(self):
        """Return the state of the optimizer. This should be used as the last argument of the
        function that `get_function()` returns, and its new value is returned as the last return
        value of this function.

        The state of an optimizer will be constructed when `opt.state` is called for the first time.
        Before that, you can freely add parameters using `opt.add_params()`.
        """
        return self._state

    @state.setter
    def state(self, value):
        """Setter of state.

        If `state` property is overloaded, `state` setter must be overloaded at the same time.
        """
        self._state = value

    def get_function(self) -> Function:
        """In any implementation of Optimizer, we will use blockbuilder to build a optimizer
        function that executes the update of parameters and the optimizer state.

        The optimizer function will take in a tuple of parameters, a tuple of gradients of
        parameters, and a tuple of optimizer states. It will return a tuple of updated parameters,
        and a tuple of optimizer states.

        Returns
        -------
        func : Function
            The optimizer function.

        Examples
        --------
        An example of the returned optimizer function. This function executes the stochastic
        gradient descent method with lr = 0.1.

        .. code-block:: python
            @R.function
            def SGD(
                params: R.Tuple(R.Tensor((3, 3), dtype="float32")),
                gradients: R.Tuple(R.Tensor((3, 3), dtype="float32")),
                optim_states: R.Tuple(R.Tensor((), dtype="int64")),
            ) -> R.Tuple(
                R.Tuple(R.Tensor((3, 3), dtype="float32")), R.Tuple(R.Tensor((), dtype="int64"))
            ):
                with R.dataflow():
                    lv3: R.Tensor((3, 3), dtype="float32") = params[0]
                    lv12: R.Tensor((3, 3), dtype="float32") = gradients[0]
                    lv21: R.Tensor((), dtype="int64") = optim_states[0]
                    lv31: R.Tensor((), dtype="int64") = R.add(lv21, R.const(1, "int64"))
                    lv4: R.Tensor((3, 3), dtype="float32") = R.multiply(
                        R.const(0.1, "float32"), lv12
                    )
                    lv5: R.Tensor((3, 3), dtype="float32") = R.subtract(lv3, lv4)
                    gv1: R.Tuple(R.Tensor((3, 3), dtype="float32")) = (lv5,)
                    gv11: R.Tuple(R.Tensor((), dtype="int64")) = (lv31,)
                    R.output(gv1, gv11)
                return (gv1, gv11)
        """
        raise NotImplementedError()


def _get_np_shape(var):
    return [int(val) for val in var.struct_info.shape]


def _get_np_dtype(var):
    return str(var.struct_info.dtype)


class SGD(Optimizer):
    """Implements stochastic gradient descent.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def SGD(param_tuple, grad_tuple, state_tuple):
            num_steps = state_tuple[0]
            state_tuple[0] = num_steps + 1
            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                param_tuple[i] = param - lr * (grad + weight_decay * param)
            return param_tuple, state_tuple

    Parameters
    ----------
    lr : float
        learning rate

    weight_decay : Optional[float]
        weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, param_list, lr, weight_decay=0):
        super().__init__(param_list)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def state(self):
        """The state of SGD is `(num_steps,)`."""
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
        lr = rx.const(self.lr, "float32")
        weight_decay = rx.const(self.weight_decay, "float32")
        one = rx.const(1, "int64")

        builder = rx.BlockBuilder()
        with builder.function("SGD", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_var_list = [
                    builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)
                ]
                grad_var_list = [
                    builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)
                ]
                state_var_list = [builder.emit(rx.TupleGetItem(state_var, 0))]

                # computation logic
                state_var_list[0] = builder.emit(add(state_var_list[0], one))
                for i in range(len(self._param_list)):
                    p, g = param_var_list[i], grad_var_list[i]
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g))
                    p = builder.emit(subtract(p, multiply(lr, g)))
                    param_var_list[i] = p

                # handle return values
                param_var_tuple = rx.Tuple(param_var_list)
                state_var_tuple = rx.Tuple(state_var_list)
                gv0, gv1 = builder.emit_output(param_var_tuple), builder.emit_output(
                    state_var_tuple
                )
            builder.emit_func_output((gv0, gv1))
        return builder.get()["SGD"]


class MomentumSGD(Optimizer):
    """Implements stochastic gradient descent with momentum. Optionally supports Nesterov
    momentum.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def np_func(param_tuple, grad_tuple, state_tuple):
            num_steps = state_tuple[0]
            state_tuple[0] = num_steps + 1

            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                velocity = state_tuple[i + 1]
                grad = param * weight_decay + grad
                velocity = momentum * velocity + grad * (1 - dampening)
                if nesterov:
                    param = param - (grad + momentum * velocity) * lr
                else:
                    param = param - velocity * lr
                param_tuple[i] = param
                state_tuple[i + 1] = velocity

            return param_tuple, state_tuple

    Parameters
    ----------
    lr : float
        learning rate

    momentum : float
        momentum factor (default: 0)

    weight_decay : Optional[float]
        weight decay (L2 penalty) (default: 0)

    dampening : Optional[float]
        dampening for momentum (default: 0)

    nesterov : Optional[bool]
        enables Nesterov momentum (default: False)
    """

    def __init__(self, param_list, lr, momentum, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(param_list)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.dampening = float(dampening)
        self.nesterov = nesterov

    @property
    def state(self):
        """The state of momentum SGD:
        `(num_steps, velocity_of_param_0, ..., velocity_of_param_n-1)`
        """
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps = 0
                    tvm.nd.array(np.zeros((), "int64")),
                    # v_{param} is initialized to all zeros
                    *(
                        tvm.nd.array(np.zeros(_get_np_shape(p), _get_np_dtype(p)))
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
        state_var = Var(
            "optim_states",
            rx.TupleStructInfo([rx.TensorStructInfo((), "int64"), *(p.struct_info for p in plist)]),
        )

        # constants
        lr = rx.const(self.lr, "float32")
        momentum = rx.const(self.momentum, "float32")
        weight_decay = rx.const(self.weight_decay, "float32")
        dampening_inv = rx.const(1.0 - self.dampening, "float32")
        one = rx.const(1, "int64")

        builder = rx.BlockBuilder()
        with builder.function("MomentumSGD", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_var_list = [
                    builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)
                ]
                grad_var_list = [
                    builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)
                ]
                state_var_list = [
                    builder.emit(rx.TupleGetItem(state_var, i)) for i in range(len_param + 1)
                ]

                state_var_list[0] = builder.emit(add(state_var_list[0], one))
                for i in range(len(self._param_list)):
                    p, g, v = param_var_list[i], grad_var_list[i], state_var_list[i + 1]
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g))
                    damp_g = multiply(dampening_inv, g) if self.dampening else g
                    v = builder.emit(add(multiply(momentum, v), damp_g))
                    g_new = builder.emit(add(g, multiply(momentum, v))) if self.nesterov else v
                    p = builder.emit(subtract(p, multiply(lr, g_new)))
                    param_var_list[i] = p
                    state_var_list[i + 1] = v

                # handle return values
                param_var_tuple = rx.Tuple(param_var_list)
                state_var_tuple = rx.Tuple(state_var_list)
                gv0, gv1 = builder.emit_output(param_var_tuple), builder.emit_output(
                    state_var_tuple
                )
            builder.emit_func_output((gv0, gv1))
        return builder.get()["MomentumSGD"]


class Adam(Optimizer):
    """Implements Adam optimization algorithm.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def np_func(param_tuple, grad_tuple, state_tuple):
            state_tuple[0] += 1
            state_tuple[1] *= betas[0]
            state_tuple[2] *= betas[1]
            num_steps = state_tuple[0]

            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                m = state_tuple[i + 3]
                v = state_tuple[i + 3 + len(param_tuple)]
                grad = grad + weight_decay * param
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad * grad
                m_hat = m / (1 - betas[0] ** num_steps)
                v_hat = v / (1 - betas[1] ** num_steps)
                param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
                param_tuple[i] = param
                state_tuple[i + 3] = m
                state_tuple[i + 3 + len(param_tuple)] = v

            return param_tuple, state_tuple

    Parameters
    ----------
    lr : float
        learning rate

    betas : Optional[Tuple[float, float]]
        coefficients used for computing running averages of gradient and its square
        (default: (0.9, 0.999))

    eps : Optional[float]
        term added to the denominator to improve numerical stability (default: 1e-8)

    weight_decay : Optional[float]
        weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, param_list, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super().__init__(param_list)
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    @property
    def state(self):
        """The state of Adam:

        .. code-block:: python
            (num_steps, beta_0_prod, # beta0 ** num_steps
            beta_1_prod, # beta1 ** num_steps
            first_momentum_of_param_0, ..., first_momentum_of_param_n-1,
            second_momentum_of_param_0, ..., second_momentum_of_param_n-1)
        """
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps, beta_0_prod, beta_1_prod
                    tvm.nd.array(np.zeros((), "int64")),
                    tvm.nd.array(np.ones((), "float32")),
                    tvm.nd.array(np.ones((), "float32")),
                    # first_momentum
                    *(
                        tvm.nd.array(np.zeros(_get_np_shape(p), _get_np_dtype(p)))
                        for p in self._param_list
                    ),
                    # second_momentum
                    *(
                        tvm.nd.array(np.zeros(_get_np_shape(p), _get_np_dtype(p)))
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
        state_var = Var(
            "optim_states",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo((), "int64"),
                    rx.TensorStructInfo((), "float32"),
                    rx.TensorStructInfo((), "float32"),
                    *(p.struct_info for p in plist),
                    *(p.struct_info for p in plist),
                ]
            ),
        )

        # constants
        lr = rx.const(self.lr, "float32")
        beta1 = rx.const(self.beta1, "float32")
        beta2 = rx.const(self.beta2, "float32")
        beta1_inv = rx.const(1 - self.beta1, "float32")
        beta2_inv = rx.const(1 - self.beta2, "float32")
        eps = rx.const(self.eps, "float32")
        weight_decay = rx.const(self.weight_decay, "float32")
        one_int = rx.const(1, "int64")
        one_float = rx.const(1, "float32")

        builder = rx.BlockBuilder()
        with builder.function("Adam", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_var_list = [
                    builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)
                ]
                grad_var_list = [
                    builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)
                ]
                state_var_list = [
                    builder.emit(rx.TupleGetItem(state_var, i)) for i in range(len_param * 2 + 3)
                ]

                state_var_list[0] = builder.emit(add(state_var_list[0], one_int))
                state_var_list[1] = builder.emit(multiply(state_var_list[1], beta1))
                state_var_list[2] = builder.emit(multiply(state_var_list[2], beta2))

                for i in range(len(self._param_list)):
                    p, g, m, v = (
                        param_var_list[i],
                        grad_var_list[i],
                        state_var_list[i + 3],
                        state_var_list[i + 3 + len_param],
                    )
                    g = builder.emit(add(multiply(weight_decay, p), g)) if self.weight_decay else g
                    m = builder.emit(add(multiply(beta1, m), multiply(beta1_inv, g)))
                    v = builder.emit(add(multiply(beta2, v), multiply(beta2_inv, multiply(g, g))))
                    m_hat = builder.emit(divide(m, subtract(one_float, state_var_list[1])))
                    v_hat = builder.emit(divide(v, subtract(one_float, state_var_list[2])))
                    p = builder.emit(
                        subtract(p, multiply(lr, divide(m_hat, add(sqrt(v_hat), eps))))
                    )
                    param_var_list[i] = p
                    state_var_list[i + 3] = m
                    state_var_list[i + 3 + len_param] = v

                # handle return values
                gv0 = builder.emit_output(rx.Tuple(param_var_list))
                gv1 = builder.emit_output(rx.Tuple(state_var_list))
            builder.emit_func_output((gv0, gv1))
        return builder.get()["Adam"]


# An example
# import numpy as np
# import pytest
# import tvm
# from tvm.relax.block_builder import BlockBuilder
# from tvm.runtime.container import tuple_object
# import tvm.script
# from tvm import relax
# from tvm.script.parser import ir as I, relax as R, tir as T
# from tvm.relax.transform import LegalizeOps


# # Define the input Vars and the forward function "main"
# x = relax.Var("x", R.Tensor((3, 3), "float32"))
# y = relax.Var("y", R.Tensor((3, 3), "float32"))

# builder = BlockBuilder()
# with builder.function("main", [x, y]):
#     with builder.dataflow():
#         lv = builder.emit(R.subtract(x, y))
#         lv1 = builder.emit(R.multiply(lv, lv))
#         gv = builder.emit_output(R.sum(lv1))
#     builder.emit_func_output(gv)
# mod = builder.get()

# # AD process, differentiate "main" and generate a new function "main_adjoint"
# mod = relax.transform.Gradient(mod.get_global_var("main"), x)(mod)

# # Optimizer function generation
# # Note that `opt.state` would be used later to get the state of the optimizer
# opt = relax.optimizer.SGD(x, 0.1)
# mod["SGD"] = opt.get_function()

# # Show the complete IRModule
# mod.show()

# # Build and legalize module
# lowered_mod = LegalizeOps()(mod)
# ex = relax.vm.build(lowered_mod, target="llvm")
# vm = relax.VirtualMachine(ex, tvm.cpu())

# # Runtime inputs
# x_input = tvm.nd.array(np.random.rand(3, 3).astype(np.float32))
# x_input_tuple = tuple_object([x_input])
# y_input = tvm.nd.array(np.zeros((3, 3), "float32"))

# # Training process
# steps = 100
# for i in range(steps):
#     res, x_grad = vm["main_adjoint"](*x_input_tuple, y_input)
#     x_input_tuple, opt.state = vm["SGD"](x_input_tuple, x_grad, opt.state)
#     print("Step:", i)
#     print("loss =", res.numpy())
#     print("x =", x_input_tuple[0].numpy(), "\n")
