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

from decimal import Decimal
from typing import List, Optional, Union

import numpy as np  # type: ignore

import tvm
from tvm import relax as rx
from tvm.relax.struct_info import TensorStructInfo
from tvm.runtime.container import tuple_object
from tvm.relax.op import add, subtract, multiply, divide, sqrt
from tvm.relax import Var, Function


class Optimizer:
    """Relax training optimizer. This class could generate relax Functions for optimizing specified
    parameters, and store the states used in the optimization process, such as momentum.

    Parameters
    ----------
    params : Union[Var, List[Var]]
        The parameter or the list of parameters to optimize.

        Parameters should all be Vars of floating point Tensors, including float32, float64,
        float16, etc. Currently, all parameters should have the same dtype, and that dtype
        will be used as the dtype of the optimizer states.

    Examples
    --------
    The usage of optimizers should resemble the following pattern. We will take SGD as an example.
    For detailed examples, please see the tutorial.

    .. code-block:: python
        # Generate the optimizer function.
        # x is the relax Var we want to optimize
        opt = relax.optimizer.SGD(x, 0.1)
        mod["SGD"] = opt.get_function()

        # Backward process
        # vm is the module after legalization and building.
        # vm["main_adjoint"] is the backward function. See also relax.transform.Gradient.
        # param_tuple is the tuple of input parameters, label is the input that do not need
        # optimization
        loss, param_gradient = vm["main_adjoint"](*param_tuple, label)

        # Update parameters and optimizer states
        # vm["SGD"] is the optimizer function
        # opt.state is the state of the optimizer
        param_tuple, opt.state = vm["SGD"](param_tuple, param_gradient, opt.state)
    """

    _param_list: List[Var]
    _state: tvm.runtime.container.ADT
    _dtype: str

    def __init__(self, params: Union[Var, List[Var]]) -> None:
        if not isinstance(params, list):
            params = [params]
        self._state = None
        self._dtype = None
        self._check_params_and_dtype(params)
        self._param_list = params

    def _check_params_and_dtype(self, params: List[Var]) -> None:
        """Check params is legal and set the dtype of the optimizer."""
        params_set = set()
        for x in params:
            if not isinstance(x, Var):
                raise ValueError(f"Parameter {x} is not a Var")
            if not isinstance(x.struct_info, TensorStructInfo):
                raise ValueError(f"Only support Tensor parameters, but parameter {x.name_hint} has struct info {x.struct_info}")
            data_type = tvm.DataType(x.struct_info.dtype)
            if not data_type.type_code in (tvm.DataTypeCode.BFLOAT, tvm.DataTypeCode.FLOAT):
                raise ValueError(f"Only support Tensor parameters of floating point dtype, but parameter {x.name_hint} has struct info {x.struct_info}")
            if self._dtype is None:
                self._dtype = x.struct_info.dtype
            else:
                if self._dtype != x.struct_info.dtype:
                    raise ValueError("All parameters should have the same dtype, but parameter {x.name_hint} has dtype {x.struct_info.dtype}, which differs from previous dtype {self._dtype}")
            if x in params_set:
                raise ValueError("Parameter {x.name_hint} appears more than once")
            params_set.add(x)

    @property
    def state(self) -> tvm.runtime.container.ADT:
        """Return the state of the optimizer.

        The states of the optimizer can store information useful in the optimization process, such
        as the number of steps, the momentum in momentum SGD, etc.

        `opt.state` should be used as the last argument of the function that is got through
        `get_function()`, and its new value is returned as the last return value of that function.

        The state of an optimizer will be constructed when `opt.state` is called for the first time.

        Returns
        -------
        res : ADT
            An ADT object representing the state of the optimizer.
        """
        return self._state

    @state.setter
    def state(self, value: tvm.runtime.container.ADT) -> None:
        """Setter of state.

        If `state` property is overloaded, `state` setter must be overloaded at the same time.
        """
        self._state = value

    def get_function(self) -> Function:
        """In any implementation of Optimizer, we will use blockbuilder to build an optimizer
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
                    x: R.Tensor((3, 3), dtype="float32") = params[0]
                    x_grad: R.Tensor((3, 3), dtype="float32") = gradients[0]
                    num_steps: R.Tensor((), dtype="int64") = optim_states[0]
                    num_steps_new: R.Tensor((), dtype="int64") = R.add(
                        num_steps, R.const(1, "int64")
                    )
                    x_grad_lr: R.Tensor((3, 3), dtype="float32") = R.multiply(
                        R.const(0.1, "float32"), x_grad
                    )
                    x_new: R.Tensor((3, 3), dtype="float32") = R.subtract(x, x_grad_lr)
                    params_new: R.Tuple(R.Tensor((3, 3), dtype="float32")) = (x_new,)
                    optim_states_new: R.Tuple(R.Tensor((), dtype="int64")) = (num_steps_new,)
                    R.output(params_new, optim_states_new)
                return (params_new, optim_states_new)
        """
        raise NotImplementedError()


# TODO(chaofan, yixin): Support symbolic shapes
def _get_shape_as_int_list(var: Var) -> List[int]:
    return [int(val) for val in var.struct_info.shape]


# We need to subtract on hyperparameters, but do not want to introduce floating point error.
# Floating point error would lead to a few problems, such as making assert_structural_equal not
# passed in unit tests
def _high_precision_subtract(lhs: float, rhs: float) -> float:
    return float(Decimal(str(lhs)) - Decimal(str(rhs)))


class SGD(Optimizer):
    """Implements stochastic gradient descent.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def SGD(param_tuple, grad_tuple, state_tuple):
            # lr and weight_decay are float constant hyperparameters
            num_steps = state_tuple[0]
            state_tuple[0] = num_steps + 1
            for i in range(len(param_tuple)):
                param = param_tuple[i]
                grad = grad_tuple[i]
                param_tuple[i] = param - lr * (grad + weight_decay * param)
            return param_tuple, state_tuple

    Parameters
    ----------
    params : Union[Var, List[Var]]
        The parameter or the list of parameters to optimize.

        Parameters should all be Vars of floating point Tensors, including float32, float64,
        float16, etc. Currently, all parameters should have the same dtype, and that dtype
        will be used as the dtype of the optimizer states.

    lr : float
        learning rate

    weight_decay : Optional[float]
        weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, param_list: Union[Var, List[Var]], lr: float, weight_decay: Optional[float]=0) -> None:
        super().__init__(param_list)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def state(self) -> tvm.runtime.container.ADT:
        """The state of SGD is `(num_steps,)`.

        Returns
        -------
        res : ADT
            The state of SGD.
        """
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps = 0
                    tvm.nd.array(np.zeros((), "int64")),
                )
            )
        return self._state

    @state.setter
    def state(self, value: tvm.runtime.container.ADT) -> None:
        self._state = value

    def get_function(self) -> Function:
        plist = self._param_list
        len_param = len(plist)
        dtype = self._dtype

        # input variables
        param_var = Var("params", rx.TupleStructInfo([p.struct_info for p in plist]))
        grad_var = Var("gradients", rx.TupleStructInfo([p.struct_info for p in plist]))
        state_var = Var("optim_states", rx.TupleStructInfo([rx.TensorStructInfo((), "int64")]))

        # constants
        lr = rx.const(self.lr, dtype)
        weight_decay = rx.const(self.weight_decay, dtype)
        one = rx.const(1, "int64")

        builder = rx.BlockBuilder()
        with builder.function("SGD", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_list = [builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)]
                grad_list = [builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)]
                state_list = [builder.emit(rx.TupleGetItem(state_var, 0))]

                param_list_new, state_list_new = [], []
                # computation logic
                state_list_new.append(builder.emit(add(state_list[0], one)))
                for i in range(len_param):
                    p, g = param_list[i], grad_list[i]
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g))
                    p_new = builder.emit(subtract(p, multiply(lr, g)))
                    param_list_new.append(p_new)

                # handle return values
                gv0 = builder.emit_output(rx.Tuple(param_list_new))
                gv1 = builder.emit_output(rx.Tuple(state_list_new))
            builder.emit_func_output((gv0, gv1))
        return builder.get()["SGD"]


class MomentumSGD(Optimizer):
    """Implements stochastic gradient descent with momentum. Optionally supports Nesterov
    momentum.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def MomentumSGD(param_tuple, grad_tuple, state_tuple):
            # lr, momentum, weight_decay and dampening are float constant hyperparameters
            # nesterov is a boolean constant hyperparameter
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
    params : Union[Var, List[Var]]
        The parameter or the list of parameters to optimize.

        Parameters should all be Vars of floating point Tensors, including float32, float64,
        float16, etc. Currently, all parameters should have the same dtype, and that dtype
        will be used as the dtype of the optimizer states.

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

    def __init__(self, param_list: Union[Var, List[Var]], lr: float, momentum: float, dampening: Optional[float]=0, weight_decay: Optional[float]=0, nesterov: Optional[bool]=False) -> None:
        super().__init__(param_list)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.dampening = float(dampening)
        self.nesterov = nesterov

    @property
    def state(self) -> tvm.runtime.container.ADT:
        """The state of momentum SGD is
        `(num_steps, velocity_of_param_0, ..., velocity_of_param_n-1)`

        Returns
        -------
        res : ADT
            The state of momentum SGD.
        """
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps = 0
                    tvm.nd.array(np.zeros((), "int64")),
                    # v_{param} is initialized to all zeros
                    *(
                        tvm.nd.array(np.zeros(_get_shape_as_int_list(p), p.struct_info.dtype))
                        for p in self._param_list
                    ),
                )
            )
        return self._state

    @state.setter
    def state(self, new_value):
        self._state = new_value

    def get_function(self) -> Function:
        plist = self._param_list
        len_param = len(plist)
        dtype = self._dtype

        # input variables
        param_var = Var("params", rx.TupleStructInfo([p.struct_info for p in plist]))
        grad_var = Var("gradients", rx.TupleStructInfo([p.struct_info for p in plist]))
        state_var = Var(
            "optim_states",
            rx.TupleStructInfo([rx.TensorStructInfo((), "int64"), *(p.struct_info for p in plist)]),
        )

        # constants
        lr = rx.const(self.lr, dtype)
        momentum = rx.const(self.momentum, dtype)
        weight_decay = rx.const(self.weight_decay, dtype)
        dampening_inv = rx.const(_high_precision_subtract(1, self.dampening), dtype)
        one = rx.const(1, "int64")

        builder = rx.BlockBuilder()
        with builder.function("MomentumSGD", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_list = [builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)]
                grad_list = [builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)]
                state_list = [
                    builder.emit(rx.TupleGetItem(state_var, i)) for i in range(len_param + 1)
                ]

                param_list_new, state_list_new = [], []
                state_list_new.append(builder.emit(add(state_list[0], one)))
                for i in range(len_param):
                    p, g, v = param_list[i], grad_list[i], state_list[i + 1]
                    if self.weight_decay:
                        g = builder.emit(add(multiply(weight_decay, p), g))
                    damp_g = multiply(dampening_inv, g) if self.dampening else g
                    v_new = builder.emit(add(multiply(momentum, v), damp_g))
                    g_new = (
                        builder.emit(add(g, multiply(momentum, v_new))) if self.nesterov else v_new
                    )
                    p_new = builder.emit(subtract(p, multiply(lr, g_new)))
                    param_list_new.append(p_new)
                    state_list_new.append(v_new)

                # handle return values
                gv0 = builder.emit_output(rx.Tuple(param_list_new))
                gv1 = builder.emit_output(rx.Tuple(state_list_new))
            builder.emit_func_output((gv0, gv1))
        return builder.get()["MomentumSGD"]


class Adam(Optimizer):
    """Implements Adam optimization algorithm.

    The returned function is equivalent to the following numpy code:

    .. code-block:: python
        def Adam(param_tuple, grad_tuple, state_tuple):
            # lr, eps and weight_decay are float constant hyperparameters
            # betas is a tuple of two float constant hyperparameters
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
        """The state of Adam is

        .. code-block:: python
            (num_steps, beta_0_prod, # beta0 ** num_steps
            beta_1_prod, # beta1 ** num_steps
            first_momentum_of_param_0, ..., first_momentum_of_param_n-1,
            second_momentum_of_param_0, ..., second_momentum_of_param_n-1)

        Returns
        -------
        res : ADT
            The state of Adam.
        """
        if self._state is None:
            self._state = tuple_object(
                (
                    # num_steps, beta_0_prod, beta_1_prod
                    tvm.nd.array(np.zeros((), "int64")),
                    tvm.nd.array(np.ones((), self._dtype)),
                    tvm.nd.array(np.ones((), self._dtype)),
                    # first_momentum
                    *(
                        tvm.nd.array(np.zeros(_get_shape_as_int_list(p), p.struct_info.dtype))
                        for p in self._param_list
                    ),
                    # second_momentum
                    *(
                        tvm.nd.array(np.zeros(_get_shape_as_int_list(p), p.struct_info.dtype))
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
        dtype = self._dtype

        # input variables
        param_var = Var("params", rx.TupleStructInfo([p.struct_info for p in plist]))
        grad_var = Var("gradients", rx.TupleStructInfo([p.struct_info for p in plist]))
        state_var = Var(
            "optim_states",
            rx.TupleStructInfo(
                [
                    rx.TensorStructInfo((), "int64"),
                    rx.TensorStructInfo((), dtype),
                    rx.TensorStructInfo((), dtype),
                    *(p.struct_info for p in plist),
                    *(p.struct_info for p in plist),
                ]
            ),
        )

        # constants
        lr = rx.const(self.lr, dtype)
        beta1 = rx.const(self.beta1, dtype)
        beta2 = rx.const(self.beta2, dtype)
        beta1_inv = rx.const(_high_precision_subtract(1, self.beta1), dtype)
        beta2_inv = rx.const(_high_precision_subtract(1, self.beta2), dtype)
        eps = rx.const(self.eps, dtype)
        weight_decay = rx.const(self.weight_decay, dtype)
        one_int = rx.const(1, "int64")
        one_float = rx.const(1, dtype)

        builder = rx.BlockBuilder()
        with builder.function("Adam", [param_var, grad_var, state_var]):
            with builder.dataflow():
                # get variables in tuples
                param_list = [builder.emit(rx.TupleGetItem(param_var, i)) for i in range(len_param)]
                grad_list = [builder.emit(rx.TupleGetItem(grad_var, i)) for i in range(len_param)]
                state_list = [
                    builder.emit(rx.TupleGetItem(state_var, i)) for i in range(len_param * 2 + 3)
                ]

                param_list_new = []
                state_list_new = [None] * len(state_list)
                state_list_new[0] = builder.emit(add(state_list[0], one_int))
                state_list_new[1] = builder.emit(multiply(state_list[1], beta1))
                state_list_new[2] = builder.emit(multiply(state_list[2], beta2))

                for i in range(len_param):
                    p, g, m, v = (
                        param_list[i],
                        grad_list[i],
                        state_list[i + 3],
                        state_list[i + 3 + len_param],
                    )
                    g = builder.emit(add(multiply(weight_decay, p), g)) if self.weight_decay else g
                    m_new = builder.emit(add(multiply(beta1, m), multiply(beta1_inv, g)))
                    v_new = builder.emit(
                        add(multiply(beta2, v), multiply(beta2_inv, multiply(g, g)))
                    )
                    m_hat = builder.emit(divide(m_new, subtract(one_float, state_list_new[1])))
                    v_hat = builder.emit(divide(v_new, subtract(one_float, state_list_new[2])))
                    p_new = builder.emit(
                        subtract(p, multiply(lr, divide(m_hat, add(sqrt(v_hat), eps))))
                    )
                    param_list_new.append(p_new)
                    state_list_new[i + 3] = m_new
                    state_list_new[i + 3 + len_param] = v_new

                # handle return values
                gv0 = builder.emit_output(rx.Tuple(param_list_new))
                gv1 = builder.emit_output(rx.Tuple(state_list_new))
            builder.emit_func_output((gv0, gv1))
        return builder.get()["Adam"]
