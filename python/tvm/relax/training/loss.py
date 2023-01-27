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
# pylint: disable=redefined-builtin
"""Loss functions library for relax."""

from typing import Optional, Union
from tvm import relax
from ..expr import Expr, Var, Function, StructInfo

from ..op import abs, sum, mean, subtract, multiply
from ..op.nn import log_softmax, nll_loss


__all__ = ["L1Loss", "MSELoss", "CrossEntropyLoss"]


def _create_param_var(param: Union[Var, StructInfo], param_name) -> Var:
    if isinstance(param, StructInfo):
        param = Var(param_name, param)
    assert isinstance(param, Var)
    return param


class Loss:
    """Base class of all loss.

    Parameters
    ----------
    loss_name : str
        The name of the loss function.

    reduction : str
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    reduction: str
    loss_name: str

    def __init__(self, loss_name: str, reduction: str = "mean") -> None:
        self.loss_name = loss_name
        self.reduction = reduction

        valid_reductions = ["mean", "sum", "none"]

        if self.reduction not in valid_reductions:
            raise ValueError("Reduction can only be one of these values: ", valid_reductions)

    def __call__(self) -> Function:
        """Calling a loss will get its relax function.

        Usually it has some parameters with type Union[Var, StructInfo]. It means
        the necessary inputs of the loss function. If a struct info is given, it will
        construct a corresponding Var using the struct info; if a Var is given, it will
        directly use this Var as the param.

        Returns
        ----------
        The relax function of the loss with the loss name as its global symbol.
        """
        raise NotImplementedError()

    def _with_reduction(self, expr: Expr):
        """Add a reduction to the final loss.

        Parameters
        ----------
        expr : Expr
            The loss expr.
        """
        if self.reduction == "sum":
            expr = sum(expr)
        elif self.reduction == "mean":
            expr = mean(expr)
        else:
            assert self.reduction == "none"
        return expr


class L1Loss(Loss):
    """Mean element-wise absolute value difference.

    Parameters
    ----------
    reduction : str
        See the doc of Loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(L1Loss, self).__init__("l1_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        bb = relax.BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self.loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = abs(subtract(predictions, targets))
                loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)


class MSELoss(Loss):
    """Measures the element-wise mean squared error.

    Parameters
    ----------
    reduction : str
        See the doc of Loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(MSELoss, self).__init__("mse_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        bb = relax.BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self.loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = subtract(predictions, targets)
                lv = multiply(lv, lv)
                loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)


class CrossEntropyLoss(Loss):
    """CrossEntropyLoss.

    Parameters
    ----------
    reduction : str
        See the doc of Loss.

    weights : Optional[Union[Var, StructInfo]]
        a manual rescaling weight given to each class. It has to be a Tensor of size C.

    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
    """

    ignore_index: int

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weights: Optional[Union[Var, StructInfo]] = None,
    ) -> None:
        super(CrossEntropyLoss, self).__init__("cross_entropy_loss", reduction)
        self.ignore_index = ignore_index
        if weights:
            self.weights = _create_param_var(weights, "weights")
        else:
            self.weights = None

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        bb = relax.BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        arg_list = [predictions, targets]
        if self.weights:
            arg_list.append(self.weights)

        with bb.function(self.loss_name, arg_list):
            with bb.dataflow():
                logits = bb.emit(log_softmax(predictions))
                loss = bb.emit_output(
                    nll_loss(logits, targets, self.weights, self.reduction, self.ignore_index)
                )
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)
