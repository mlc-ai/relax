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
# pylint: disable=redefined-builtin, invalid-name
"""Loss functions library for relax."""

from typing import Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on

from tvm import relax
from ..expr import Expr, Var, Function, StructInfo

from ..op import abs, sum, mean, subtract, multiply
from ..op.nn import log_softmax, nll_loss


def _create_param_var(param: Union[Var, StructInfo], param_name: str) -> Var:
    if isinstance(param, StructInfo):
        param = Var(param_name, param)
    if not isinstance(param, Var):
        raise TypeError("The type of param should be Var or StructInfo, but got " + type(param))
    return Var(param.name_hint, param.struct_info)


class _Loss:
    r"""Base class of all loss.

    Parameters
    ----------
    loss_name : str
        The name of the loss function.

    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    def __init__(self, loss_name: str, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        self.loss_name = loss_name
        self.reduction = reduction

        valid_reductions = ["mean", "sum", "none"]

        if self.reduction not in valid_reductions:
            raise ValueError("Reduction can only be one of these values: ", valid_reductions)

    def _with_reduction(self, expr: Expr) -> Expr:
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


class L1Loss(_Loss):
    r"""Mean element-wise absolute value difference.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__("l1_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        """Get the relax function of L1Loss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.
        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        Returns
        ----------
        The relax function of L1Loss with the loss name as its global symbol.
        """
        bb = relax.BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self.loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = abs(subtract(predictions, targets))
                loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)


class MSELoss(_Loss):
    r"""Measures the element-wise mean squared error.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__("mse_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        """Get the relax function of MSELoss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.
        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        Returns
        ----------
        The relax function of MSELoss with the loss name as its global symbol.
        """
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


class CrossEntropyLoss(_Loss):
    r"""CrossEntropyLoss. It is a combination of a log_softmax computation and a nll_loss.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.

    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
    """

    ignore_index: int

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__("cross_entropy_loss", reduction)
        self.ignore_index = ignore_index

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
        weights: Optional[Union[Var, StructInfo]] = None,
    ) -> Function:
        """Get the relax function of CrossEntropyLoss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.

        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        weights : Optional[Union[Var, StructInfo]]
            a manual rescaling weight given to each class. It has to be a Tensor of size C.

        Returns
        ----------
        The relax function of CrossEntropyLoss with the loss name as its global symbol.
        """
        bb = relax.BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        arg_list = [predictions, targets]
        if weights:
            weights = _create_param_var(weights, "weights")
            arg_list.append(weights)

        with bb.function(self.loss_name, arg_list):
            with bb.dataflow():
                logits = bb.emit(log_softmax(predictions))
                loss = bb.emit_output(
                    nll_loss(logits, targets, weights, self.reduction, self.ignore_index)
                )
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)
