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
"""Loss functions library for relax."""

from typing import Any, List, Optional, Union
from tvm.script.parser import relax as R
from tvm.relax import Var, Function, StructInfo, BlockBuilder
from tvm import relax


def _create_param_var(param: Union[Var, StructInfo], param_name) -> Var:
    if isinstance(param, StructInfo):
        param = Var(param_name, param)
    assert isinstance(param, Var)
    return param


class Loss:
    """Base class of all loss.

    Parameters
    ----------
    """

    reduction: str
    loss_name: str

    def __init__(self, loss_name: str, reduction: str = "mean") -> None:
        self.loss_name = loss_name
        self.reduction = reduction

    def __call__(self) -> Function:
        raise NotImplementedError()


class L1Loss(Loss):
    """Mean element-wise absolute value difference.

    Parameters
    ----------
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(L1Loss, self).__init__("l1_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self.loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = bb.emit(R.subtract(logits, targets))
                if self.reduction == "none":
                    loss = bb.emit_output(R.abs(lv))  # TODO: R.abs
                else:
                    loss = bb.emit(R.abs(lv))
                    if self.reduction == "sum":
                        loss = bb.emit_output(R.sum(loss))
                    else:
                        # TODO: mean
                        pass
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)


class MSELoss(Loss):
    """Measures the element-wise mean squared error.

    Parameters
    ----------
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(MSELoss, self).__init__("mse_loss", reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self.loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = bb.emit(R.subtract(logits, targets))
                if self.reduction == "none":
                    loss = bb.emit_output(R.mutiply(lv, lv))
                else:
                    loss = bb.emit(R.mutiply(lv, lv))
                    if self.reduction == "sum":
                        loss = bb.emit_output(R.sum(loss))
                    else:
                        # TODO: mean
                        pass
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)


class CrossEntropyLoss(Loss):
    """CrossEntropyLoss.

    Parameters
    ----------
    """

    ignore_index: int

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        super(CrossEntropyLoss, self).__init__("cross_entropy_loss", reduction)
        self.ignore_index = ignore_index

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
        weights: Optional[Union[Var, StructInfo]],
    ) -> Function:
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")
        if weights:
            weights = _create_param_var(weights, "predictions")

        with bb.function(self.loss_name, [predictions, targets, weights]):
            with bb.dataflow():
                logits = bb.emit(R.nn.log_softmax(predictions))
                loss = bb.emit_output(
                    R.nn.nll_loss(logits, targets, weights, self.reduction, self.ignore_index)
                )
            bb.emit_func_output(loss)

        return bb.get()[self.loss_name].with_attr("global_symbol", self.loss_name)
