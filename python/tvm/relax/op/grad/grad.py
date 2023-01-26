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
"""Operators to implement the gradients. Used in `_op_gradient.py`."""
from typing import List, Optional, Tuple, Union

from tvm import DataType

from . import _ffi_api
from ...expr import Expr


def nll_loss_backward(
    output_grad: Expr,
    predictions: Expr,
    targets: Expr,
    weights: Optional[Expr] = None,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Expr:
    """Backward operator of relax.nll_loss. All parameters except output_grad is the same as
    relax.nll_loss. Returns the gradient w.r.t. predictions.

    Parameters
    ----------
    output_grad : relax.Expr
      The gradient w.r.t. the forward operator nll_loss.

    Returns
    -------
    result : relax.Expr
      The gradient w.r.t. predictions.
    """
    return _ffi_api.nll_loss_backward(  # type: ignore
        output_grad, predictions, targets, weights, reduction, ignore_index
    )


def conv2d_backward_weight(
    output_grad: Expr,
    data: Expr,
    weight: Expr,
    strides: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: Optional[str] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    """ """
    return _ffi_api.conv2d_backward_weight(  # type: ignore
        output_grad,
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def max_pool2d_backward(
    output_grad: Expr,
    data: Expr,
    pool_size: Tuple[int, int] = (1, 1),
    strides: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    dilation: Tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    """ """
    return _ffi_api.max_pool2d_backward(  # type: ignore
        output_grad, data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout
    )
