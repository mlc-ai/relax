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
"""Transform operators."""
from typing import List, Optional, Tuple, Union

import tvm
from tvm import relax
from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr

PrimExprLike = Union[int, PrimExpr]


def transpose(data: Expr, axes: Optional[List[int]] = None) -> Expr:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order, reverse order if not specified.

    Returns
    -------
    result : relax.Expr
        The transposed result.
    """

    if axes is not None:
        axes = list(axes)
    return _ffi_api.transpose(data, axes)


def reshape(
    data: Expr, newshape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
) -> Expr:
    """Reshape the input array.

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
            data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
            data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    newshape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
        The new shape. Should be compatible with the original shape.
    Returns
    -------
    result : relax.Expr
        The reshaped result.
    """
    if isinstance(newshape, (PrimExpr, int)):
        newshape = [newshape]
    if isinstance(newshape, (tuple, list)):
        temp_shape = []
        for shape in newshape:
            if isinstance(shape, PrimExpr):
                temp_shape.append(shape)
            elif isinstance(shape, int):
                temp_shape.append(tvm.tir.const(shape, "int32"))
            else:
                raise RuntimeError(
                    f"The input new shape of reshape operator contains unrecognized dimension {shape}"
                )
        newshape = relax.ShapeExpr(temp_shape)
    return _ffi_api.reshape(data, newshape)


def expand_dims(data: Expr, axis: Union[int, Tuple[int], List[int]]) -> Expr:
    """Insert new axes at the positions given by `axis`.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Union[int, Tuple[int], List[int]]
        The axis at which the input array is expanded.
        Should lie in range `[-data.ndim - 1, data.ndim]`.
        If `axis < 0`, it is the first axis inserted;
        If `axis >= 0`, it is the last axis inserted in Python's negative indexing.

    Returns
    -------
    result : relax.Expr
        The reshaped result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.expand_dims(data, axis)
