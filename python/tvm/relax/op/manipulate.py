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
"""Manipulation operators."""
from typing import List, Optional, Tuple, Union

from tvm.ir.expr import PrimExpr
from tvm.tir import IntImm

from . import _ffi_api
from ..expr import Expr, ShapeExpr, Tuple as RxTuple


PrimExprLike = Union[int, PrimExpr]


def reshape(data: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Reshape the input array.

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            data.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
            data.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
            data.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    shape : Union[Tuple[PrimExprLike], Expr]
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relax.Expr
        The reshaped result.

    Note
    ----
    The ``-1`` inference is only performed at compile-time.
    That is to say, in any case the dimension length of ``-1`` cannot be inferred in
    compile-time, an error will be thrown.
    """
    return _ffi_api.reshape(data, shape)  # type: ignore


def permute_dims(data: Expr, axes: Optional[List[int]] = None) -> Expr:
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
    return _ffi_api.permute_dims(data, axes)  # type: ignore


def expand_dims(data: Expr, axis: Union[int, List[int]]) -> Expr:
    """Insert new axes at the positions given by `axis`.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Union[int, List[int]]
        The axes at which the input array are expanded.
        All values are required to lie in range `[-data.ndim - 1, data.ndim]`, with the convention
        of negative indexing.

    Returns
    -------
    result : relax.Expr
        The transformed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.expand_dims(data, axis)  # type: ignore


def squeeze(data, axis: Optional[Union[int, List[int]]] = None) -> Expr:
    """Squeeze axes in the array.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Optional[Union[int, List[int]]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    Returns
    -------
    result : relax.Expr
        The squeezed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.squeeze(data, axis)  # type: ignore


def flatten(data: Expr) -> Expr:
    """Flatten all the tensor dimensions into one.

    Parameters
    ----------
    data : Expr
        The input data to the operator.

    Returns
    -------
    result : Expr
        The flattened result.
    """
    return _ffi_api.flatten(data)  # type: ignore


def concat(data: Union[Expr, List[Expr]], axis: Optional[int] = 0) -> Expr:
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    data : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing the tensors to be concatenated.

    axis : Optional[int]
        The axis along which the tensors are concatenated.
        If `axis` is `None`, the input tensor is required to be flattened before concatenation.

    Returns
    -------
    result: relax.Expr
        The concatenated tensor.
    """
    if isinstance(data, (list, tuple)):
        data = RxTuple(data)
    return _ffi_api.concat(data, axis)  # type: ignore


def split(data: Expr, indices_or_sections: Union[int, List[PrimExprLike]], axis: int = 0,) -> Expr:
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis (if possible). Last section will be smaller if the tensor
    size along the given dimension is not divisible by the integer.

    If indices_or_sections is a tuple of mixture of int or PrimExpr,
    the entries indicate the indices where along axis the array is split.

    Parameters
    ----------
    data : relax.Expr
        The tensor to be split.

    indices_or_sections : Union[int, List[PrimExprLike]]
        Indices or sections to split into. Accepts an int or a list.

    axis : int
        The axis over which to split.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    if isinstance(indices_or_sections, int):
        indices_or_sections = IntImm("int64", indices_or_sections)
    return _ffi_api.split(data, indices_or_sections, axis)  # type: ignore


def broadcast_to(data: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[Tuple[PrimExprLike], Expr]
        The target shape.

    Returns
    -------
    result : relax.Expr
        The broadcasted tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.broadcast_to(data, shape)  # type: ignore


def collapse_sum_like(data: Expr, collapse_target: Expr) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    collapse_target : relax.Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor after summation.
    """
    return _ffi_api.collapse_sum_like(data, collapse_target)


def collapse_sum_to(
    data: Expr, shape: Union[Tuple[PrimExprLike], Expr]
) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given `shape` and `data.shape` must follow broadcast rules.

    During computation, all axes of `data.shape` and `shape` are checked from right to left.
    For an axis, if it follows these rules, `data` will be summed over this axis:
    - the axis exists in `data.shape` but not in `shape`, or
    - the axis exists in `data.shape` and equals to 1 in `shape`.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[Tuple[PrimExprLike], relax.Expr]
        The shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor of the given shape after summation.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.collapse_sum_to(data, shape)
