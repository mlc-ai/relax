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


def _convert_shape_to_expr(input_shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]) -> Expr:
    if isinstance(input_shape, (PrimExpr, int)):
        input_shape = [input_shape]
    if isinstance(input_shape, (tuple, list)):
        temp_shape = []
        for shape in input_shape:
            if isinstance(shape, PrimExpr):
                temp_shape.append(shape)
            elif isinstance(shape, int):
                temp_shape.append(tvm.tir.const(shape, "int64"))
            else:
                raise RuntimeError(
                    f"The input shape contains unrecognized dimension {shape}"
                )
        input_shape = relax.ShapeExpr(temp_shape)
    return input_shape


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
    newshape = _convert_shape_to_expr(newshape)
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


def squeeze(data, axis: Optional[Union[int, List[int], Tuple[int]]] = None) -> Expr:
    """Squeeze axes in the array.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Optional[Union[int, List[int], Tuple[int]]]
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
    return _ffi_api.squeeze(data, axis)


def concatenate(data: Union[Expr, List[Expr], Tuple[Expr]], axis: Optional[int] = 0) -> Expr:
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    data : Union[Expr, List[relax.Expr], Tuple[relax.Expr]]
        A list of tensors.
    axis : Optional[int] = 0
        The axis along which the tensors are concatenated.
        If `axis` is `None`, arrays must be flattened before concatenation.

    Returns
    -------
    result: relax.Expr
        The concatenated tensor.
    """
    if isinstance(data, (list, tuple)):
        data = relax.Tuple(data)
    return _ffi_api.concatenate(data, axis)


def cumsum(data: Expr, axis: Optional[int] = None) -> Expr:
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1,2,3], [4,5,6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]
    """
    return _ffi_api.cumsum(data, axis)


def trilu(data: Expr, k: int = 0, is_upper: bool = True) -> Expr:
    """
    Given a 2-D matrix or batches of 2-D matrices, returns the
    upper or lower triangular part of the tensor.

    Parameters
    ----------
    data: relax.Expr
        The tensor that trilu will be applied to. Must be either
        a 2D matrix or a tensor of batches of 2D matrices.

    k: int
        The number of diagonals above or below the main diagonal
        to exclude or include.

    is_upper: bool
        If True, only upper triangular values of input are kept,
        if False, the lower triangular values are kept.

    Returns
    -------
    ret : relax.Expr
        The new tensor with appropriate diagonals set to zero.

    Examples
    --------
    .. code-block:: python

        x = [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]

        relay.trilu(x, 0, True) =
            [[0, 1, 2],
             [0, 4, 5],
             [0, 0, 8]]
    """
    return _ffi_api.trilu(data, k, is_upper)


def cast(data: Expr, dtype: Union[str, tvm.DataType]) -> Expr:
    """Cast input tensor to data type.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    dtype: Union[str, tvm.DataType]
        The target data type

    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    if isinstance(dtype, str):
        dtype = tvm.DataType(dtype)
    return _ffi_api.cast(data, dtype)


def wrap_param(data: Expr, dtype: Union[str, tvm.DataType] = "float32") -> Expr:
    """Cast input tensor which is model param to data type if the dtype of the input data is not
    the same as the given dtype.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    dtype: Union[str, tvm.DataType]
        The target data type

    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    assert isinstance(data, relax.Constant)
    if data.data.dtype == dtype:
        return data
    if isinstance(dtype, str):
        dtype = tvm.DataType(dtype)
    return _ffi_api.wrap_param(data, dtype)


def take(
    data: Expr, indices: Expr, axis: Optional[int] = None, batch_dims: int = 0, mode: str = "clip"
) -> Expr:
    """Take elements from an array along an axis.

    Parameters
    ----------
    data : relax.Expr
        The source array.

    indices : relax.Expr
        The indices of the values to extract.

    axis : Optional[int]
        The axis over which to select values. By default,
        the flattened input array is used.

    batch_dims : int
        The number of batch dimensions. By default is 0.

    mode : str, optional
        Specifies how out-of-bound indices will behave [clip, wrap, fast].
        clip: clip to the range (default).
        wrap: wrap around the indices.
        fast: no clip or wrap around (user must make sure indices are in-bound).

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    return _ffi_api.take(data, indices, axis, batch_dims, mode)


def full(
    fill_value: Expr,
    shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr],
    dtype: Optional[Union[str, tvm.DataType]] = None,
) -> Expr:
    """Fill array with scalar value.

    Parameters
    ----------
    fill_value : relax.Expr
        The value to fill. Must be a scalar.

    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], relax.Expr]
        The shape of the target.

    dtype : Optional[Union[str, tvm.DataType]]
        The data type of the target.
        If dtype is not given, the dtype of the target would be the dtype of fill_value.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _convert_shape_to_expr(shape)

    if isinstance(dtype, str):
        dtype = tvm.DataType(dtype)
    return _ffi_api.full(fill_value, shape, dtype)


def full_like(data: Expr, fill_value: Expr) -> Expr:
    """Return a scalar value array of the same shape and type as data, filled with fill_value.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    fill_value : relax.Expr
        The scalar value to fill.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full_like(data, fill_value)


def ones(
    shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr],
    dtype: Optional[Union[str, tvm.DataType]] = "float32"
) -> Expr:
    """Fill array of given shape and dtype with ones.

    Parameters
    ----------
    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
        The shape of the target.

    dtype : Optional[Union[str, tvm.DataType]]
        The data type of the target.
        If dtype is not given, the dtype of the target would be float32.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _convert_shape_to_expr(shape)
    if isinstance(dtype, str):
        dtype = tvm.DataType(dtype)
    return _ffi_api.ones(shape, dtype)


def zeros(
    shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr],
    dtype: Optional[Union[str, tvm.DataType]] = "float32"
) -> Expr:
    """Fill array of given shape and dtype with zeros.

    Parameters
    ----------
    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
        The shape of the target.

    dtype : Optional[Union[str, tvm.DataType]]
        The data type of the target.
        If dtype is not given, the dtype of the target would be float32.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _convert_shape_to_expr(shape)
    if isinstance(dtype, str):
        dtype = tvm.DataType(dtype)
    return _ffi_api.zeros(shape, dtype)


def ones_like(data: Expr) -> Expr:
    """Return a scalar value array of the same shape and type as data, filled with ones.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.ones_like(data)


def zeros_like(data: Expr) -> Expr:
    """Return a scalar value array of the same shape and type as data, filled with zeros.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.zeros_like(data)


def split(
    data: Expr,
    indices_or_sections: Union[int, List[PrimExprLike], Tuple[PrimExprLike]],
    axis: int = 0,
) -> Expr:
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis. If such a split is not possible, an error is raised.

    If indices_or_sections is a tuple of mixture of int or PrimExpr,
    the entries indicate the indices where along axis the array is split.

    Parameters
    ----------
    data : relax.Expr
        The source array.

    indices_or_sections : Union[int, Tuple[PrimExprLike]]
        Indices or sections to split into. Accepts an int or a tuple

    axis : int
        The axis over which to split.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    if isinstance(indices_or_sections, (tuple, list)):
        indices = []
        for idx in indices_or_sections:
            if isinstance(idx, PrimExpr):
                indices.append(idx)
            elif isinstance(idx, int):
                indices.append(tvm.tir.const(idx, "int64"))
            else:
                raise RuntimeError(
                    f'The input indices of split operator contains unrecognized index "{idx}"'
                )
        indices_or_sections = indices
    elif isinstance(indices_or_sections, int):
        indices_or_sections = tvm.tir.IntImm("int64", indices_or_sections)
    else:
        raise RuntimeError(
            f"The input `indices_or_sections` has unrecognized type {type(indices_or_sections)}"
        )
    return _ffi_api.split(data, indices_or_sections, axis)


def broadcast_to(
    data: Expr, shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
) -> Expr:
    """Return a scalar value array with the same type, broadcast to
    the provided shape.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
        Provide the shape to broadcast to.

    Returns
    -------
    result : relay.Expr
        The result tensor.
    """
    shape = _convert_shape_to_expr(shape)
    return _ffi_api.broadcast_to(data, shape)


def strided_slice(
    data: Expr,
    begin: Union[List[PrimExprLike], Tuple[PrimExprLike]],
    end: Union[List[PrimExprLike], Tuple[PrimExprLike]],
    strides: Optional[Union[List[PrimExprLike], Tuple[PrimExprLike]]] = None,
    axes: Optional[Union[List[int], Tuple[int]]] = None,
    slice_mode: str = "end",
) -> Expr:
    """Strided slice of an array.

    Parameters
    ----------
    data : relax.Expr
        The source array to be sliced.

    begin : Union[List[PrimExprLike], Tuple[PrimExprLike]],
        The indices to begin with in the slicing.

    end : Union[List[PrimExprLike], Tuple[PrimExprLike]]
        Indices indicating end of the slice.

    strides : Optional[Union[List[PrimExprLike], Tuple[PrimExprLike]]]
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    axes : Optional[Union[List[int], Tuple[int]]]
        Axes along which slicing is applied. When it is specified, the length of begin, end,
        strides, and axes must be equal.

    slice_mode : str
        The slice mode [end, size].
        end: The ending indices for the slice [default].
        size: The input strides will be ignored, input end in this mode indicates
        the size of a slice starting at the location specified by begin. If end[i]
        is -1, all remaining elements in that dimension are included in the slice.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """

    def convert_int(arr):
        res = []
        for x in arr:
            if isinstance(x, PrimExpr):
                res.append(x)
            elif isinstance(x, int):
                res.append(tvm.tir.const(x, "int64"))
            else:
                raise RuntimeError(
                    f"The input of strided_slice operator contains unrecognized value {x}"
                )
        return res

    begin = convert_int(begin)
    end = convert_int(end)
    strides = convert_int(strides) if strides else None
    return _ffi_api.strided_slice(data, begin, end, strides, axes, slice_mode)


def collapse_sum_like(data: Expr, collapse_target: Expr) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : Expr
        The input tensor.

    collapse_target : Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : Expr
        The result tensor after summation.
    """
    return _ffi_api.collapse_sum_like(data, collapse_target)


def collapse_sum_to(
    data: Expr,
    shape: Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given shape and data.shape must follow broadcast rules.

    During computation, the axes of data.shape and shape are checked from right to left. For every
    axis, if it either:
    - exist in data but not in collapse_target, or
    - is larger than 1 in data and equals to 1 in collapse_target,

    data will be summed over this axis.

    Parameters
    ----------
    data : Expr
        The input tensor.

    shape : Union[PrimExprLike, List[PrimExprLike], Tuple[PrimExprLike], Expr]
        The shape to collapse to.

    Returns
    -------
    result : Expr
        The result tensor after summation.
    """
    shape = _convert_shape_to_expr(shape)
    return _ffi_api.collapse_sum_to(data, shape)
