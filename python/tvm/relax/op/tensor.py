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
# pylint: disable=redefined-builtin
"""Basic tensor operations."""
import numpy as np  # type: ignore
import tvm

from . import _ffi_api
from ..expr import Expr


def add(lhs: Expr, rhs: Expr) -> Expr:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : Expr
        The left hand side input data
    rhs : Expr
        The right hand side input data

    Returns
    -------
    result : Expr
        The computed result.

    Examples
    --------
    .. code:: python

      x = relax.Var("a") # shape is [2, 3]
      y = relax.Var("b") # shape is [2, 1]
      z = relax.add(x, y)  # result shape is [2, 3]
    """
    return _ffi_api.add(lhs, rhs)


def subtract(lhs: Expr, rhs: Expr) -> Expr:
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relax.Expr
        The left hand side input data
    rhs : relax.Expr
        The right hand side input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.subtract(lhs, rhs)


def multiply(lhs: Expr, rhs: Expr) -> Expr:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : Expr
        The left hand side input data
    rhs : Expr
        The right hand side input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.multiply(lhs, rhs)


def ewise_fma(e1: Expr, e2: Expr, e3: Expr) -> Expr:
    return _ffi_api.ewise_fma(e1, e2, e3)  # type: ignore


def divide(lhs: Expr, rhs: Expr) -> Expr:
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relax.Expr
        The left hand side input data
    rhs : relax.Expr
        The right hand side input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.divide(lhs, rhs)


def floor_divide(lhs, rhs) -> Expr:
    """Floor division with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relax.Expr
        The left hand side input data
    rhs : relax.Expr
        The right hand side input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.floor_divide(lhs, rhs)


def negative(lhs: Expr) -> Expr:
    """Compute element-wise negative of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result
    """
    return _ffi_api.negative(lhs)


def sin(data: Expr) -> Expr:
    """Compute elementwise sin of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sin(data)


def cos(data: Expr) -> Expr:
    """Compute elementwise cos of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.cos(data)


def tanh(data: Expr) -> Expr:
    """Compute elementwise tanh of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.tanh(data)


def sqrt(data: Expr) -> Expr:
    """Compute elementwise square root of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sqrt(data)


def log(data: Expr) -> Expr:
    """Compute elementwise natural logarithm of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    ret : relax.Expr
        The computed result
    """
    return _ffi_api.log(data)


def sigmoid(data: Expr) -> Expr:
    """Compute elementwise sigmoid of data.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sigmoid(data)


def less(lhs: Expr, rhs: Expr) -> Expr:
    """Broadcasted elementwise test for (lhs < rhs).

    Parameters
    ----------
    lhs : relax.Expr
        The left hand side input data
    rhs : relax.Expr
        The right hand side input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.less(lhs, rhs)


def unique(
    data: Expr,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = -1,
) -> Expr:
    """Find the unique elements and the new index of each item in a given tensor.

    Parameters
    ----------
    data : Expr
        The input tensor.

    sorted: bool
        Whether to sort the unique elements in ascending order before
        returning as output.

    return_inverse: bool
        Whether to return an additional tensor with indices for where elements in
        the original input ended up in the returned unique list.

    return_counts: bool
        Whether to return an additional tensor with counts of each unique elements.

    dim: int
        The dimension to apply unique. If negative, the unique of the flattened input is returned.

    Returns
    -------
    ret: Expr
        The created relax call with
    """

    return _ffi_api.unique(data, sorted, return_inverse, return_counts, dim)  # type: ignore


@tvm.register_func("relax.run.unique")
def numpy_unique(
    a: tvm.nd.array,
    sort: int,
    return_inverse: int,
    return_counts: int,
    dim: int,
) -> tvm.nd.array:
    """Returns the unique elements of the input tensor.

    Uses numpy.unique to compute unique elements.
    """
    # TODO(prakalp): add support for returning a tuple when return_inverse or return_counts is True
    if bool(return_inverse) or bool(return_counts):
        raise NotImplementedError("missing support return_inverse or return_counts set to true")
    if dim < 0:
        dim = None
    a_numpy = a.numpy()
    # TODO(prakalp): use torch.unique instead of numpy when torch is installed in ci.
    output_sorted_numpy, indices = np.unique(a_numpy, return_index=True)
    if sort:
        return tvm.nd.array(output_sorted_numpy)
    output_numpy = [a_numpy.flatten()[index] for index in sorted(indices, reverse=True)]
    return tvm.nd.array(output_numpy)
