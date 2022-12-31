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
"""Creation operators."""
from typing import Optional, Tuple, Union

from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr, ShapeExpr

PrimExprLike = Union[int, PrimExpr]


def full(
    fill_value: Expr,
    shape: Union[PrimExprLike, Tuple[PrimExprLike], Expr],
    dtype: Optional[str] = None,
) -> Expr:
    """Fill array with scalar value.

    Parameters
    ----------
    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    shape : Union[PrimExprLike, Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : Optional[str]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full(fill_value, shape, dtype)  # type: ignore


def full_like(data: Expr, fill_value: Expr) -> Expr:
    """Construct a tensor such that
    - its shape and dtype is the same as the input data tensor's,
    - its value is filled with the input scalar fill value.

    Parameters
    ----------
    data : relax.Expr
        The input tensor, which provides the shape and dtype.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full_like(data, fill_value)  # type: ignore


def ones(shape: Union[PrimExprLike, Tuple[PrimExprLike], Expr], dtype: str) -> Expr:
    """Construct a tensor of all ones, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[PrimExprLike, Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : Optional[str]
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.ones(shape, dtype)  # type: ignore


def ones_like(data: Expr) -> Expr:
    """Construct a tensor with all ones, with shape and dtype of the input tensor shape.

    Parameters
    ----------
    data : relax.Expr
        The input tensor, which provides the shape and dtype.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.ones_like(data)  # type: ignore


def zeros(shape: Union[PrimExprLike, Tuple[PrimExprLike], Expr], dtype: str) -> Expr:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[PrimExprLike, Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : str
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.zeros(shape, dtype)  # type: ignore


def zeros_like(data: Expr) -> Expr:
    """Construct a tensor with all ones, with shape and dtype of the input tensor shape.

    Parameters
    ----------
    data : relax.Expr
        The input tensor, which provides the shape and dtype.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.zeros_like(data)  # type: ignore
