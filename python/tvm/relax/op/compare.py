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
# pylint: disable=redefined-builtin
"""Compare operators."""

from ..expr import Expr
from . import _ffi_api


def equal(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] == rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.equal(lhs, rhs)  # type: ignore # pylint: disable=no-member


def not_equal(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] != rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.not_equal(lhs, rhs)  # type: ignore # pylint: disable=no-member


def greater(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] > rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.greater(lhs, rhs)  # type: ignore # pylint: disable=no-member


def greater_equal(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] >= rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.greater_equal(lhs, rhs)  # type: ignore # pylint: disable=no-member


def less(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] < rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.less(lhs, rhs)  # type: ignore # pylint: disable=no-member


def less_equal(lhs: Expr, rhs: Expr) -> Expr:
    """Returns a boolean tensor where the ith element is true if lhs[i] <= rhs[i]
    which is consistent with numpy semantics.

    Parameters
    ----------
    lhs : Expr
        The left hand side tensor.

    rhs : Expr
        The right hand side tensor.

    Returns
    -------
    result : Expr
        The result tensor.
    """
    return _ffi_api.less_equal(lhs, rhs)  # type: ignore # pylint: disable=no-member
