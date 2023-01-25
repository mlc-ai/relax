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
"""Utility functions for Relax"""
from typing import List, Tuple, Union

from tvm.relax.expr import Expr, Function, ShapeExpr
from tvm.relax.expr import Tuple as rx_Tuple
from ..runtime import convert_to_object
from ..tir import PrimExpr
from . import _ffi_api


def metadata_partitioner(rx_txt: str) -> List[str]:
    """Extract Relax program and metadata section.

    Parameters
    ----------
    rx_txt : str
        The input relax text.

    Returns
    -------
    output : List[str]
        The result list of partitioned text, the first element
        is the relax program, and the second is metadata section.
    """
    partitions = []
    left_curly = 0
    meta_start = 0
    meta_end = 0
    for i, char in enumerate(rx_txt):
        if i < 0:
            raise ValueError("The program is invalid.")
        if char == "{":
            if meta_start == 0:
                meta_start = i
            left_curly += 1
        elif char == "}":
            left_curly -= 1
            if left_curly == 0:
                meta_end = i + 1
                break

    if meta_end == 0:
        raise ValueError("The metadata section was not found.")
    metadata = rx_txt[meta_start:meta_end]
    rx_program = rx_txt[meta_end:-1]

    partitions.append(rx_program)
    partitions.append(metadata)

    return partitions


def convert_to_expr(value: Union[PrimExpr, Expr, Tuple[PrimExpr, Expr]]) -> Expr:
    """Helper function to convert tuple to Expr."""
    if not isinstance(value, tuple):
        return convert_to_object(value)
    value = list(value)
    for i, v in enumerate(value):
        value[i] = convert_to_expr(v)
    if all([isinstance(f, PrimExpr) for f in value]):
        return ShapeExpr(value)
    elif all([isinstance(f, Expr) for f in value]):  # type: ignore
        return rx_Tuple(value)
    else:
        raise TypeError("Return types, with mixed PrimExpr and Relax Expr, is not supported.")


def copy_with_new_params(func: Function) -> Function:
    """Copy the given function. The parameters of the original function would be copied to
    satisfy the restriction in the well-formed check: any two functions cannot share the same
    parameter variable.

    Parameters
    ----------
    func : Function
        The relax function to copy.

    Returns
    -------
    ret : Function
        The copied function.
    """
    return _ffi_api.CopyWithNewParams(func)  # type: ignore


def extend_func(orig_func: Function, ex_func: Function) -> Function:
    """Extend a relax function by another given function. It will link orig_func with
    ex_func and return a new function.

    In detail, the result function has the arguments list of orig_func and the combination
    of their body, which passes the return values of orig_func as the arguments of ex_func. For
    those arguments of ex_func which are not mapped to some return values, they will be lifted and
    appended to the argument list of result function.

    This util can be replaced if we have Inline pass. It is equivalent to inline a tail call in some
    sense.

    Note: the return value of orig_func will be bound to DataflowVar. So it is a bad idea to use this
    util if the params of ex_func present in its R.output.

    Example:

    .. code-block:: python
        # Before.
            @R.function
            def func1(a, b):
                return a + b, a * b

            @R.function
            def func2(c, d, e):
                return d, c, c + e

        # After. func1_func2 = extend_func(orig_func=func1, ex_func=func2).
            @R.function
            def func1_func2(a, b, e):
                c = a + b
                d = a * b
                return d, c, c + e

    Parameters
    ----------
    orig_func : Function
        The function to be extended.

    ex_func : Function
        The function to be linked after the orig_func.

    Returns
    -------
    ret : Function
        The result function.
    """

    return _ffi_api.ExtendFunc(orig_func, ex_func)  # type: ignore
