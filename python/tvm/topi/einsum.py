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
# pylint: disable=invalid-name,consider-using-enumerate,redefined-outer-name
"""Einsum operator"""
from tvm.runtime import convert

from . import cpp


def einsum(subscripts, *operand, fcompute=None, fcombine=None, fidentity=None):
    """Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts : string
        Specifies the subscripts for summation as comma separated list of subscript labels.
        An implicit (classical Einstein summation) calculation is performed unless the
        explicit indicator ‘->’ is included as well as subscript labels of the precise
        output form.

    a_tuple : tuple of tvm.te.Tensor
        These are the Tensors for the operation.
        The only difference of einsum between in tvm and numpy is it needs an extra brackets
        for the tensors. For example, topi.einsum("ij, jk -> ik", (A, B)).

    fcompute : function(List[value] -> List[value])
        Specifies the computation expression of the innermost loop.

    fcombine : function(Expr, Expr -> Expr)
        Specifies the associative computation involved in constructing the commutative reduction.

    fidentity: function(List[str] -> List[Expr])
        Establishes the identity elements for the commutative reduction process.


    Returns
    -------
    out : tvm.te.Tensor
        The calculation based on the Einstein summation convention.
    """

    def wrap_fcompute(fcompute):
        if fcompute is None:
            return None

        # On the C++ side, fcompute is utilized with an input of Array<Var>,
        # and is expects to return Array<PrimExpr>.
        def wrapped_fcompute(array_var):
            args = [array_var[i] for i in range(len(array_var))]

            result = fcompute(*args)
            if not isinstance(result, (list, tuple)):
                result = [result]
            result = convert(result)
            return result

        return wrapped_fcompute

    # On the C++ side, fcompute is expects to return Array<PrimExpr>.
    def wrap_fcombine(fcombine):
        if fcombine is None:
            return None

        def wrapped_fcombine(x, y):
            result = fcombine(x, y)
            if not isinstance(result, (list, tuple)):
                result = [result]
            result = convert(result)
            return result

        return wrapped_fcombine

    # On the C++ side, fcompute is utilized with an input of Array<String>,
    # and is expects to return Array<PrimExpr>.
    def wrap_fidentity(fidentity):
        if fidentity is None:
            return None

        def wrapped_fidentity(array_dtype):
            dtypes = [array_dtype[i] for i in range(len(array_dtype))]

            result = fidentity(*dtypes)
            if not isinstance(result, (list, tuple)):
                result = [result]
            result = convert(result)
            return result

        return wrapped_fidentity

    result = cpp.einsum(
        subscripts,
        operand,
        wrap_fcompute(fcompute),
        wrap_fcombine(fcombine),
        wrap_fidentity(fidentity),
    )
    if len(result) == 1:
        result = result[0]
    return result
