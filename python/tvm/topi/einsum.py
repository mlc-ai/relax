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
import inspect
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

    Returns
    -------
    out : tvm.te.Tensor
        The calculation based on the Einstein summation convention.
    """

    def wrap_fcompute(fcompute):
        if fcompute is None:
            return None

        def wrapped_fcompute(arrayOfOperands):
            args = [arrayOfOperands[i] for i in range(len(arrayOfOperands))]

            body = fcompute(*args)
            if not isinstance(body, (list, tuple)):
                body = [body]
            body = convert(body)
            return body

        return wrapped_fcompute

    def wrap_fcombine(fcombine):
        if fcombine is None:
            return None

        def wrapped_fcombine(x, y):
            ret = fcombine(x, y)
            if not isinstance(ret, (list, tuple)):
                ret = [ret]
            ret = convert(ret)
            return ret

        return wrapped_fcombine

    def wrap_fidentity(fidentity):
        if fidentity is None:
            return None

        def wrapped_fidentity(arrayOfString):
            dtypes = [arrayOfString[i] for i in range(len(arrayOfString))]

            ret = fidentity(*dtypes)
            if not isinstance(ret, (list, tuple)):
                ret = [ret]
            ret = convert(ret)
            return ret

        return wrapped_fidentity

    # check fcombine and fid should be consisstent

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
