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
"""Relax linear algebra operators"""
from typing import Optional, Union

from tvm import DataType

from . import _ffi_api
from ..expr import Expr


def matmul(a: Expr, b: Expr, out_dtype: Optional[Union[str, DataType]] = None) -> Expr:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions.

    The semantics and output shape deduction rule is specified as
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.

    Parameters
    ----------
    a : relax.Expr
        The left operand of the matmul.

    b : relax.Expr
        The right operand of the matmul.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the the same as input dtype.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.matmul(a, b, out_dtype)  # type: ignore
