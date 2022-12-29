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
from typing import Union

import tvm
from tvm import relax

from . import _ffi_api
from ..expr import Expr


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
