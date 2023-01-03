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
    """General matrix multiplication of two tensors.

    (The below is copied from torch.matmul)
    The behavior depends on the dimensionality of the tensors as follows:
    * If both tensors are 1-dimensional, the dot product (scalar) is returned.
    * If both arguments are 2-dimensional, the matrix-matrix product is returned.
    * If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply. After the
      matrix multiply, the prepended dimension is removed.
    * If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    * If both arguments are at least 1-dimensional and at least one argument is N-dimensional
      (where N > 2), then a batched matrix multiply is returned. If the first argument is
      1-dimensional, a 1 is prepended to its dimension for the purpose of the batched
      matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is
      appended to its dimension for the purpose of the batched matrix multiple and remove
      after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
      broadcastable). For example, if `a` is a `(j, 1, n, n)` tensor and `b` is a `(k, n, n)`
      tensor, the result will be a `(j, k, n, n)` tensor.

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
