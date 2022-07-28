#Licensed to the Apache Software Foundation(ASF) under one
# or more contributor license agreements.See the NOTICE file
#distributed with this work for additional information
#regarding copyright ownership.The ASF licenses this file
#to you under the Apache License, Version 2.0(the
#"License"); you may not use this file except in compliance
#with the License.You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing,
#software distributed under the License is distributed on an
#"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#KIND, either express or implied.See the License for the
#specific language governing permissions and limitations
"""Basic tensor operations."""
import numpy as np
import tvm

from . import _ffi_api
from ..expr import Expr


def add(lhs: Expr, rhs: Expr) -> Expr:
    return _ffi_api.add(lhs, rhs)


def multiply(lhs: Expr, rhs: Expr) -> Expr:
    return _ffi_api.multiply(lhs, rhs)

def dense(lhs: Expr, rhs: Expr) -> Expr:
    return _ffi_api.dense(lhs, rhs)

def conv2d(
    lhs: Expr,
    rhs: Expr,
    kernel_size,
    stride=(1, 1),
    padding=[0, 0],
    dilation=[1, 1]
) -> Expr:
    return _ffi_api.conv2d(lhs, rhs, kernel_size, stride, padding, dilation)

def relu(data: Expr) ->Expr:
    return _ffi_api.relu(data)

def softmax(data:Expr) -> Expr:
    return _ffi_api.softmax(data)

def flatten(data:Expr) -> Expr:
    return _ffi_api.flatten(data)

def max_pool2d(data:Expr, 
    kernel_size, 
    stride = None, 
    padding = (0, 0), 
    dilation = (1, 1)
) -> Expr:
    if stride is None: 
        stride = kernel_size
    return _ffi_api.max_pool2d(data, kernel_size, stride, padding, dilation)

@tvm.register_func("relax.run.unique")
def unique(
    a: tvm.nd.array,
    sort: int,
    return_inverse: int,
    return_counts: int,
    dim: int,
) -> tvm.nd.array:
    """Returns the unique elements of the input tensor.

    Uses numpy.unique to compute unique elements.
    """
#TODO(prakalp) : add support for returning a tuple when return_inverse or return_counts is True
    if bool(return_inverse) or bool(return_counts):
        raise NotImplementedError("missing support return_inverse or return_counts set to true")
    if dim < 0:
        dim = None
    a_numpy = a.numpy()
#TODO(prakalp) : use torch.unique instead of numpy when torch is installed in ci.
    output_sorted_numpy, indices = np.unique(a_numpy, return_index=True)
    if sort:
        return tvm.nd.array(output_sorted_numpy)
    output_numpy = [a_numpy.flatten()[index] for index in sorted(indices, reverse=True)]
    return tvm.nd.array(output_numpy)
