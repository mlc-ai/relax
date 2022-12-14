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
"""Gradient definitions for Relax operators"""
from tvm.relax import const, Tuple
from tvm.relay.op import register_gradient
import tvm.relax.op.nn as nn
from tvm.relax.op import (
    sum,
    mean,
    less,
    where,
    collapse_sum_to,
    log,
    multiply,
    divide,
    negative,
    subtract,
    transpose,
    sigmoid,
    tanh,
    ones,
    zeros,
    expand_dims,
)
from tvm.relax.expr import Call, Var


@register_gradient("relax.add")
def add_grad(orig: Call, grad: Var):
    """Returns [grad, grad]."""
    print(type(orig), type(grad))
    return [collapse_sum_to(grad, orig.args[0].shape), collapse_sum_to(grad, orig.args[1].shape)]


@register_gradient("relax.subtract")
def subtract_grad(orig: Call, grad: Var):
    """Returns [grad, -grad]."""
    return [
        collapse_sum_to(grad, orig.args[0].shape),
        collapse_sum_to(negative(grad), orig.args[1].shape),
    ]


@register_gradient("relax.multiply")
def multiply_grad(orig: Call, grad: Var):
    """Returns [grad * y, grad * x]."""
    x, y = orig.args
    return [
        collapse_sum_to(multiply(grad, y), x.shape),
        collapse_sum_to(multiply(grad, x), y.shape),
    ]


@register_gradient("relax.transpose")
def transpose_grad(orig: Call, grad: Var):
    """Returns grad transposed over the complement of original transpose axes."""
    axes = orig.attrs["axes"]
    if axes:
        return [transpose(grad, axes=axes)]
    else:
        return [transpose(grad)]


@register_gradient("relax.nn.relu")
def relu_grad(orig: Call, grad: Var):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    x_zeros = zeros(x.shape)
    x_ones = ones(x.shape)
    return [where(less(x, x_zeros), x_zeros, multiply(x_ones, grad))]


@register_gradient("relax.nn.matmul")
def matmul_grad(orig: Call, grad: Var):
    """Gradients for matmul.

        c = relax.nn.matmul(a, b)

    Generally, returns [grad @ b^T, a^T @ grad]. Here we only transpose the last two dimensions
    because of the definition of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig.args

    a_dim = len(tensor_a.shape)
    b_dim = len(tensor_b.shape)
    grad_dim = len(grad.shape)

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return transpose(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = nn.matmul(grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = nn.matmul(_transpose_last_two_dim(tensor_a, a_dim), grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(grad, -2)
        a_grad = nn.matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = nn.matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(grad, -1)
        a_grad = nn.matmul(grad_expand, b_expand)
        b_grad = mean(
            nn.matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = multiply(grad, tensor_b)
        b_grad = multiply(grad, tensor_a)

    return [collapse_sum_to(a_grad, tensor_a.shape), collapse_sum_to(b_grad, tensor_b.shape)]


@register_gradient("relax.sum")
def sum_grad(orig: Call, grad: Var):
    """Returns [grad * ones(x.shape)]."""
    return [multiply(grad, ones(orig.args[0].shape))]


@register_gradient("relax.nn.softmax")
def softmax_grad(orig: Call, grad: Var):
    """Gradient of softmax."""
    return [multiply(subtract(grad, sum(multiply(grad, orig), orig.attrs.axis, True)), orig)]


@register_gradient("relax.nn.cross_entropy")
def cross_entropy_grad(orig: Call, grad: Var):
    """Gradient of cross_entropy.

        z = relax.nn.cross_entropy(x, y)

    Returns [- grad * y / x, - grad * log(x)].
    """
    x, y = orig.args
    return [negative(multiply(grad, divide(y, x))), negative(multiply(grad, log(x)))]


@register_gradient("relax.nn.softmax_cross_entropy")
def softmax_cross_entropy_grad(orig: Call, grad: Var):
    """Gradient of softmax_cross_entropy.

        z = relax.nn.softmax_cross_entropy(x, y)

    This gradient is based on the assumption that sum(y)=1.
    Returns [grad * (y - softmax(x)), grad * (-log(-softmax(x)))].
    """
    y_hat = nn.softmax(orig.args[0])
    return [multiply(grad, subtract(y_hat, orig.args[1])), multiply(grad, negative(log(y_hat)))]


@register_gradient("relax.sigmoid")
def sigmoid_grad(orig: Call, grad: Var):
    """Gradient of sigmoid.

        y = relax.sigmoid(x)

    Returns [grad * sigmoid(x) * (1 - sigmoid(x))].
    """
    out = sigmoid(orig.args[0])
    return [multiply(grad, multiply(out, subtract(ones(orig.args[0].shape), out)))]


@register_gradient("relax.tanh")
def tanh_grad(orig: Call, grad: Var):
    """Gradient of tanh.

        y = relax.tanh(x)

    Returns [grad * (1 - tanh(x) * tanh(x))].
    """
    out = tanh(orig.args[0])
    return [multiply(grad, subtract(ones(orig.args[0].shape), multiply(out, out)))]
