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
from tvm.relax.op import register_gradient
from tvm.relax.op import sum as _sum
from tvm.relax.op import (
    mean,
    less,
    where,
    collapse_sum_to,
    broadcast_to,
    multiply,
    negative,
    subtract,
    permute_dims,
    matmul,
    ones,
    zeros,
    expand_dims,
    concat,
    split,
)
from tvm.relax.expr import Call, Var
from typing import List


@register_gradient("relax.add")
def add_grad(orig: Call, grad: Var):
    """Gradients for add.

        z = relax.add(x, y)

    Returns [z_grad, z_grad].
    """
    return [
        collapse_sum_to(grad, orig.args[0].struct_info.shape),
        collapse_sum_to(grad, orig.args[1].struct_info.shape),
    ]


@register_gradient("relax.subtract")
def subtract_grad(orig: Call, grad: Var):
    """Gradients for subtract.

        z = relax.subtract(x, y)

    Returns [z_grad, -z_grad].
    """
    return [
        collapse_sum_to(grad, orig.args[0].struct_info.shape),
        collapse_sum_to(negative(grad), orig.args[1].struct_info.shape),
    ]


@register_gradient("relax.multiply")
def multiply_grad(orig: Call, grad: Var):
    """Gradients for multiply.

        z = relax.multiply(x, y)

    Returns [z_grad * y, z_grad * x].
    """
    x, y = orig.args
    return [
        collapse_sum_to(multiply(grad, y), x.struct_info.shape),
        collapse_sum_to(multiply(grad, x), y.struct_info.shape),
    ]


@register_gradient("relax.permute_dims")
def permute_dims_grad(orig: Call, grad: Var):
    """Returns grad transposed over the complement of original permute_dims axes."""
    axes = orig.attrs["axes"]
    if axes:
        dims = len(axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(axes[i])] = i
        return [permute_dims(grad, axes=new_axes)]
    else:
        return [permute_dims(grad)]


@register_gradient("relax.nn.relu")
def relu_grad(orig: Call, grad: Var):
    """Gradients for relu.

        y = relax.relu(x)

    Returns [y_grad * (where(x < 0, 0, 1))].
    """
    x = orig.args[0]
    x_zeros = zeros(x.struct_info.shape, x.struct_info.dtype)
    x_ones = ones(x.struct_info.shape, x.struct_info.dtype)
    return [where(less(x, x_zeros), x_zeros, multiply(x_ones, grad))]


@register_gradient("relax.matmul")
def matmul_grad(orig: Call, grad: Var):
    """Gradients for matmul.

        c = relax.matmul(a, b)

    Generally, returns [c_grad @ b^T, a^T @ c_grad].
    Here we only transpose the last two dimensions because of the definition
    of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig.args

    a_dim = len(tensor_a.struct_info.shape)
    b_dim = len(tensor_b.struct_info.shape)

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return permute_dims(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = matmul(grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(_transpose_last_two_dim(tensor_a, a_dim), grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(grad, -2)
        a_grad = matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(grad, -1)
        a_grad = matmul(grad_expand, b_expand)
        b_grad = mean(
            matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
        # legalizer now not support squeeze
        # squeeze(
        #     matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        # )
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = multiply(grad, tensor_b)
        b_grad = multiply(grad, tensor_a)

    return [
        collapse_sum_to(a_grad, tensor_a.struct_info.shape),
        collapse_sum_to(b_grad, tensor_b.struct_info.shape),
    ]


@register_gradient("relax.sum")
def sum_grad(orig: Call, grad: Var):
    """Gradient of sum."""
    axis = orig.attrs["axis"]
    keepdims = orig.attrs["keepdims"]
    if not keepdims and axis:
        grad = expand_dims(grad, [int(ax) for ax in axis])
    return [broadcast_to(grad, orig.args[0].struct_info.shape)]


@register_gradient("relax.nn.softmax")
def softmax_grad(orig: Call, grad: Var):
    """Gradient of softmax.

        y = relax.softmax(x, axis)

    Returns [(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]
    """
    return [multiply(subtract(grad, _sum(multiply(grad, orig), orig.attrs.axis, True)), orig)]


@register_gradient("relax.sigmoid")
def sigmoid_grad(orig: Call, grad: Var):
    """Gradient of sigmoid.

        y = relax.sigmoid(x)

    Returns [y_grad * y * (1 - y)].
    """
    return [
        multiply(
            grad,
            multiply(
                orig,
                subtract(
                    ones(orig.args[0].struct_info.shape, orig.args[0].struct_info.dtype), orig
                ),
            ),
        )
    ]


@register_gradient("relax.tanh")
def tanh_grad(orig: Call, grad: Var):
    """Gradient of tanh.

        y = relax.tanh(x)

    Returns [y_grad * (1 - y * y)].
    """
    return [
        multiply(
            grad,
            subtract(
                ones(orig.args[0].struct_info.shape, orig.args[0].struct_info.dtype),
                multiply(orig, orig),
            ),
        )
    ]


@register_gradient("relax.concat")
def concat_grad(orig: Call, grad: Var):
    """Gradient of concat.

        y = concat((x1, x2, x3), axis)

    Returns [split(y_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)].
    """
    axis = orig.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    split_indices: List[int] = []
    for i in range(len(orig.args[0]) - 1):
        sinfo = orig.args[0].struct_info.fields[i]
        index = sinfo.shape[axis]
        if i > 0:
            index += split_indices[i - 1]
        split_indices.append(index)
    return [split(grad, split_indices, axis)]


@register_gradient("relax.split")
def split_grad(orig: Call, grad: Var):
    """Gradient of split.

        y = split(x, indices, axis)

    Returns [concat(y_grad, axis)].
    """
    axis = orig.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    return [concat(grad, axis)]
