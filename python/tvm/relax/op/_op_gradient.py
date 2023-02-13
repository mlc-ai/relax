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
# pylint: disable=unused-argument, redefined-builtin
"""Gradient definitions for Relax operators"""
from typing import List
from tvm import relax
from tvm.arith import Analyzer
from tvm.relax.expr import Call, Var, Expr, ShapeExpr
from tvm._ffi.base import TVMError

from ..block_builder import BlockBuilder
from ...tir import PrimExpr
from .base import register_gradient

from .unary import (
    cos,
    exp,
    log,
    sin,
    sqrt,
)
from .binary import less
from .statistical import sum
from .create import (
    zeros,
    ones,
    zeros_like,
)
from .search import where
from .linear_algebra import matmul
from .manipulate import (
    collapse_sum_to,
    broadcast_to,
    permute_dims,
    expand_dims,
    concat,
    split,
    squeeze,
)


def _get_shape(expr: Expr) -> ShapeExpr:
    """Get the shape from a Tensor expr."""
    try:
        shape = expr.struct_info.shape
    except Exception as error:
        raise TVMError(
            f"Get the shape of {expr} failed. Please normalize it first and ensure it is a Tensor."
        ) from error
    return shape


def _get_dtype(expr: Expr) -> str:
    """Get the dtype from a Tensor expr."""
    try:
        dtype = expr.struct_info.dtype
    except Exception as error:
        raise TVMError(
            f"Get the dtype of {expr} failed. Please normalize it first and ensure it is a Tensor."
        ) from error
    return dtype


def _ones(expr: Expr) -> Expr:
    return ones(_get_shape(expr), _get_dtype(expr))


def _zeros(expr: Expr) -> Expr:
    return zeros(_get_shape(expr), _get_dtype(expr))


def _fit_shape(expr: Expr, expr_shape: ShapeExpr, target: Expr) -> Expr:
    target_shape = _get_shape(target)
    expr_sinfo = expr_shape.struct_info
    target_sinfo = target_shape.struct_info
    assert isinstance(expr_sinfo, relax.ShapeStructInfo)
    assert isinstance(target_sinfo, relax.ShapeStructInfo)

    def _check_shape_equal():
        if len(expr_sinfo.values) != len(target_sinfo.values):
            return False
        analyzer = Analyzer()
        for i, field in enumerate(expr_sinfo.values):
            if not analyzer.can_prove_equal(field, target_sinfo.values[i]):
                return False
        return True

    return expr if _check_shape_equal() else collapse_sum_to(expr, target_shape)


##################### Binary #####################


@register_gradient("relax.add")
def add_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of add.

    Forward Form:
        z = relax.add(x, y)

    Backward:
        Returns [z_output_grad, z_grad].
    """
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad, output_grad_shape, orig_call.args[0]),
        _fit_shape(output_grad, output_grad_shape, orig_call.args[1]),
    ]


@register_gradient("relax.subtract")
def subtract_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of subtract.

    Forward Form:
        z = relax.subtract(x, y)

    Backward:
        Returns [z_output_grad, -z_grad].
    """
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad, output_grad_shape, orig_call.args[0]),
        _fit_shape(-output_grad, output_grad_shape, orig_call.args[1]),
    ]


@register_gradient("relax.multiply")
def multiply_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of multiply.

    Forward Form:
        z = relax.multiply(x, y)

    Backward:
        Returns [z_grad * y, z_grad * x].
    """
    x, y = orig_call.args
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad * y, output_grad_shape, x),
        _fit_shape(output_grad * x, output_grad_shape, y),
    ]


@register_gradient("relax.divide")
def divide_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of divide.

    Forward Form:
        z = relax.divide(x, y)

    Backward:
        Returns [z_grad / y,  - z_grad * z / y].
    """
    x, y = orig_call.args
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad / y, output_grad_shape, x),
        _fit_shape(-output_grad * orig_var / y, output_grad_shape, y),
    ]


# TODO(chaofan): floor-divide

# For most comparison operators, the gradients are just zeros.
def _binary_zeros(call: Call) -> List[Expr]:
    return [_zeros(call.args[0]), _zeros(call.args[1])]


@register_gradient("relax.equal")
def equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


@register_gradient("relax.greater")
def greater_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


@register_gradient("relax.greater_equal")
def greater_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


@register_gradient("relax.less")
def less_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


@register_gradient("relax.less_equal")
def less_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


@register_gradient("relax.not_equal")
def not_equal_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return _binary_zeros(orig_call)


##################### Create #####################


# For these create operators, the gradients are just zeros.
@register_gradient("relax.zeros_like")
def zeros_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return [orig_var]


@register_gradient("relax.ones_like")
def ones_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return [zeros_like(orig_call.args[0], orig_call.attrs.dtype)]


@register_gradient("relax.full_like")
def full_like_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    return [zeros_like(orig_call.args[0], orig_call.attrs.dtype)]


##################### Unary #####################


@register_gradient("relax.abs")
def abs_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of abs.

    Forward Form:
        y = relax.abs(x)

    Backward:
        Returns [y_grad * where(x < 0, -1, 1)].
    """
    x = orig_call.args[0]
    x_zeros = _zeros(x)
    x_ones = _ones(x)
    return [output_grad * where(less(x, x_zeros), -x_ones, x_ones)]


@register_gradient("relax.cos")
def cos_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of cos.

    Forward Form:
        y = relax.cos(x)

    Backward:
        Returns [-y_grad * sin(x)].
    """
    return [-output_grad * sin(orig_call.args[0])]


@register_gradient("relax.exp")
def exp_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of exp.

    Forward Form:
        y = relax.exp(x)

    Backward:
        Returns [y_grad * y].
    """
    return [output_grad * orig_var]


@register_gradient("relax.log")
def log_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of log.

    Forward Form:
        y = relax.log(x)

    Backward:
        Returns [y_grad / x].
    """
    return [output_grad / orig_call.args[0]]


@register_gradient("relax.negative")
def negative_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of negative.

    Forward Form:
        y = relax.negative(x)

    Backward:
        Returns [- y_grad].
    """
    return [-output_grad]


@register_gradient("relax.sigmoid")
def sigmoid_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of sigmoid.

    Forward Form:
        y = relax.sigmoid(x)

    Backward:
        Returns [y_grad * y * (1 - y)].
    """
    x_ones = _ones(orig_call.args[0])
    return [output_grad * orig_var * (x_ones - orig_var)]


@register_gradient("relax.sin")
def sin_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of sin.

    Forward Form:
        y = relax.sin(x)

    Backward:
        Returns [y_grad * cos(x)].
    """
    return [output_grad * cos(orig_call.args[0])]


@register_gradient("relax.sqrt")
def sqrt_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of sqrt.

    Forward Form:
        y = relax.sqrt(x)

    Backward:
        Returns [0.5 * y_grad / sqrt(x)].
    """
    x = orig_call.args[0]
    cst = relax.const(0.5, dtype=_get_dtype(x))
    return [cst * output_grad / sqrt(x)]


@register_gradient("relax.tanh")
def tanh_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of tanh.

    Forward Form:
        y = relax.tanh(x)

    Backward:
        Returns [y_grad * (1 - y * y)].
    """
    x_ones = _ones(orig_call.args[0])
    return [output_grad * (x_ones - orig_var * orig_var)]


##################### Statistical #####################


@register_gradient("relax.sum")
def sum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of sum.

    Forward Form:
        y = relax.sum(x, axis, keepdims)

    Backward:
        Returns [broadcast_to(y_output_grad, x.shape)].
        If `keepdims=False`, the summed axis will be added back.
    """
    axis = orig_call.attrs["axis"]
    keepdims = orig_call.attrs["keepdims"]
    if not keepdims and axis:
        output_grad = expand_dims(output_grad, axis)
    return [broadcast_to(output_grad, _get_shape(orig_call.args[0]))]


##################### Manipulate #####################


@register_gradient("relax.permute_dims")
def permute_dims_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of permute_dims.

    Forward Form:
        y = relax.permute_dims(x, axes)

    Backward:
        Returns grad transposed over the **inverse permutation** of the original permute_dims axes.
    """
    axes = orig_call.attrs["axes"]
    if axes:
        dims = len(axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(axes[i])] = i
        return [permute_dims(output_grad, axes=new_axes)]
    else:
        return [permute_dims(output_grad)]


# TODO(yixin, chaofan): handle symbolic shape
@register_gradient("relax.concat")
def concat_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of concat.

    Forward Form:
        y = concat((x1, x2, x3), axis)

    Backward:
        Returns [split(y_output_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)].
    """
    axis = orig_call.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    split_indices: List[PrimExpr] = []
    sinfo = orig_call.args[0].struct_info
    assert isinstance(sinfo, relax.TupleStructInfo)
    for i in range(len(sinfo.fields) - 1):
        tensor_sinfo = sinfo.fields[i]
        assert isinstance(tensor_sinfo, relax.TensorStructInfo)
        assert tensor_sinfo.shape is not None
        index = tensor_sinfo.shape[axis]
        if i > 0:
            index += split_indices[i - 1]
        split_indices.append(index)
    return [split(output_grad, split_indices, axis)]


@register_gradient("relax.split")
def split_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of split.

    Forward Form:
        y = split(x, indices, axis)

    Backward:
        Returns [concat(y_output_grad, axis)].
    """
    axis = orig_call.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    return [concat(output_grad, axis)]


##################### Linear Algebra #####################


@register_gradient("relax.matmul")
def matmul_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of matmul.

    Forward Form:
        c = relax.matmul(a, b)

    Backward:
        Generally, returns [c_grad @ b^T, a^T @ c_grad].
        Here we only transpose the last two dimensions because of the definition
        of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig_call.args

    a_dim = len(_get_shape(tensor_a))
    b_dim = len(_get_shape(tensor_b))

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return permute_dims(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = matmul(output_grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(_transpose_last_two_dim(tensor_a, a_dim), output_grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(output_grad, -2)
        a_grad = matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(output_grad, -1)
        a_grad = matmul(grad_expand, b_expand)
        b_grad = squeeze(
            matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = output_grad * tensor_b
        b_grad = output_grad * tensor_a

    output_grad_shape = _get_shape(output_grad)

    return [
        _fit_shape(a_grad, output_grad_shape, tensor_a),
        _fit_shape(b_grad, output_grad_shape, tensor_b),
    ]


##################### Neural network #####################


@register_gradient("relax.nn.relu")
def relu_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of relu.

    Forward Form:
        y = relax.relu(x)

    Backward:
        Returns [y_grad * (where(x < 0, 0, 1))].
    """
    x = orig_call.args[0]
    x_zeros = _zeros(x)
    x_ones = _ones(x)
    return [where(less(x, x_zeros), x_zeros, x_ones) * output_grad]


@register_gradient("relax.nn.softmax")
def softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of softmax.

    Forward Form:
        y = relax.softmax(x, axis)

    Backward:
        Returns [(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]
    """
    return [(output_grad - sum(output_grad * orig_var, orig_call.attrs.axis, True)) * orig_var]


@register_gradient("relax.nn.log_softmax")
def log_softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of log_softmax.

    Forward Form:
        y = relax.log_softmax(x, axis)

    Backward:
        Returns [y_grad - sum(y_output_grad, axis, keepdims=True) * softmax(x)]
    """
    x_softmax = exp(orig_var)
    return [(output_grad - sum(output_grad, orig_call.attrs.axis, True) * x_softmax)]


def _divide_batch(x: Expr, expr: Expr):
    if x.struct_info.ndim > 1:
        # TODO(chaofan, yixin): support symbolic shape
        x_shape = _get_shape(x)
        batch_size = int(x_shape[0])
        # batch_size = take(shape_of(x), relax.const(0, dtype="int32"), axis=0)
        # expr = divide(expr, batch_size)
        expr = expr / relax.const(batch_size, dtype=_get_dtype(expr))
    return expr


@register_gradient("relax.nn.cross_entropy_without_logits")
def cross_entropy_without_logits_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_without_logits(x, y)

    Backward:
        Returns [-z_grad * y / x, -z_grad * log(x)].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig_call.args
    output_grad = _divide_batch(x, output_grad)
    return [-output_grad * y / x, -output_grad * log(x)]


@register_gradient("relax.nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_with_logits(x, y)

    Backward:
        Returns [-z_grad * y, -z_grad * x].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig_call.args
    output_grad = _divide_batch(x, output_grad)
    return [-output_grad * y, -output_grad * x]
