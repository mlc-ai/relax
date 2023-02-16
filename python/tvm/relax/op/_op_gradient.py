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
from tvm._ffi.base import TVMError
from tvm.arith import Analyzer
from tvm.relax.op.index import strided_slice
from tvm.relax.op.nn.nn import conv2d

from ..block_builder import BlockBuilder
from ..expr import Call, Var, Expr, ShapeExpr
from ...tir import PrimExpr

from .base import register_gradient
from .unary import cos, exp, sin, sqrt
from .binary import less
from .statistical import sum
from .create import zeros, ones, zeros_like
from .search import where
from .linear_algebra import matmul
from .manipulate import (
    collapse_sum_to,
    broadcast_to,
    permute_dims,
    expand_dims,
    concat,
    reshape,
    split,
    squeeze,
    tile,
)
from .nn import conv2d_transpose
from .grad import (
    nll_loss_backward,
    max_pool2d_backward,
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
        `z = relax.add(x, y)`

    Backward:
        Returns `[z_output_grad, z_grad]`.
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
        `z = relax.subtract(x, y)`

    Backward:
        Returns `[z_output_grad, -z_grad]`.
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
        `z = relax.multiply(x, y)`

    Backward:
        Returns `[z_grad * y, z_grad * x]`.
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
        `y = relax.sigmoid(x)`

    Backward:
        Returns `[y_grad * y * (1 - y)]`.
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
        `y = relax.tanh(x)`

    Backward:
        Returns `[y_grad * (1 - y * y)]`.
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
        `y = relax.sum(x, axis, keepdims)`

    Backward:
        Returns `[broadcast_to(y_output_grad, x.shape)]`.

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
        `y = relax.permute_dims(x, axes)`

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
        `y = concat((x1, x2, x3), axis)`

    Backward:
        Returns `[split(y_output_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)]`.
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
        `y = split(x, indices, axis)`

    Backward:
        Returns `[concat(y_output_grad, axis)]`.
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
        `c = relax.matmul(a, b)`

    Backward:
        Generally, returns `[c_grad @ b^T, a^T @ c_grad]`.

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
        `y = relax.relu(x)`

    Backward:
        Returns `[y_grad * (where(x < 0, 0, 1))]`.
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
        `y = relax.softmax(x, axis)`

    Backward:
        Returns `[(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]`
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
        `y = relax.log_softmax(x, axis)`

    Backward:
        Returns `[y_grad - sum(y_output_grad, axis, keepdims=True) * softmax(x)]`
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


@register_gradient("relax.nn.nll_loss")
def nll_loss_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of nll_loss.

    Forward Form:
        `z = nll_loss(predictions, targets, weights, reduction, ignore_index)`

        Suppose that `out = nll_loss(predictions, targets, weights, "none", ignore_index)`, and
        `z = reduction(out)` where reduction is in `["none", "mean", "sum"]`.

    Backward:
        First find the gradient w.r.t. `out`. Assume it is `out_grad`.

        Gererally, the gradient w.r.t. predictions is

        `predictions_grad[n, c, i_1, ..., i_k] = -o * w if c == t else 0`, where
        - `o = out_grad[n, i_1, ..., i_k]`,
        - `w = weights[n, i_1, ..., i_k]`,
        - `t = targets[n, i_1, ..., i_k]`.

        Additional checks are added if `ignore_index >= 0`, `weights=None`, or the predictions
        provided do not have batch.

        The gradient w.r.t. targets and weights are not available. Now `nll_loss_grad` return zeros
        for them.
    """
    pred_grad = nll_loss_backward(  # type: ignore
        output_grad,
        *orig_call.args,
        reduction=orig_call.attrs.reduction,
        ignore_index=orig_call.attrs.ignore_index,
    )
    tgt_grad = zeros(orig_call.args[1].struct_info.shape, orig_call.args[1].struct_info.dtype)
    if len(orig_call.args) == 2:
        return [pred_grad, tgt_grad]

    weight_grad = zeros(orig_call.args[2].struct_info.shape, orig_call.args[2].struct_info.dtype)
    return [pred_grad, tgt_grad, weight_grad]


@register_gradient("relax.nn.conv2d")
def conv2d_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
) -> List[Expr]:
    """Gradient of nll_loss.

    Forward Form:
        `z = nll_loss(predictions, targets, weights, reduction, ignore_index)`

        Suppose that `out = nll_loss(predictions, targets, weights, "none", ignore_index)`, and
        `z = reduction(out)` where reduction is in `["none", "mean", "sum"]`.

    Backward:
        First find the gradient w.r.t. `out`. Assume it is `out_grad`.

        Gererally, the gradient w.r.t. predictions is

        `predictions_grad[n, c, i_1, ..., i_k] = -o * w if c == t else 0`, where
        - `o = out_grad[n, i_1, ..., i_k]`,
        - `w = weights[n, i_1, ..., i_k]`,
        - `t = targets[n, i_1, ..., i_k]`.

        Additional checks are added if `ignore_index >= 0`, `weights=None`, or the predictions
        provided do not have batch.

        The gradient w.r.t. targets and weights are not available. Now `nll_loss_grad` return zeros
        for them.
    """
    attrs = orig_call.attrs
    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout == "NCHW", "only support NCHW output layout"

    assert len(attrs.padding) == 4
    assert len(attrs.strides) == 2
    assert len(attrs.dilation) == 2

    # calculate output_padding
    data, weight = orig_call.args
    batch, out_channel, grad_h, grad_w = _get_shape(orig_var)
    _, in_channel, in_h, in_w = _get_shape(data)
    _, _, filter_h, filter_w = _get_shape(weight)

    fpad_top, fpad_left, fpad_bottom, fpad_right = attrs.padding
    stride_h, stride_w = attrs.strides
    dilation_h, dilation_w = attrs.dilation

    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w

    output_padding = (in_h - out_h, in_w - out_w)

    data_grad = conv2d_transpose(  # type: ignore
        output_grad,
        orig_call.args[1],
        attrs.strides,
        attrs.padding,
        output_padding,
        attrs.dilation,
        attrs.groups,
        attrs.out_layout,
        attrs.kernel_layout[1] + attrs.kernel_layout[0] + attrs.kernel_layout[2:],
        attrs.data_layout,
        attrs.out_dtype,
    )

    grad = ctx.normalize(tile(output_grad, [1, in_channel // attrs.groups, 1, 1]))
    # (batch * oc * ic // groups, 1, out_h, out_w)
    grad = ctx.normalize(reshape(grad, [-1, 1, out_h, out_w]))
    # (1, batch * ic, in_h, in_w)
    data = ctx.normalize(reshape(data, [1, -1, in_h, in_w]))

    weight_grad = ctx.normalize(
        conv2d(
            data,
            grad,
            strides=attrs.dilation,
            padding=attrs.padding,
            dilation=attrs.strides,
            groups=int(in_channel * batch),
            out_dtype=attrs.out_dtype,
        )
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1

    weight_grad = ctx.normalize(
        reshape(
            weight_grad,
            [
                batch,
                in_channel // attrs.groups,
                out_channel,
                padded_weight_grad_h,
                padded_weight_grad_w,
            ],
        )
    )
    weight_grad = ctx.normalize(sum(weight_grad, axis=0))
    weight_grad = ctx.normalize(permute_dims(weight_grad, [1, 0, 2, 3]))

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w

    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        weight_grad = ctx.normalize(
            strided_slice(
                weight_grad,
                axes=[0, 1, 2, 3],
                begin=[0, 0, 0, 0],
                end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
            )
        )

    return [data_grad, weight_grad]


@register_gradient("relax.nn.max_pool2d")
def max_pool2d_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of max_pool2d."""
    return [
        max_pool2d_backward(  # type: ignore
            output_grad,
            orig_call.args[0],
            orig_call.attrs.pool_size,
            orig_call.attrs.strides,
            orig_call.attrs.padding,
            orig_call.attrs.dilation,
            orig_call.attrs.ceil_mode,
            orig_call.attrs.layout,
            orig_call.attrs.out_layout,
        )
    ]
