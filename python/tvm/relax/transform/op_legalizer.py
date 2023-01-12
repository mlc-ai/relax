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
# pylint: disable=abstract-method,invalid-name,missing-module-docstring
import logging

import tvm
from tvm import ir, te, topi, relax
from tvm.relax import struct_info
from tvm.ir.module import IRModule

from ..analysis import remove_all_unused
from ..expr import Call, Function, ShapeExpr, Tuple, TupleGetItem, Var
from ..expr_functor import mutator, PyExprMutator
from ..block_builder import BlockBuilder


# Todo(ruihang): confirm symbolic shape support


def has_known_shape_value(sinfo: struct_info.StructInfo):
    if isinstance(sinfo, struct_info.TensorStructInfo):
        return isinstance(sinfo.shape, ShapeExpr)
    elif isinstance(sinfo, struct_info.ShapeStructInfo):
        return sinfo.values is not None
    elif isinstance(sinfo, struct_info.TupleStructInfo):
        return all([has_known_shape_value(field_sinfo) for field_sinfo in sinfo.fields])
    else:
        return False


# Todo(ruihang): check how much TOPI can handle layout such as NCHW16c?
def _nn_conv2d(bb: BlockBuilder, call: Call):
    if call.attrs.out_layout != call.attrs.data_layout:
        logging.info(
            "TOPI conv2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.conv2d,
        input=call.args[0],
        filter=call.args[1],
        strides=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        data_layout=call.attrs.data_layout,
        kernel_layout=call.attrs.kernel_layout,
        out_dtype=call.attrs.out_dtype if call.attrs.out_dtype != "" else None,
    )


def _add(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.add, call.args[0], call.args[1])


def _subtract(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.subtract, call.args[0], call.args[1])


def _multiply(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.multiply, call.args[0], call.args[1])


def _divide(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.divide, call.args[0], call.args[1])


def _floor_divide(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.floor_divide, call.args[0], call.args[1])


def _sin(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.sin, call.args[0])


def _cos(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.cos, call.args[0])


def _tanh(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.tanh, call.args[0])


def _negative(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.negative, call.args[0])


def _log(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.log, call.args[0])


def _sqrt(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.sqrt, call.args[0])


def _sigmoid(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.sigmoid, call.args[0])


def _less(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.less, call.args[0], call.args[1])


def _nn_relu(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.nn.relu, call.args[0])


def _nn_gelu(bb: BlockBuilder, call: Call):
    def gelu(x):
        dtype = x.dtype
        return x * (
            tvm.tir.const(0.5, dtype)
            + topi.erf(x * tvm.tir.const(0.5**0.5, dtype)) * tvm.tir.const(0.5, dtype)
        )

    return bb.call_te(gelu, call.args[0], primfunc_name_hint="gelu")


def _nn_silu(bb: BlockBuilder, call: Call):
    sig = bb.emit_te(topi.sigmoid, call.args[0])
    return bb.call_te(topi.multiply, call.args[0], sig)


def _reshape(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.reshape, call.args[0], call.args[1])


def _permute_dims(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.transpose, call.args[0], call.attrs.axes)


def _concat(bb: BlockBuilder, call: Call):
    t = call.args[0]
    n_field = len(t.struct_info.fields)
    while isinstance(t, Var):
        binding = bb.lookup_binding(t)
        if not isinstance(binding, (Tuple, Var)):
            break
        t = binding

    assert isinstance(t, (Tuple, Var))
    fields = (
        t.fields if isinstance(t, Tuple) else [bb.emit(TupleGetItem(t, i)) for i in range(n_field)]
    )
    return bb.call_te(
        topi.concatenate, fields, None if call.attrs.axis is None else call.attrs.axis.value
    )


def _expand_dims(bb: BlockBuilder, call: Call):
    # old approach:
    def te_expand_dims(data, axis):
        data_relax = relax.Var("data", relax.TensorStructInfo(data.shape))
        f_infer_sinfo = call.op.get_attr("FInferStructInfo")
        output_shape = f_infer_sinfo(relax.op.expand_dims(data_relax, axis), bb).shape
        output_ndim = len(output_shape)

        data_dims = []
        for i in range(output_ndim):
            if i not in axis and (i - output_ndim) not in axis:
                data_dims.append(i)
        return te.compute(
            output_shape,
            lambda *idx: data(*[idx[dim] for dim in data_dims]),
            name="expand_dims",
        )

    return bb.call_te(
        te_expand_dims, call.args[0], call.attrs.axis, primfunc_name_hint="expand_dims"
    )

    # Equivalent approach with existing TOPI (which is good) but more
    # complicated PrimFunc (which is not perfect):
    ###  return bb.call_te(topi.reshape, call.args[0], call.struct_info.shape)


# Todo(ruihang): not yet introduced
# def _cumsum(bb: BlockBuilder, call: Call):
#     return bb.call_te(topi.cumsum, args[0], attrs.axis)


def _tril(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.trilu, call.args[0], tvm.tir.const(call.attrs.k, "int32"), upper=False)


def _triu(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.trilu, call.args[0], tvm.tir.const(call.attrs.k, "int32"), upper=True)


def _astype(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.cast, call.args[0], call.attrs.dtype)


def _take(bb: BlockBuilder, call: Call):
    # Todo(ruihang): revisit the mode
    return bb.call_te(topi.take, call.args[0], call.args[1], call.attrs.axis, mode="fast")


def _full(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.full, call.args[0], call.struct_info.dtype, call.args[1])


def _zeros(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.full, call.args[0], call.struct_info.dtype, 0.0)


def _ones(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.full, call.args[0], call.struct_info.dtype, 1.0)


def _full_like(bb: BlockBuilder, call: Call):
    return bb.call_te(
        topi.full, call.args[0].struct_info.shape, call.struct_info.dtype, call.args[1]
    )


def _ones_like(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.full, call.args[0].struct_info.shape, call.struct_info.dtype, 1.0)


def _zeros_like(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.full, call.args[0].struct_info.shape, call.struct_info.dtype, 0.0)


def _collapse_sum_like(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.collapse_sum, call.args[0], call.args[1].struct_info.shape)


def _collapse_sum_to(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.collapse_sum, call.args[0], call.args[1])


def _split(bb: BlockBuilder, call: Call):
    indices_or_sections = (
        call.attrs.indices_or_sections.value
        if isinstance(call.attrs.indices_or_sections, tvm.tir.IntImm)
        else call.attrs.indices_or_sections
    )
    return bb.call_te(topi.split, call.args[0], indices_or_sections, call.attrs.axis)


def _broadcast_to(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.broadcast_to, call.args[0], call.args[1])


def _strided_slice(bb: BlockBuilder, call: Call):
    return bb.call_te(
        topi.strided_slice,
        call.args[0],
        call.attrs.begin,
        call.attrs.end,
        call.attrs.strides,
        call.attrs.axes,
        slice_mode="end",
    )


def _where(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.where, call.args[0], call.args[1], call.args[2])


def _nn_max_pool2d(bb: BlockBuilder, call: Call):
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI max_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    return bb.call_te(
        topi.nn.pool2d,
        call.args[0],
        kernel=call.attrs.pool_size,
        stride=call.attrs.strides,
        dilation=call.attrs.dilation,
        padding=call.attrs.padding,
        pool_type="max",
        layout=call.attrs.layout,
    )


def _nn_batch_norm(bb: BlockBuilder, call: Call):
    return bb.call_te(
        topi.nn.batch_norm,
        data=call.args[0],
        gamma=call.args[1],
        beta=call.args[2],
        moving_mean=call.args[3],
        moving_var=call.args[4],
        axis=call.attrs.axis,
        epsilon=call.attrs.epsilon,
        center=call.attrs.center,
        scale=call.attrs.scale,
    )


def _nn_layer_norm(bb: BlockBuilder, call: Call):
    def te_layer_norm(x, gamma, beta, axes, eps):
        shape_prod = tvm.tir.const(1, "int32")
        for dim in axes:
            shape_prod = shape_prod * x.shape[dim.value]
        mean = topi.sum(x, axis=axes, keepdims=True) / shape_prod
        var = topi.sum((x - mean) * (x - mean), axis=axes, keepdims=True) / shape_prod
        return gamma * ((x - mean) / topi.sqrt(var + eps)) + beta

    return bb.call_te(
        te_layer_norm,
        call.args[0],
        call.args[1],
        call.args[2],
        axes=call.attrs.axes,
        eps=call.attrs.epsilon,
        primfunc_name_hint="layer_norm",
    )


def _matmul(bb: BlockBuilder, call: Call):
    def te_matmul(a, b):
        a_shape = list(a.shape)
        b_shape = list(b.shape)
        a_prepended = False
        b_appended = False
        if len(a_shape) == 1:
            a_prepended = True
            a_shape.insert(0, 1)
        if len(b_shape) == 1:
            b_appended = True
            b_shape.append(1)

        is_a_larger = len(a_shape) > len(b_shape)
        offset = len(a_shape) - len(b_shape) if is_a_larger else len(b_shape) - len(a_shape)

        a_relax = relax.Var("a", relax.TensorStructInfo(a.shape))
        b_relax = relax.Var("b", relax.TensorStructInfo(b.shape))
        f_infer_sinfo = call.op.get_attr("FInferStructInfo")
        output_shape = f_infer_sinfo(relax.op.matmul(a_relax, b_relax), bb).shape
        print(output_shape)

        def matmul_compute(*idx_spatial):
            k = te.reduce_axis((0, a_shape[-1]), name="k")

            def multiply_compute(idx_reduce):
                a_indices = []
                b_indices = []

                for i in range(offset):
                    if is_a_larger:
                        a_indices.append(idx_spatial[i])
                    else:
                        b_indices.append(idx_spatial[i])
                for i in range(offset, len(output_shape) - (2 - a_prepended - b_appended)):
                    a_dim = a_shape[i if is_a_larger else i - offset]
                    b_dim = b_shape[i if not is_a_larger else i - offset]
                    a_dim_is_one = isinstance(a_dim, tvm.tir.IntImm) and a_dim == 1
                    b_dim_is_one = isinstance(b_dim, tvm.tir.IntImm) and b_dim == 1
                    a_indices.append(0 if a_dim_is_one else idx_spatial[i])
                    b_indices.append(0 if b_dim_is_one else idx_spatial[i])
                if not a_prepended:
                    a_indices.append(idx_spatial[-2 + b_appended])
                a_indices.append(idx_reduce)
                b_indices.append(idx_reduce)
                if not b_appended:
                    b_indices.append(idx_spatial[-1])

                dtype = call.attrs.out_dtype
                if dtype != "":
                    return a(*a_indices).astype(dtype) * b(*b_indices).astype(dtype)
                else:
                    return a(*a_indices) * b(*b_indices)

            return te.sum(multiply_compute(k), axis=k)

        return te.compute(
            output_shape,
            lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
            name="matmul",
        )

    return bb.call_te(te_matmul, call.args[0], call.args[1], primfunc_name_hint="matmul")


def _nn_softmax(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.nn.softmax, call.args[0], call.attrs.axis)


def _flatten(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.nn.flatten, call.args[0])


def _nn_adaptive_max_pool2d(bb: BlockBuilder, call: Call):
    if call.attrs.out_layout != call.attrs.layout:
        logging.info(
            "TOPI adaptive_max_pool2d does not support different input-output "
            "layouts, and thus cannot be legalized by TOPI"
        )
        return call

    if call.attrs.output_size is None:
        # Todo(ruihang): seems that TOPI adaptive_avg_pool cannot handle the
        # cases where output_size is not provided
        return call

    return bb.call_te(
        topi.nn.adaptive_pool,
        call.args[0],
        call.attrs.output_size,
        pool_type="avg",
        layout=call.attrs.layout,
    )


# def _nn_softmax_cross_entropy(bb: BlockBuilder, call: Call):
#     def te_softmax_cross_entropy(x, y):
#         return _te_cross_entropy(topi.nn.softmax(x), y)

#     return bb.call_te(
#         te_softmax_cross_entropy, args[0], args[1], primfunc_name_hint="softmax_cross_entropy"
#     )


def _sum(bb: BlockBuilder, call: Call):
    return bb.call_te(topi.sum, call.args[0], call.attrs.axis, call.attrs.keepdims)


def _mean(bb: BlockBuilder, call: Call):
    shape_prod = tvm.tir.const(1, "int32")
    axis = (
        call.attrs.axis
        if call.attrs.axis is not None
        else range(0, len(call.args[0].struct_info.shape))
    )
    for dim in axis:
        dim_value = dim if isinstance(dim, int) else dim.value
        if dim_value < 0:
            dim_value += len(call.args[0].struct_info.shape)
        shape_prod = shape_prod * call.args[0].struct_info.shape[dim_value]
    sum_var = bb.emit_te(topi.sum, call.args[0], axis, call.attrs.keepdims)
    return bb.call_te(topi.divide, sum_var, shape_prod)


def _image_resize2d(bb: BlockBuilder, call: Call):
    return bb.call_te(
        topi.image.resize2d,
        call.args[0],
        roi=call.attrs.roi,
        size=call.attrs.size,
        layout=call.attrs.layout,
        method=call.attrs.method,
        coordinate_transformation_mode=call.attrs.coordinate_transformation_mode,
        rounding_method=call.attrs.rounding_method,
        bicubic_alpha=call.attrs.cubic_alpha,
        bicubic_exclude=call.attrs.cubic_exclude,
        extrapolation_value=call.attrs.extrapolation_value,
    )


op_legalization_map = {
    ir.Op.get("relax.nn.conv2d"): _nn_conv2d,
    ir.Op.get("relax.add"): _add,
    ir.Op.get("relax.subtract"): _subtract,
    ir.Op.get("relax.multiply"): _multiply,
    ir.Op.get("relax.divide"): _divide,
    ir.Op.get("relax.floor_divide"): _floor_divide,
    ir.Op.get("relax.sin"): _sin,
    ir.Op.get("relax.cos"): _cos,
    ir.Op.get("relax.sqrt"): _sqrt,
    ir.Op.get("relax.sigmoid"): _sigmoid,
    ir.Op.get("relax.less"): _less,
    ir.Op.get("relax.nn.relu"): _nn_relu,
    ir.Op.get("relax.nn.gelu"): _nn_gelu,
    ir.Op.get("relax.nn.silu"): _nn_silu,
    ir.Op.get("relax.reshape"): _reshape,
    ir.Op.get("relax.permute_dims"): _permute_dims,
    ir.Op.get("relax.concat"): _concat,
    ir.Op.get("relax.expand_dims"): _expand_dims,
    # ir.Op.get("relax.cumsum"): _cumsum,
    ir.Op.get("relax.tril"): _tril,
    ir.Op.get("relax.triu"): _triu,
    ir.Op.get("relax.astype"): _astype,
    ir.Op.get("relax.take"): _take,
    ir.Op.get("relax.full"): _full,
    ir.Op.get("relax.full_like"): _full_like,
    ir.Op.get("relax.ones"): _ones,
    ir.Op.get("relax.ones_like"): _ones_like,
    ir.Op.get("relax.zeros"): _zeros,
    ir.Op.get("relax.zeros_like"): _zeros_like,
    ir.Op.get("relax.collapse_sum_like"): _collapse_sum_like,
    ir.Op.get("relax.collapse_sum_to"): _collapse_sum_to,
    ir.Op.get("relax.split"): _split,
    ir.Op.get("relax.strided_slice"): _strided_slice,
    ir.Op.get("relax.where"): _where,
    ir.Op.get("relax.broadcast_to"): _broadcast_to,
    ir.Op.get("relax.nn.max_pool2d"): _nn_max_pool2d,
    ir.Op.get("relax.nn.batch_norm"): _nn_batch_norm,
    ir.Op.get("relax.nn.layer_norm"): _nn_layer_norm,
    ir.Op.get("relax.matmul"): _matmul,
    ir.Op.get("relax.nn.softmax"): _nn_softmax,
    ir.Op.get("relax.flatten"): _flatten,
    ir.Op.get("relax.nn.adaptive_avg_pool2d"): _nn_adaptive_max_pool2d,
    ir.Op.get("relax.sum"): _sum,
    ir.Op.get("relax.mean"): _mean,
    ir.Op.get("relax.image.resize2d"): _image_resize2d,
    ir.Op.get("relax.tanh"): _tanh,
    ir.Op.get("relax.negative"): _negative,
    ir.Op.get("relax.log"): _log,
}


@mutator
class OperatorLegalizer(PyExprMutator):
    """The operator legalizer leverages CallTE of Relax BlockBuilder and
    the existing TOPI functions or newly written TE functions to lower
    high-level operator calls down to CallTIRs with TIR PrimFuncs.
    """

    # Todo(ruihang): better document the mutator

    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def _convert_op(self, call: Call) -> Call:
        if call.op in op_legalization_map:
            # We only transform the op calls with known shape values
            if not all(
                [has_known_shape_value(arg.struct_info) for arg in call.args]
            ) or not has_known_shape_value(call.struct_info):
                return call
            return op_legalization_map[call.op](self.builder_, call)

        if call.op.name != "relax.call_tir":
            logging.info("No legalization func for %s is found.", call.op.name)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, Function):
                continue
            updated_func = self.visit_expr(func)
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):  # pylint: disable=arguments-differ
        call = self.visit_expr_post_order(call)
        return self._convert_op(call)
