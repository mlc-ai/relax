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
import math

import tvm
from tvm import relax, te, topi
from tvm.ir.module import IRModule
from tvm.relax.block_builder import BlockBuilder


@relax.expr_functor.mutator
class OperatorLegalizer(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def _convert_op(self, call: relax.Call) -> relax.Call:
        bb: BlockBuilder = self.builder_
        op = call.op
        attrs = call.attrs
        args = call.args

        if op.name == "relax.nn.conv2d":
            return bb.call_te(
                topi.nn.conv2d,
                args[0],
                args[1],
                attrs.strides,
                attrs.padding,
                attrs.dilation,
                attrs.data_layout,
                attrs.kernel_layout,
            )
        elif op.name == "relax.add":
            return bb.call_te(topi.add, args[0], args[1])
        elif op.name == "relax.subtract":
            return bb.call_te(topi.subtract, args[0], args[1])
        elif op.name == "relax.multiply":
            return bb.call_te(topi.multiply, args[0], args[1])
        elif op.name == "relax.divide":
            return bb.call_te(topi.divide, args[0], args[1])
        elif op.name == "relax.floor_divide":
            return bb.call_te(topi.floor_divide, args[0], args[1])
        elif op.name == "relax.sin":
            return bb.call_te(topi.sin, args[0])
        elif op.name == "relax.cos":
            return bb.call_te(topi.cos, args[0])
        elif op.name == "relax.sqrt":
            return bb.call_te(topi.sqrt, args[0])
        elif op.name == "relax.nn.relu":
            return bb.call_te(topi.nn.relu, args[0])
        elif op.name == "relax.nn.gelu":

            def te_gelu(x):
                return te.compute(
                    x.shape,
                    lambda *i: 0.5
                    * x(*i)
                    * (
                        1
                        + te.tanh(math.sqrt(2 / math.pi) * (x(*i) + 0.044715 * te.power(x(*i), 3)))
                    ),
                )

            return bb.call_te(te_gelu, args[0])
        elif op.name == "relax.nn.silu":
            sig = bb.emit_te(topi.sigmoid, args[0])
            return bb.call_te(topi.multiply, args[0], sig)
        elif op.name == "relax.reshape":
            return bb.emit_te(topi.reshape, args[0], args[1])
        elif op.name == "relax.transpose":
            return bb.emit_te(topi.transpose, args[0], attrs.axes)
        elif op.name == "relax.concatenate":
            n_field = len(args[0].shape_.fields)
            fields = []
            for i in range(n_field):
                fields.append(bb.emit(relax.TupleGetItem(args[0], i)))
            return bb.emit_te(
                topi.concatenate, fields, None if attrs.axis is None else attrs.axis.value
            )
        elif op.name == "relax.nn.layer_norm":

            def te_layer_norm(x, gamma, beta, axis, eps):
                shape_prod = tvm.tir.const(1, "int32")
                for dim in axis:
                    shape_prod = shape_prod * x.shape[dim.value]
                mean = topi.sum(x, axis=axis, keepdims=True) / shape_prod
                var = topi.sum((x - mean) * (x - mean), axis=axis, keepdims=True) / shape_prod
                return gamma * ((x - mean) / topi.sqrt(var + eps)) + beta

            return bb.emit_te(
                te_layer_norm, args[0], args[1], args[2], axis=attrs.axis, eps=attrs.epsilon
            )
        elif op.name == "relax.nn.softmax":
            return bb.emit_te(topi.nn.softmax, args[0], attrs.axis)
        elif op.name == "relax.sum":
            return bb.emit_te(topi.sum, args[0], attrs.axis, attrs.keepdims)
        elif op.name == "relax.mean":
            shape_prod = tvm.tir.const(1, "int32")
            axis = attrs.axis if attrs.axis is not None else range(0, len(args[0].shape))
            for dim in axis:
                shape_prod = shape_prod * args[0].shape[dim.value]
            sum_var = bb.emit_te(topi.sum, args[0], axis, attrs.keepdims)
            return bb.emit_te(topi.divide, sum_var, shape_prod)

        if op.name != "relax.call_tir":
            print(op.name)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        return self._convert_op(call)
