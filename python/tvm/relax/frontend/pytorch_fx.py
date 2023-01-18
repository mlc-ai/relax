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

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch FX frontend of Relax."""
from typing import Dict

from tvm import relax
import numpy as np  # type: ignore


class TorchFXTranslator:
    """A translator from PyTorch FX to Relax."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore
        from torch import fx

        self.env: Dict[fx.node.Node, relax.Expr] = {}
        self.params: Dict[torch.Tensor, relax.Constant] = {}
        self.params_transpose: Dict[torch.Tensor, relax.Constant] = {}
        self.named_modules: Dict[str, torch.Module] = None
        self.block_builder: relax.BlockBuilder = None
        self.create_convert_map()

    @staticmethod
    def _convert_data_type(input_type):
        """converts the PyTorch scalar type input_type to a TVM dtype."""

        input_type = input_type.lower()
        if input_type in ["float", "float32", "torch.float32"]:
            return "float32"
        elif input_type in ["float16", "torch.float16"]:
            return "float16"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _fetch_attr(model, target: str):
        import torch  # type: ignore

        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        if isinstance(attr_itr, torch.Tensor):
            return TorchFXTranslator._convert_torch_tensor_to_relax(attr_itr)
        return attr_itr

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        shape = tensor.data.shape
        dtype = TorchFXTranslator._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), relax.TensorStructInfo(shape, dtype))

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""
        import torch  # type: ignore

        if isinstance(tensor, relax.Var):
            return tensor.struct_info.shape
        elif isinstance(tensor, torch.Tensor):
            return tensor.shape
        elif isinstance(tensor, relax.Constant):
            return tensor.data.shape
        raise ValueError("Unsupported type: {}".format(type(tensor)))

    def retrive_args(self, node):
        return self._retrive_args(node.args)

    def _retrive_args(self, node):
        from torch import fx

        if isinstance(node, fx.node.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrive_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrive_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrive_args(k): self._retrive_args(v) for k, v in node.items()}
        else:
            return node

    @staticmethod
    def _promote_binary_op_args(lhs, rhs):
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return lhs, rhs
        elif isinstance(lhs, relax.Expr):
            assert isinstance(lhs.struct_info, relax.TensorStructInfo)
            return lhs, relax.const(rhs, lhs.struct_info.dtype)
        elif isinstance(rhs, relax.Expr):
            assert isinstance(rhs.struct_info, relax.TensorStructInfo)
            return relax.const(lhs, rhs.struct_info.dtype), rhs
        else:
            assert False

    def _call_binary_op(self, op, lhs, rhs):
        lhs, rhs = TorchFXTranslator._promote_binary_op_args(lhs, rhs)
        return self.block_builder.emit(op(lhs, rhs))

    def normalize_axes(self, axes, ndim):
        if not isinstance(axes, (tuple, list)):
            axes = [axes]
        axes = tuple(self._normalize_axis(axis, ndim) for axis in axes)
        return axes

    def _normalize_axis(self, axis, ndim):
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis %d is out of bounds for array of dimension %d" % (axis, ndim))
        return axis

    def _conv2d(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]

        conv2d = self.block_builder.emit(
            relax.op.nn.conv2d(
                x,
                weight,
                strides=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if module.bias is None:
            return conv2d

        bias = self.params[module.bias]
        if len(bias.data.shape) == 1:
            bias_data = bias.data.numpy().reshape(1, -1, 1, 1)
            reshaped_bias = relax.const(
                bias_data, relax.TensorStructInfo(bias_data.shape, bias.data.dtype)
            )
            bias = self.params[module.bias] = reshaped_bias

        return self.block_builder.emit(relax.op.add(conv2d, bias))

    def _linear(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        if module.weight not in self.params_transpose:
            self.params_transpose[module.weight] = self._convert_torch_tensor_to_relax(
                module.weight.T
            )

        weight_T = self.params_transpose[module.weight]
        dense = self._matmul_impl(x, weight_T)

        if module.bias is None:
            return dense

        bias = self.params[module.bias]
        if len(bias.data.shape) == 1:
            bias_data = bias.data.numpy().reshape(1, -1)
            reshaped_bias = relax.const(
                bias_data, relax.TensorStructInfo(bias_data.shape, bias.data.dtype)
            )
            bias = self.params[module.bias] = reshaped_bias

        return self.block_builder.emit(relax.op.add(dense, bias))

    def _relu(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.nn.relu(x))

    def _max_pool2d(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if node.target in self.named_modules:
            module = self.named_modules[node.target]
            kernel = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
        else:
            nargs = len(node.args)
            kernel = node.args[1] if nargs > 1 else node.kwargs["kernel_size"]
            stride = node.args[2] if nargs > 2 else node.kwargs["stride"]
            padding = node.args[3] if nargs > 3 else node.kwargs["padding"]
            dilation = node.args[4] if nargs > 4 else node.kwargs["dilation"]

        kernel = kernel if isinstance(kernel, (list, tuple)) else (kernel, kernel)
        stride = kernel if stride is None else stride
        stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)
        padding = (
            padding if isinstance(padding, (list, tuple)) else (padding, padding, padding, padding)
        )
        dilation = dilation if isinstance(dilation, (list, tuple)) else (dilation, dilation)

        return self.block_builder.emit(
            relax.op.nn.max_pool2d(
                x,
                pool_size=kernel,
                strides=stride,
                padding=padding,
                dilation=dilation,
                layout="NCHW",
            )
        )

    def _embedding(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        x = self.block_builder.emit(relax.op.astype(x, "int32"))
        return self.block_builder.emit(relax.op.take(weight, x, axis=0))

    def _adaptive_avg_pool2d(self, node: fx.node.Node) -> relax.Var:
        module = self.named_modules[node.target]
        x = self.env[node.args[0]]

        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, module.output_size, layout="NCHW")
        )

    def _flatten(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if node.target in self.named_modules:
            module = self.named_modules[node.target]
            start_dim = module.start_dim
            end_dim = module.end_dim
        else:
            start_dim = node.args[1] if len(node.args) >= 2 else 0
            end_dim = node.args[2] if len(node.args) == 3 else -1
        assert start_dim == 1 and end_dim == -1
        return self.block_builder.emit(relax.op.flatten(x))

    def _batch_norm_2d(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        weight = self.params[module.weight]
        bias = self.params[module.bias]
        dtype = self._convert_data_type(str(module.running_mean.dtype))
        running_mean = relax.const(module.running_mean.cpu().detach().numpy(), dtype)
        running_var = relax.const(module.running_var.cpu().detach().numpy(), dtype)
        eps = module.eps

        res_tuple = self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
            )
        )

        return self.block_builder.emit(relax.TupleGetItem(res_tuple, 0))

    def _add(self, node: fx.node.Node) -> relax.Expr:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.add, lhs, rhs)
        return lhs + rhs

    def _sub(self, node: fx.node.Node) -> relax.Expr:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.subtract, lhs, rhs)
        return lhs - rhs

    def _size(self, node: fx.node.Node) -> relax.Expr:
        x = self.env[node.args[0]]
        shape = self.shape_of(x)
        if len(node.args) == 1:
            assert isinstance(shape, relax.ShapeExpr)
            return tuple(relax.const(s.value) for s in shape.values)
        assert len(node.args) == 2
        idx = node.args[1]
        return self.shape_of(x)[idx].value

    def _type(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        return self.block_builder.emit(relax.op.astype(args[0], args[1]))

    def _getattr(self, node: fx.node.Node) -> relax.Var:
        if isinstance(self.env[node.args[0]], relax.Expr):
            if node.args[1] == "dtype":
                return self.env[node.args[0]].struct_info.dtype
            elif node.args[1] == "shape":
                return self.shape_of(self.env[node.args[0]])
        return getattr(self.env[node.args[0]], node.args[1])

    def _getitem(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.struct_info, relax.TensorStructInfo)
            begin = []
            end = []
            stride = []
            axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            for index in node.args[1]:
                if isinstance(index, int):
                    begin.append(index)
                    end.append(index + 1)
                    stride.append(1)
                    axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    begin.append(0 if index.start is None else index.start)
                    end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                begin.append(0)
                end.append(shape[i])
                axes.append(i)
                i = i + 1
            sliced = self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        else:
            assert False

    def _mul(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.multiply, lhs, rhs)
        else:
            return lhs * rhs

    def _sin(self, node: fx.node.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.sin(self.env[node.args[0]]))

    def _cos(self, node: fx.node.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.cos(self.env[node.args[0]]))

    def _sqrt(self, node: fx.node.Node) -> relax.Var:
        arg = self.env[node.args[0]]
        if isinstance(arg, (int, float)):
            arg = relax.const(arg, "float32")
        return self.block_builder.emit(relax.op.sqrt(arg))

    def _cat(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        return self.block_builder.emit(relax.op.concat(args[0], axis=node.kwargs["dim"]))

    def _truediv(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.divide, lhs, rhs)
        else:
            return lhs / rhs

    def _floordiv(self, node: fx.node.Node) -> relax.Var:
        lhs, rhs = self.retrive_args(node)
        if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
            return self._call_binary_op(relax.op.floor_divide, lhs, rhs)
        else:
            return lhs // rhs

    def _matmul(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        res = self._matmul_impl(
            args[0],
            args[1],
        )
        return res

    def _matmul_impl(self, a: relax.Expr, b: relax.Expr):
        return self.block_builder.emit(relax.op.linear_algebra.matmul(a, b, out_dtype="float32"))

    def _gelu(self, node: fx.node.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.nn.gelu(self.env[node.args[0]]))

    def _interpolate(self, node: fx.node.Node) -> relax.Var:
        # torch.nn.functional.interpolate(
        #   input, size=None, scale_factor=None, mode='nearest', align_corners=None,
        #   recompute_scale_factor=None, antialias=False)
        # (TODO) this is a temporary implementation for interpolate that only considers NCHW layout
        # it basically replicates the implementation in tvm.relay.frontend.pytorch
        data = self.env[node.args[0]]
        size = node.kwargs["size"]
        scale_factor = node.kwargs["scale_factor"]
        method = node.kwargs["mode"]
        align_corners = node.kwargs["align_corners"]
        recompute_scale_factor = node.kwargs["recompute_scale_factor"]
        antialias = node.kwargs["antialias"]

        assert recompute_scale_factor is None
        assert antialias is False

        if size is None:
            shape = self.shape_of(data)
            assert isinstance(shape, relax.ShapeExpr)
            size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        if method.startswith("nearest"):
            method = "nearest_neighbor"
        elif method[0:2] == "bi":
            method = method[2:]

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return self.block_builder.emit(
            relax.op.image.resize2d(
                data, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _addmm(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]
        z = self.env[node.args[2]]
        matmul = self.block_builder.emit(relax.op.linear_algebra.matmul(y, z, out_dtype="float32"))
        return self.block_builder.emit(relax.op.add(x, matmul))

    def _split(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        split_size = node.args[1]
        if "dim" in node.kwargs:
            dim = node.kwargs["dim"]
        else:
            dim = 0
        n_section = self.shape_of(x)[dim].value // split_size
        return self.block_builder.emit(relax.op.split(x, n_section, dim))

    def _tril(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        k = node.args[1] if len(node.args) > 1 else 0
        assert isinstance(k, int)
        return self.block_builder.emit(relax.op.create.tril(x, k))

    def _new_ones(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        self_var = args[0]
        size = args[1:]
        if not isinstance(size, (list, tuple)):
            size = (size,)
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, self_var.struct_info.dtype),
                self_var.struct_info.dtype,
            )
        )

    def _expand(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        return self.block_builder.emit(relax.op.broadcast_to(args[0], args[1:]))

    def _float(self, node: fx.node.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float32"))

    def _half(self, node: fx.node.Node) -> relax.Var:
        return self.block_builder.emit(relax.op.astype(self.env[node.args[0]], "float16"))

    def _permute(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        return self.block_builder.emit(relax.op.permute_dims(args[0], args[1:]))

    def _reshape(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        infer_idx = -1
        prod = 1
        for i in range(1, len(args)):
            if args[i] == -1:
                infer_idx = i
            else:
                prod *= args[i]
        if infer_idx != -1:
            args[infer_idx] = np.prod(args[0].shape).value // prod
        return self.block_builder.emit(relax.op.reshape(args[0], args[1:]))

    def _transpose(self, node: fx.node.Node) -> relax.Var:
        args = self.retrive_args(node)
        full_idx = list(range(len(self.shape_of(args[0]))))
        full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
        return self.block_builder.emit(relax.op.permute_dims(args[0], full_idx))

    def _softmax(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if node.target in self.named_modules:
            module = self.named_modules[node.target]
            dim = module.dim
        else:
            nargs = len(node.args)
            dim = node.args[1] if nargs > 1 else node.kwargs["dim"]
        assert dim is not None
        return self.block_builder.emit(relax.op.nn.softmax(x, dim))

    def _view(self, node: fx.node.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrive_args(node)
        if isinstance(args[1], (torch.Size, tuple, list)):
            return self.block_builder.emit(relax.op.reshape(args[0], tuple(args[1])))
        return self.block_builder.emit(relax.op.reshape(args[0], args[1:]))

    def _silu(self, node: fx.node.Node) -> relax.Var:
        x = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.nn.silu(x))

    def _group_norm(self, node: fx.node.Node) -> relax.Var:
        # torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05,
        #                    affine=True, device=None, dtype=None)
        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        num_groups = module.num_groups
        num_channels = module.num_channels
        eps = module.eps
        affine = module.affine

        shape = self.shape_of(x)
        assert len(shape) == 4
        N, C, H, W = shape[0], shape[1], shape[2], shape[3]
        assert C == num_channels
        assert C % num_groups == 0
        grouped_x = self.block_builder.emit(
            relax.op.reshape(x, [N, num_groups, C // num_groups, H, W])
        )
        mean_x = self.block_builder.emit(relax.op.mean(grouped_x, [2, 3, 4], keepdims=True))
        sub_x = self.block_builder.emit(relax.op.subtract(grouped_x, mean_x))
        square_x = self.block_builder.emit(relax.op.multiply(sub_x, sub_x))
        sum_square_x = self.block_builder.emit(relax.op.sum(square_x, [2, 3, 4], keepdims=True))
        var_x = self._call_binary_op(relax.op.divide, sum_square_x, (C // num_groups * H * W).value)
        var_x_eps = self._call_binary_op(relax.op.add, var_x, eps)
        std_x = self.block_builder.emit(relax.op.sqrt(var_x_eps))
        norm_x = self.block_builder.emit(relax.op.divide(sub_x, std_x))

        if affine:
            weight = self.params[module.weight]
            bias = self.params[module.bias]
            weight_reshape = self.block_builder.emit(
                relax.op.reshape(weight, (1, num_groups, C // num_groups, 1, 1))
            )
            bias_reshape = self.block_builder.emit(
                relax.op.reshape(bias, (1, num_groups, C // num_groups, 1, 1))
            )
            norm_x = self.block_builder.emit(relax.op.multiply(norm_x, weight_reshape))
            norm_x = self.block_builder.emit(relax.op.add(norm_x, bias_reshape))
        return self.block_builder.emit(relax.op.reshape(norm_x, (N, C, H, W)))

    def _layer_norm(self, node: fx.node.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]

        if module.elementwise_affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.normalized_shape), x.checked_type)
            beta = relax.const(torch.zeros_like(module.normalized_shape), x.checked_type)
        dim_num = len(module.normalized_shape)
        axes = list(range(-dim_num, 0))

        return self.block_builder.emit(
            relax.op.nn.layer_norm(
                x,
                gamma,
                beta,
                axes=axes,
                epsilon=module.eps,
            )
        )

    def create_convert_map(self):
        from torch import nn

        self.convert_map = {
            # call_module
            nn.Conv2d: self._conv2d,
            nn.Linear: self._linear,
            nn.ReLU: lambda node: self.block_builder.emit(relax.op.nn.relu(self.env[node.args[0]])),
            nn.MaxPool2d: self._max_pool2d,
            nn.AdaptiveAvgPool2d: self._adaptive_avg_pool2d,
            nn.Flatten: self._flatten,
            nn.BatchNorm2d: self._batch_norm_2d,
            nn.modules.sparse.Embedding: self._embedding,
            nn.Dropout: lambda node: self.env[node.args[0]],
            nn.LayerNorm: self._layer_norm,
            nn.SiLU: self._silu,
            nn.GroupNorm: self._group_norm,
            nn.Softmax: self._softmax,
            # call_function
            "add": self._add,
            "sub": self._sub,
            "flatten": self._flatten,
            "size": self._size,
            "unsqueeze": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "getattr": self._getattr,
            "getitem": self._getitem,
            "mul": self._mul,
            "sin": self._sin,
            "cos": self._cos,
            "sqrt": self._sqrt,
            "cat": self._cat,
            "truediv": self._truediv,
            "floordiv": self._floordiv,
            "matmul": self._matmul,
            "gelu": self._gelu,
            "interpolate": self._interpolate,
            "addmm": self._addmm,
            "split": self._split,
            "tril": self._tril,
            # call_method
            "new_ones": self._new_ones,
            "expand": self._expand,
            "float": self._float,
            "half": self._half,
            "permute": self._permute,
            "reshape": self._reshape,
            "transpose": self._transpose,
            "softmax": self._softmax,
            "view": self._view,
            "type": self._type,
            "contiguous": lambda node: self.env[node.args[0]],
        }

    def fetch_attr(self, model, target: str):
        target_atoms = target.split(".")
        attr_itr = model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def from_pytorch(self, model, input_info):
        """Convert a PyTorch model to a Relax program.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to convert.

        input_info : Dict[str, Tuple[Tuple[int], str]]
            A dictionary mapping input names to their shapes and data types.

        Returns
        -------
        module : tvm.ir.module.IRModule
            The converted Relax program.
        """
        from torch import fx

        self.named_modules = dict(model.named_modules())

        # fx.symbolic_trace(model).graph.print_tabular()

        graph: fx.Graph = fx.Tracer().trace(model)
        # Create input variables.
        inputs = {}
        for name, (shape, dtype) in input_info.items():
            input_var = relax.Var(name, relax.TensorStructInfo(shape, dtype))
            inputs[name] = input_var

        # Translate model parameters.
        for _, param in model.named_parameters():
            shape = param.data.shape
            dtype = self._convert_data_type(str(param.data.dtype))
            if dtype in ("float32", "float16"):
                self.params[param] = relax.const(
                    param.data.cpu().numpy(), relax.TensorStructInfo(shape, dtype)
                )
            else:
                raise ValueError("Unsupported data type for model parameters: %s" % dtype)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        with self.block_builder.function(name="main", params=list(inputs.values())):
            output = None
            with self.block_builder.dataflow():
                for node in graph.nodes:
                    if node.op == "placeholder":
                        assert node.name in inputs, "The function input {} is not found".format(
                            node.name
                        )
                        self.env[node] = inputs[node.name]
                    elif node.op == "output":
                        output = self.block_builder.emit_output(self.env[node.args[0]])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = TorchFXTranslator._fetch_attr(model, node.target)
                    elif node.op == "call_module":
                        module = self.named_modules[node.target]
                        assert (
                            type(module) in self.convert_map
                        ), f"Unsupported module type {type(module)}"
                        self.env[node] = self.convert_map[type(module)](node)
                    elif node.op == "call_function":
                        func_name = node.name.rstrip("0123456789_")
                        assert (
                            func_name in self.convert_map
                        ), f"Unsupported function type {func_name}"
                        self.env[node] = self.convert_map[func_name](node)
                    elif node.op == "call_method":
                        assert (
                            node.target in self.convert_map
                        ), f"Unsupported function target {node.target}"
                        self.env[node] = self.convert_map[node.target](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        return self.block_builder.get()
