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
"""Relax Neural Network (NN) operators"""
from typing import List, Optional, Tuple, Union

import tvm
from tvm.ir.expr import PrimExpr
from tvm.relay.op.nn.utils import get_pad_tuple2d

from . import _ffi_api
from ...expr import Expr, ShapeExpr

PrimExprLike = Union[int, PrimExpr]


def dense(data, weight, units=None, out_dtype=""):
    r"""Dense operator.
    Applies a linear transformation

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : Expr
        The input data to the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)`.

    weight : Expr
        The weight expressions, 2-D matrix,
        of shape `(units, units_in)`.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense,
        of shape `(d_1, d_2, ..., d_n, units)`.

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.dense(data, weight, units, out_dtype)


def conv2d(
    data,
    weight,
    kernel_size,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size[0], kernel_size[1])`
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, y, x] = \sum_{dy, dx, k}
           \mbox{data}[b, k, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] *
           \mbox{weight}[c, k, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCHW` for data and `OIHW` for weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    data : Expr
        The input data to the operator.

    weight : Expr
        The weight expressions.

    strides : Optional[int, Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Number of groups for grouped convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    # TODO enforce 4-way padding in topi/nn/conv2d after #4644 merged
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)

    return _ffi_api.conv2d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def relu(data: Expr) -> Expr:
    """Rectified linear unit.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.relu(data)


def gelu(data: Expr) -> Expr:
    """Gaussian Error Linear Units function

    .. math::
       text{GELU}(x) = 0.5 * x * (1 + text{Tanh}(sqrt(2 / pi) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.gelu(data)


def silu(data: Expr) -> Expr:
    """Sigmoid Linear Unit function

    .. math::
       text{SILU}(x) = x * sigmoid(x)

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.silu(data)


def softmax(data: Expr, axis=-1) -> Expr:
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: Expr
        The input data to the operator.

    axis: int, optional
        The axis to sum over when computing softmax

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.softmax(data, axis)


def flatten(data: Expr) -> Expr:
    """Flatten.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : Expr
        The input data

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.flatten(data)


def max_pool2d(
    data: Expr,
    pool_size,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
) -> Expr:
    r"""2D maximum pooling operator.

    This operator takes data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w) and pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \max_{m=0, \ldots, kh-1} \max_{n=0, \ldots, kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = get_pad_tuple2d(padding)

    return _ffi_api.max_pool2d(
        data, pool_size, strides, padding, dilation, layout, out_layout, ceil_mode
    )


def batch_norm(
    data: Expr,
    gamma: Expr,
    beta: Expr,
    moving_mean: Expr,
    moving_var: Expr,
    axis: int = 1,
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
) -> Expr:
    r"""
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    .. math::

        data\_mean[i] = mean(data[:,i,:,...]) \\
        data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

        out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}}
            * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated by

    .. code:: python

        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
        moving_var = moving_var * momentum + data_var * (1 - momentum)

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.
    Specifying -1 sets the channel axis to be the last item in the input shape.

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relax.Expr
        Input to which batch_norm will be applied.

    gamma : tvm.relax.Expr
        The gamma scale factor.

    beta : tvm.relax.Expr
        The beta offset factor.

    moving_mean : tvm.relax.Expr
        Running mean of input,

    moving_var : tvm.relax.Expr
        Running variance of input.

    axis : int, default=1
        Specify along which shape axis the channel is specified.

    epsilon : float, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : bool, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : bool, default=True
        If true, multiply by gamma. If False, gamma is not used.
        When the next layer is piecewise linear (also e.g. nn.relu),
        this can be disabled since the scaling will be done by the next layer.

    Returns
    -------
    result : relax.Tuple([tvm.relax.Expr, tvm.relax.Expr, tvm.relax.Expr])
        Tuple of normed data (same shape as input),
        new running mean (k-length vector),
        and new running variance (k-length vector)
    """
    # Todo(ruihang): actually it returns a call to the batch-norm op
    return _ffi_api.batch_norm(
        data, gamma, beta, moving_mean, moving_var, axis, epsilon, center, scale
    )


def dropout(data: Expr, rate: float = 0.5) -> Expr:
    """Applies the dropout operation to the input array.

    During training, each element of the input is set to zero with
    probability ``p``. The whole array is rescaled by ``1/(1-p)``
    to keep the expected sum of the input unchanged.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    rate : float, default=0.5
        The probability for an element to be reset to 0.

    Returns
    -------
    result : relax.Expr
        The result of dropout, which is a tuple of two tensors.
        The first one is the original tensor and the second one is a
        mask tensor (1.0 where element not dropped, 0.0 where dropped)
    """
    return _ffi_api.dropout(data, rate)


def layer_norm(
    data: Expr,
    gamma: Expr,
    beta: Expr,
    axis: Union[int, List[int]] = -1,
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
):
    r"""
    Layer normalization (Lei Ba and et al., 2016).
    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
            * gamma + beta

    Unlike batch normalization, the mean and var are computed along the channel dimension.

    Assume the input has size k on axis 1, then both gamma and beta have shape (k,).

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : relax.Expr
        Input to which layer_norm will be applied.

    gamma : relax.Expr
        The gamma scale factor.

    beta : relax.Expr
        The beta offset factor.

    axis : Union[int, List[int]], default=-1
        The axes that should be normalized, typically the axis of the channels.

    epsilon : double, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, default=True
        If True, multiply by gamma. If False, gamma is not used.

    Returns
    -------
    result : relax.Expr
        The normalized data.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.layer_norm(data, gamma, beta, axis, epsilon, center, scale)


def matmul(a: Expr, b: Expr) -> Expr:
    """
    General matrix multiplication of two tensors.

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

    Returns
    -------
    result : relax.Expr
        The result of the matmul.
    """
    return _ffi_api.matmul(a, b)


def adaptive_avg_pool2d(
    data: Expr,
    output_size: Optional[Union[PrimExprLike, Tuple[PrimExprLike], List[PrimExprLike]]] = None,
    layout: str = "NCHW",
) -> Expr:
    r"""2D adaptive average pooling operator. This operator is experimental.

    This operator takes data as input and does 2D average value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size) for any input (NCHW).

        If a tuple of integers (height, width) are provided for output_size,
        the output size is (N x C x height x width) for any input (NCHW).

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    output_size : Optional[Union[PrimExprLike, Tuple[PrimExprLike], List[PrimExprLike]]]
        Output height and width.

    layout : str
        Layout of the input.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if output_size is not None:
        if isinstance(output_size, (PrimExpr, int)):
            output_size = [output_size]
        temp_size = []
        for shape in output_size:
            if isinstance(shape, PrimExpr):
                temp_size.append(shape)
            elif isinstance(shape, int):
                temp_size.append(tvm.tir.const(shape, "int32"))
            else:
                raise RuntimeError(
                    f"The input new shape of reshape operator contains unrecognized dimension {shape}"
                )
        output_size = temp_size
    return _ffi_api.adaptive_avg_pool2d(data, output_size, layout)

def gradrelu_(data: Expr) -> Expr:
    return _ffi_api.gradrelu_(data)

def cross_entropy(lhs: Expr, rhs: Expr) -> Expr:
    return _ffi_api.cross_entropy(lhs, rhs)

def softmax_cross_entropy(lhs: Expr, rhs: Expr) -> Expr:
    return _ffi_api.softmax_cross_entropy(lhs, rhs)

def sigmoid(data: Expr) -> Expr:
    return _ffi_api.sigmoid(data)

def tanh(data: Expr) -> Expr:
    return _ffi_api.tanh(data)
