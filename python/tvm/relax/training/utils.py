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
"""Utility functions for relax training."""

from ..expr import Function
from . import _ffi_api


def append_loss(orig_func: Function, loss_func: Function) -> Function:
    """Local helper to append a specified loss function after the original function.

    In detail, the result function has the arguments list of orig_func and the combination
    of their body, which passes the return values of orig_func as the arguments of loss_func. For
    those arguments of loss_func which are not mapped to some return values, they will be lifted
    and appended to the argument list of result function.

    Note
    -------
    1. This uitl is dedicated to loss functions, not for general purposes.
    2. This util can be replaced if we have Inline pass. It is equivalent to inline a tail call in
    some sense.

    Example
    -------
    >>> @R.function
    ... def orig(x: R.Tensor((2, 4), "float32"), y: R.Tensor((2, 4), "float32")):
    ...     with R.dataflow():
    ...         out = R.add(x, y)
    ...         R.output(out)
    ...     return out

    >>> @R.function
    ... def loss(predictions: R.Tensor((2, 4), "float32"), labels: R.Tensor((2, 4), "float32")):
    ...     with R.dataflow():
    ...         lv = R.subtract(predictions, labels)
    ...         lv1 = R.multiply(lv, lv)
    ...         gv = R.sum(lv1)
    ...         R.output(gv)
    ...     return gv

    >>> expected = append_loss(orig, loss)
    >>> print(expected)

    Will get

    >>> @R.function
    ... def expected(x: R.Tensor((2, 4), "float32"), y: R.Tensor((2, 4), "float32"),
    ...             labels: R.Tensor((2, 4), "float32")) -> R.Tensor((), "float32"):
    ...     with R.dataflow():
    ...         out: R.Tensor((2, 4), "float32") = R.add(x, y)
    ...         lv: R.Tensor((2, 4), "float32") = R.subtract(out, labels)
    ...         lv1: R.Tensor((2, 4), "float32") = R.multiply(lv, lv)
    ...         gv: R.Tensor((), "float32") = R.sum(lv1)
    ...         R.output(gv)
    ...     return gv

    Parameters
    ----------
    orig_func : Function
        The function to be appended to.

    loss_func : Function
        The loss function.

    Returns
    -------
    ret : Function
        The result function.
    """
    return _ffi_api.AppendLoss(orig_func, loss_func)  # type: ignore
