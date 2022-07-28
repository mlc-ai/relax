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

from . import _make
from ...expr import Expr


def dense(lhs: Expr, rhs: Expr) -> Expr:
    return _make.dense(lhs, rhs)


def conv2d(
    lhs: Expr, rhs: Expr, kernel_size, stride=(1, 1), padding=[0, 0], dilation=[1, 1]
) -> Expr:
    return _make.conv2d(lhs, rhs, kernel_size, stride, padding, dilation)


def relu(data: Expr) -> Expr:
    return _make.relu(data)


def softmax(data: Expr) -> Expr:
    return _make.softmax(data)


def flatten(data: Expr) -> Expr:
    return _make.flatten(data)


def max_pool2d(data: Expr, kernel_size, stride=None, padding=(0, 0), dilation=(1, 1)) -> Expr:
    if stride is None:
        stride = kernel_size
    return _make.max_pool2d(data, kernel_size, stride, padding, dilation)

