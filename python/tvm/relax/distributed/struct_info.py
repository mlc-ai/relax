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
# pylint: disable=redefined-builtin, invalid-name
"""Struct Info for distributed tensor."""
import tvm
from tvm.relax.struct_info import StructInfo, TensorStructInfo
from tvm.ir import Span
from tvm.runtime.object import Object

from .global_info import DeviceMesh
from . import _ffi_api


@tvm._ffi.register_object("relax.distributed.Placement")
class Placement(Object):
    """Describes how data is distributed in each dimension of the device mesh

    Parameters
    ----------
    text_format: str
        The text format of placement.
    """

    def __init__(self, text_format: str):
        self.__init_handle_by_constructor__(_ffi_api.Placement, text_format)  # type: ignore


@tvm._ffi.register_object("relax.DTensorStructInfo")
class DTensorStructInfo(StructInfo):
    """StructInfo of a Distributed Tensor value.

    Parameters
    ----------
    tensor_sinfo: TensorStructInfo
        The struct info inherited from TensorStructInfo
    device_mesh: DeviceMesh
        The device mesh of the tensor.
    placement: Placement
        The placement of the tensor among the device mesh

    """

    tensor_sinfo: TensorStructInfo
    device_mesh: DeviceMesh
    placement: Placement

    def __init__(
        self,
        tensor_sinfo: TensorStructInfo,
        device_mesh: DeviceMesh,
        placement: Placement,
        span: Span = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DTensorStructInfo, tensor_sinfo, device_mesh, placement, span  # type: ignore
        )
