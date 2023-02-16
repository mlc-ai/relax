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
# pylint: disable=invalid-name
"""Legalize high-level operator calls in Relax functions to call_tir."""
from .binary import *
from .creation import *
from .datatype import *
from .image import *
from .index import *
from .linear_algebra import *
from .manipulate import *
from .nn import *
from .search import *
from .statistical import *
from .unary import *

from .. import _ffi_api


def LegalizeOps():
    """Legalize high-level operator calls in Relax functions to call_tir
    with corresponding low-level TIR PrimFuncs.

    For each high-level operator, we register the way of legalizing it as a
    function, which takes a context BlockBuilder and the Call being legalized
    as input, and returns the legalized call. Here the input BlockBuilder is
    mainly used for adding the PrimFunc created by call_te into the context
    IRModule.
    """

    return _ffi_api.LegalizeOps()  # type: ignore
