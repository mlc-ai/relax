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
# pylint: disable=unused-import, redefined-builtin
"""Namespace for Tensor-level IR"""
from tvm.ir import PrimExpr
from tvm.runtime import const

from . import analysis, schedule, stmt_functor, transform, usmp
from .generic import cast
from .buffer import Buffer, DataProducer, decl_buffer
from .data_layout import BijectiveLayout, Layout, bijective_layout, layout
from .expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Any,
    Broadcast,
    BufferLoad,
    Call,
    CallEffectKind,
    Cast,
    CommReducer,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    IterVar,
    Let,
    Load,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    ProducerLoad,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
)
from .function import IndexMap, PrimFunc, TensorIntrin
from .op import (
    TVMBackendAllocWorkspace,
    TVMBackendFreeWorkspace,
    abs,
    acos,
    acosh,
    address_of,
    all,
    any,
    asin,
    asinh,
    assume,
    atan,
    atan2,
    atanh,
    call_cpacked,
    call_cpacked_lowered,
    call_extern,
    call_intrin,
    call_llvm_intrin,
    call_llvm_pure_intrin,
    call_packed,
    call_packed_lowered,
    call_pure_extern,
    ceil,
    ceildiv,
    clz,
    comm_reducer,
    copysign,
    cos,
    cosh,
    div,
    erf,
    exp,
    exp2,
    exp10,
    floor,
    floordiv,
    floormod,
    fmod,
    hypot,
    if_then_else,
    indexdiv,
    indexmod,
    isfinite,
    isinf,
    isnan,
    isnullptr,
    ldexp,
    likely,
    log,
    log1p,
    log2,
    log10,
    lookup_param,
    max,
    max_value,
    min,
    min_value,
    mma_fill,
    mma_store,
    nearbyint,
    nextafter,
    popcount,
    power,
    ptx_commit_group,
    ptx_cp_async,
    ptx_ldmatrix,
    ptx_mma,
    ptx_mma_sp,
    ptx_wait_group,
    q_multiply_shift,
    ret,
    round,
    rsqrt,
    shift_left,
    shift_right,
    sigmoid,
    sin,
    sinh,
    sqrt,
    sum,
    tan,
    tanh,
    trace,
    trunc,
    truncdiv,
    truncmod,
    tvm_access_ptr,
    tvm_bmma_sync,
    tvm_fill_fragment,
    tvm_load_matrix_sync,
    tvm_mma_sync,
    tvm_stack_alloca,
    tvm_stack_make_array,
    tvm_stack_make_shape,
    tvm_store_matrix_sync,
    tvm_struct_get,
    tvm_struct_set,
    tvm_thread_allreduce,
    tvm_throw_last_error,
    tvm_tuple,
    undef,
    vectorcombine,
    vectorhigh,
    vectorlow,
)
from .schedule import BlockScope, Schedule, ScheduleError, ScheduleState, StmtSRef
from .stmt import (
    Allocate,
    AllocateConst,
    AssertStmt,
    AttrStmt,
    Block,
    BlockRealize,
    BufferRealize,
    BufferRegion,
    BufferStore,
    DeclBuffer,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    LetStmt,
    MatchBufferRegion,
    Prefetch,
    ProducerRealize,
    ProducerStore,
    SeqStmt,
    Stmt,
    Store,
    While,
    stmt_list,
    stmt_seq,
    type_annotation,
)
