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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
import tvm.testing
from tvm import te, tir
from tvm.meta_schedule.testing import te_workload
from tvm.script import tir as T
from tvm.tir.schedule.schedule import ScheduleError
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def matmul_before(
    A: T.Buffer((128, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((128, 127), "float32"),
) -> None:
    for i0, i1, i2 in T.grid(128, 127, 127):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_pad_matmul():
    sch = tir.Schedule(matmul_before, debug_mask="all")
    C = sch.get_block("C_shared")
    sch.pad_einsum(C, [1, 32, 32])
    sch.mod.show(black_format=False)
    # tvm.ir.assert_structural_equal(matmul_expected, sch.mod["main"])
    # verify_trace_roundtrip(sch, mod=matmul_before)


if __name__ == "__main__":
    test_pad_matmul()
    # tvm.testing.main()
