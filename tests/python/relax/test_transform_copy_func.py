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
import pytest
import tvm
import tvm.script
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import ir as I, relax as R, tir as T


def test_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            gv = R.add(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            gv = R.add(x, y)
            return gv

        @R.function
        def main_new(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            gv = R.add(x, y)
            return gv

    After = relax.transform.CopyFunc(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_new"], Expected["main_new"])

    After_name = relax.transform.CopyFunc(Before.get_global_var("main"), "copied")(Before)
    assert_structural_equal(After_name["copied"], Expected["main_new"])


def test_params():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            gv = R.add(x, y)
            return gv

    After = relax.transform.CopyFunc(Before.get_global_var("main"))(Before)
    assert len(Before["main"].params) == len(After["main_new"].params)
    for i in range(len(After["main_new"].params)):
        assert Before["main"].params[i] != After["main_new"].params[i]
        assert After["main"].params[i] != After["main_new"].params[i]


if __name__ == "__main__":
    pytest.main([__file__])
