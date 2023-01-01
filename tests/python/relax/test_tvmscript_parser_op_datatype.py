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

from typing import Optional, Union

import tvm
import tvm.testing
from tvm import IRModule, relax
from tvm.script.parser import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.parse(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_astype():
    @R.function
    def expected(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float16"):
        gv: R.Tensor((2, 3, 4), "float16") = R.astype(x, "float16")
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.astype(x, "float16"))
        bb.emit_func_output(gv)

    _check(expected, bb.get()["main"])


def test_wrap_param():
    metadata = tvm.ir.load_json(
        {
            "root": 1,
            "nodes": [
                {"type_key": ""},
                {"type_key": "Map", "keys": ["relax.expr.Constant"], "data": [2]},
                {"type_key": "Array", "data": [3]},
                {
                    "type_key": "relax.expr.Constant",
                    "attrs": {
                        "_checked_type_": "11",
                        "data": "0",
                        "span": "0",
                        "struct_info_": "4",
                    },
                },
                {
                    "type_key": "relax.TensorStructInfo",
                    "attrs": {"dtype": "float16", "ndim": "2", "shape": "5", "span": "0"},
                },
                {
                    "type_key": "relax.expr.ShapeExpr",
                    "attrs": {
                        "_checked_type_": "10",
                        "span": "0",
                        "struct_info_": "9",
                        "values": "6",
                    },
                },
                {"type_key": "Array", "data": [7, 8]},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "2"}},
                {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "3"}},
                {
                    "type_key": "relax.ShapeStructInfo",
                    "attrs": {"ndim": "2", "span": "0", "values": "6"},
                },
                {"type_key": "relax.ShapeType", "attrs": {"ndim": "2", "span": "0"}},
                {
                    "type_key": "relax.DynTensorType",
                    "attrs": {"dtype": "float16", "ndim": "2", "span": "0"},
                },
            ],
            "b64ndarrays": [
                "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAgAAAAIQAQACAAAAAAAAAAMAAAAAAAAADAAAAAAAAAAAPABAAEIARABFAEY="
            ],
            "attrs": {"tvm_version": "0.11.dev0"},
        }
    )

    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float16", ndim=2):
        gv: R.Tensor((2, 3), "float16") = R.wrap_param(
            metadata["relax.expr.Constant"][0], "float16"
        )
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.wrap_param(metadata["relax.expr.Constant"][0], "float16"))
        bb.emit_func_output(gv)

    _check(expected, bb.get()["main"])


if __name__ == "__main__":
    tvm.testing.main()
