
from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import numpy as np
import tvm
from tvm import relax
from tvm.error import DiagnosticError
from tvm.relax.testing import transform
from tvm.relax.transform import OperatorLegalizer
from tvm.script import relax as R
import tvm.testing

x = relax.Var("x", [2, 3], relax.DynTensorType(ndim=2, dtype="float32"))
bb = relax.BlockBuilder()
with bb.function("main", [x]):
    gv = bb.emit(relax.op.ones((2, 3)))
    bb.emit_func_output(gv)

mod = OperatorLegalizer(bb.get()).transform()
