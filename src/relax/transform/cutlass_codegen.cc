/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ir/transform.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

const static constexpr char* kCutlassCodegen = "cutlass_codegen";
const static constexpr char* kCSource = "c_source";
const static constexpr char* kCSourceFmt = "c_source_fmt";
const static constexpr char* kCSourceFmtCuda = "cu";
const static constexpr char* kGeneratedCode = R"""(
#define CUTLASS_ENABLE_CUBLAS 1
#define CUTLASS_NAMESPACE cutlass
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
#define NDEBUG

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

namespace {

using namespace tvm;
using namespace tvm::runtime;

void _GEMM(NDArray A, NDArray B, NDArray C) {
  // A: [M, K], B: [K, N]
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);
  CHECK_EQ(A->shape[1], B->shape[0]);
  int M = A->shape[0];
  int K = A->shape[1];
  int N = B->shape[1];
  CHECK_EQ(C->shape[0], M);
  CHECK_EQ(C->shape[1], N);
  CHECK_EQ(A.DataType(), DataType::Float(16));
  CHECK_EQ(B.DataType(), DataType::Float(16));
  CHECK_EQ(C.DataType(), DataType::Float(16));
  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<cutlass::half_t,            // ElementA
                                           cutlass::layout::RowMajor,  // LayoutA
                                           cutlass::half_t,            // ElementB
                                           cutlass::layout::RowMajor,  // LayoutB
                                           cutlass::half_t,            // ElementOutput
                                           cutlass::layout::RowMajor   // LayoutOutput
                                           >;
  Gemm gemm_op;
  cutlass::half_t alpha(1.0);
  cutlass::half_t beta(0.0);
  cutlass::layout::ColumnMajor::Stride::Index lda(K);
  cutlass::layout::ColumnMajor::Stride::Index ldb(N);
  cutlass::layout::ColumnMajor::Stride::Index ldc(N);
  cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
  cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
  cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
  cutlass::Status status = gemm_op({
      {M, N, K},     //
      {a, lda},      //
      {b, ldb},      //
      {c, ldc},      //
      {c, ldc},      //
      {alpha, beta}  //
  });
  CHECK(status == cutlass::Status::kSuccess);
}

}  // namespace
TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _GEMM);
)""";

void StringReplace(std::string* subject, const std::string& search, const std::string& replace) {
  for (size_t pos = 0; (pos = subject->find(search, pos)) != std::string::npos;
       pos += replace.length()) {
    subject->replace(pos, search.length(), replace);
  }
}

ExternFunc CodegenWithCutlass(const tir::PrimFuncNode* pf, String global_symbol) {
  using namespace tvm::tir;
  std::string source = kGeneratedCode;
  StringReplace(&source, "{global_symbol}", global_symbol);
  ExternFunc ret(global_symbol);
  ret = WithAttrs(std::move(ret), Map<String, ObjectRef>{
                                      {String(kCSource), String(source)},
                                      {String(kCSourceFmt), String(kCSourceFmtCuda)},
                                  });
  return ret;
}

namespace transform {

Pass CutlassCodegen() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) -> IRModule {
    IRModuleNode* mod = m.CopyOnWrite();
    Map<GlobalVar, BaseFunc> functions;
    for (const auto& kv : mod->functions) {
      GlobalVar gv = kv.first;
      BaseFunc base_func = kv.second;
      if (const tir::PrimFuncNode* pf = base_func.as<tir::PrimFuncNode>()) {
        if (Optional<Integer> _ = pf->attrs.GetAttr<Integer>(kCutlassCodegen)) {
          functions.Set(gv, CodegenWithCutlass(pf, gv->name_hint));
          continue;
        }
      }
      functions.Set(gv, kv.second);
    }
    mod->functions = std::move(functions);
    return GetRef<IRModule>(mod);
  };
  return CreateModulePass(pass_func, 0, "CutlassCodegen", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CutlassCodegen").set_body_typed(CutlassCodegen);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
