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

// simple specialized impl, can be replaced by
// call into libraries.
void _HGEMM(NDArray A, NDArray B, NDArray C) {
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
TVM_DLL_EXPORT_TYPED_FUNC(HGEMM, _HGEMM);
