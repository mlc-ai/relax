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
void _HGEMM(NDArray A, NDArray B, NDArray Bias, NDArray C) {
  // A: [M, K], B: [K, N], BIAS: [1, N], C: [M, N]
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(Bias->ndim, 2);
  CHECK_EQ(C->ndim, 2);
  CHECK_EQ(A->shape[1], B->shape[0]);
  int M = A->shape[0];
  int K = A->shape[1];
  int N = B->shape[1];
  CHECK_EQ(C->shape[0], M);
  CHECK_EQ(C->shape[1], N);
  CHECK_EQ(Bias->shape[0], 1);
  CHECK_EQ(Bias->shape[1], N);
  CHECK_EQ(A.DataType(), DataType::Float(16));
  CHECK_EQ(B.DataType(), DataType::Float(16));
  CHECK_EQ(Bias.DataType(), DataType::Float(16));
  CHECK_EQ(C.DataType(), DataType::Float(16));

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,

    cutlass::epilogue::thread::LinearCombinationRelu<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
  >;

  Gemm gemm_op;

  cutlass::half_t alpha(1.0);
  cutlass::half_t beta(0.0);
  cutlass::layout::ColumnMajor::Stride::Index lda(K);
  cutlass::layout::ColumnMajor::Stride::Index ldb(N);
  cutlass::layout::ColumnMajor::Stride::Index ld_bias(N);
  cutlass::layout::ColumnMajor::Stride::Index ldc(N);
  cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
  cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
  cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
  cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);

  cutlass::Status status = gemm_op({
      {M, N, K},            // GemmCoord problem_size_
      {a, lda},             // TensorRef<ElementA const, LayoutA> ref_A_
      {b, ldb},             // TensorRef<ElementB const, LayoutB> ref_B_
      {bias, ld_bias},      // TensorRef<ElementC const, LayoutC> ref_C_
      {c, ldc},             // TensorRef<ElementC, LayoutC> ref_D_
      {alpha, beta}         // typename EpilogueOutputOp::Params epilogue_
  });
  CHECK(status == cutlass::Status::kSuccess);
}

}  // namespace
TVM_DLL_EXPORT_TYPED_FUNC(HGEMM, _HGEMM);
