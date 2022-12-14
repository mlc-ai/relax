#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../cuda/cuda_common.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

class CUDAGraphNode : public Object {
 public:
  cudaGraph_t handle_;
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDAGraphNode, Object);
};

class CUDAGraph : public ObjectRef {
 public:
  explicit CUDAGraph(cudaGraph_t handle) {
    auto n = make_object<CUDAGraphNode>();
    n->handle_ = handle;
    data_ = std::move(n);
  }
  TVM_DEFINE_OBJECT_REF_METHODS(CUDAGraph, ObjectRef, CUDAGraphNode);
};

class CUDAGraphCache : public Object {
 public:
  struct Entry {
    std::vector<ObjectRef> states;
    CUDAGraph graph;
  };
  static std::unordered_map<String, Entry> cache_;
};

std::unordered_map<String, CUDAGraphCache::Entry> CUDAGraphCache::cache_;

struct CaptureContext {
  cudaGraph_t graph;
  cudaStream_t stream;
  bool is_warp_up;
};

CaptureContext* capture_context = nullptr;

TVM_REGISTER_GLOBAL("vm.builtin.get_captured_cuda_graph")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK(args.size() == 2);
      void* vm_ptr = args[0];
      VirtualMachine* vm = static_cast<VirtualMachine*>(vm_ptr);
      String func_name = args[1];
      if (auto it = CUDAGraphCache::cache_.find(func_name); it == CUDAGraphCache::cache_.end()) {
        CaptureContext ctx;

        const auto& device = vm->devices[0];  // FIXME: can't assume it's devices[0]
        auto* device_api = DeviceAPI::Get(device);

        CUDA_CALL(cudaStreamCreate(&ctx.stream));
        // ctx.stream = static_cast<cudaStream_t>(device_api->CreateStream(device));
        ctx.is_warp_up = true;
        capture_context = &ctx;

        PackedFunc func{nullptr};
        func = vm->GetFunction(func_name, GetObjectPtr<Object>(vm));
        ICHECK(func != nullptr);

        // warm up
        TVMArgs func_args(nullptr, nullptr, 0);
        TVMRetValue func_rv;
        func.CallPacked(func_args, &func_rv);

        // capture
        ctx.is_warp_up = false;
        func.CallPacked(func_args, &func_rv);

        ADT tensors = func_rv;
        CUDAGraphCache::Entry entry;
        for (size_t i = 0; i < tensors.size(); ++i) {
          NDArray tensor = Downcast<NDArray>(tensors[i]);
          entry.states.push_back(tensor);
        }
        entry.graph = CUDAGraph(ctx.graph);
        CUDAGraphCache::cache_[func_name] = entry;
        CUDA_CALL(cudaStreamDestroy(ctx.stream));
      }
      const auto& cached = CUDAGraphCache::cache_[func_name];
      *rv = ADT::Tuple(ObjectRef(cached.graph), ObjectRef(ADT::Tuple(cached.states)));
    });

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_begin_capture").set_body_typed([]() {
  ICHECK(capture_context != nullptr);
  if (capture_context->is_warp_up) {
    return;
  }
  std::swap(capture_context->stream, CUDAThreadEntry::ThreadLocal()->stream);
  CUDA_CALL(
      cudaStreamBeginCapture(CUDAThreadEntry::ThreadLocal()->stream, cudaStreamCaptureModeGlobal));
});

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_end_capture").set_body_typed([]() {
  ICHECK(capture_context != nullptr);
  if (capture_context->is_warp_up) {
    return;
  }
  CUDA_CALL(cudaStreamEndCapture(CUDAThreadEntry::ThreadLocal()->stream, &capture_context->graph));

  cudaGraphNode_t* nodes = NULL;
  size_t numNodes = 0;
  CUDA_CALL(cudaGraphGetNodes(capture_context->graph, nodes, &numNodes));
  LOG(INFO) << "Num of nodes in the cuda graph created using stream capture API = " << numNodes;
  std::swap(capture_context->stream, CUDAThreadEntry::ThreadLocal()->stream);
});

TVM_REGISTER_GLOBAL("vm.builtin.cuda_graph_launch").set_body_typed([](CUDAGraph cuda_graph) {
  cudaGraphExec_t cuda_graph_exec;
  CUDA_CALL(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph->handle_, NULL, NULL, 0));
  CUDA_CALL(cudaGraphLaunch(cuda_graph_exec, CUDAThreadEntry::ThreadLocal()->stream));
  CUDA_CALL(cudaGraphExecDestroy(cuda_graph_exec));
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
