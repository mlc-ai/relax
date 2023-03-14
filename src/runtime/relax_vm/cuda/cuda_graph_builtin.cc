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

/*!
 * \file src/runtime/relax_vm/cuda_graph_builtin.cc
 * \brief The CUDA graph related builtin functions for Relax virtual machine.
 */

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

#include "../../cuda/cuda_common.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

using tvm::runtime::NDArray;

/*! \brief Represents a CUDA graph. */
class CUDAGraphNode : public Object {
 public:
  cudaGraph_t handle_ = nullptr;

  ~CUDAGraphNode() {
    if (handle_ != nullptr) {
      cudaGraphDestroy(handle_);
    }
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(CUDAGraphNode, Object);
};

/*!
 * \brief Managed reference to CUDAGraphNode
 * \sa CUDAGraphNode
 */
class CUDAGraph : public ObjectRef {
 public:
  explicit CUDAGraph(cudaGraph_t handle) {
    auto n = make_object<CUDAGraphNode>();
    n->handle_ = handle;
    data_ = std::move(n);
  }
  TVM_DEFINE_OBJECT_REF_METHODS(CUDAGraph, ObjectRef, CUDAGraphNode);
};

/*! \brief The cache states of a CUDA graph. */
class CUDAGraphCache : public Object {
 public:
  struct Entry {
    /*! \brief The tensors allocated in the cuda graph */
    Array<ObjectRef> alloc_storages;
    /*! \brief Intemediate tensors in the capture func that will be used outside the capture func */
    Array<ObjectRef> states;
    /*! \brief The cuda graph instance */
    CUDAGraph graph;
  };

  static CUDAGraphCache* Get() { return dmlc::ThreadLocalStore<CUDAGraphCache>::Get(); }

  /*! \brief Get the captured states of a cuda graph from the cache or run CUDA graph capture
   * \param vm The virutal machine.
   * \param alloc_func () -> Tuple[Tensor0, ..., TensorN]. The function to allocate tensors.
   * \param capture_func (Tuple[Tensor0, ..., TensorN]) -> (). The function to capture a cuda graph.
   * \return The cache entry.
   */
  Entry GetOrCapture(VirtualMachine* vm, const ObjectRef& alloc_func,
                     const ObjectRef& capture_func) {
    if (auto it = entries_.find(capture_func); it != entries_.end()) {
      return it->second;
    }

    cudaStream_t capture_stream;
    CUDA_CALL(cudaStreamCreate(&capture_stream));
    CUDAGraphCache::Entry entry;

    // Invoke the alloc function
    TVMArgs alloc_func_args(nullptr, nullptr, 0);
    TVMRetValue alloc_func_rv;
    vm->InvokeClosurePacked(alloc_func, alloc_func_args, &alloc_func_rv);
    entry.alloc_storages = alloc_func_rv;

    // Set up arguments for the graph execution
    std::vector<TVMValue> values(1);
    std::vector<int> tcodes(1);
    TVMArgsSetter setter(values.data(), tcodes.data());
    setter(0, entry.alloc_storages);
    TVMRetValue capture_func_rv;

    // Warm up run
    vm->InvokeClosurePacked(capture_func, TVMArgs(values.data(), tcodes.data(), 1),
                            &capture_func_rv);

    // Run the graph in capture mode
    cudaGraph_t graph;
    std::swap(capture_stream, CUDAThreadEntry::ThreadLocal()->stream);
    CUDA_CALL(cudaStreamBeginCapture(CUDAThreadEntry::ThreadLocal()->stream,
                                     cudaStreamCaptureModeGlobal));

    vm->InvokeClosurePacked(capture_func, TVMArgs(values.data(), tcodes.data(), 1),
                            &capture_func_rv);
    entry.states = capture_func_rv;
    CUDA_CALL(cudaStreamEndCapture(CUDAThreadEntry::ThreadLocal()->stream, &graph));
    std::swap(capture_stream, CUDAThreadEntry::ThreadLocal()->stream);

    entry.graph = CUDAGraph(graph);
    entries_[capture_func] = entry;
    CUDA_CALL(cudaStreamDestroy(capture_stream));
    return entry;
  }

 private:
  /*! \brief The cache entries. The key is the (unique) function for the CUDA graph capture.
   * The value is the cached allocations and the captured CUDA graph instance.
   *
   * @TODO(wuwei): refacotr to use global index to speed up the lookup.
   */
  std::unordered_map<ObjectRef, Entry, ObjectPtrHash, ObjectPtrEqual> entries_;
};

TVM_REGISTER_GLOBAL("vm.builtin.get_captured_cuda_graph")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 3);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      ObjectRef alloc_func = args[1];  // () -> Tuple[Tensor0, ... TensorN]
      ObjectRef capture_func =
          args[2];  // (Tuple[Tensor0, ... TensorN]) -> Tuple[Tensor0, ... TensorN']

      CUDAGraphCache* cache = CUDAGraphCache::Get();
      auto cached = cache->GetOrCapture(vm, alloc_func, capture_func);
      *rv = Array<ObjectRef>{cached.graph, cached.alloc_storages, cached.states};
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
