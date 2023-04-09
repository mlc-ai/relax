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

#include <tvm/tir/op.h>

#include <optional>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check if buffer indices are all Vars and expr
 * \param buffer_access The BufferLoad or BufferStore
 * \return The indices if the indices are all Vars, otherwise NullOpt
 */
Optional<Array<Var>> CheckTrivialBufferIndices(const Array<PrimExpr>& buffer_access) {
  Array<Var> indices;
  for (const PrimExpr& index : buffer_access) {
    const VarNode* var = index.as<VarNode>();
    if (var == nullptr) {
      return NullOpt;
    }
    indices.push_back(GetRef<Var>(var));
  }
  return indices;
}

Optional<Array<Var>> CheckTrivialBufferAccess(const BufferRegion& buffer_region) {
  Array<Var> indices;
  indices.reserve(buffer_region->region.size());
  for (const Range& range : buffer_region->region) {
    if (!tir::is_one(range->extent)) {
      return NullOpt;
    }
    if (const auto* var = range->min.as<VarNode>()) {
      indices.push_back(GetRef<Var>(var));
    } else {
      return NullOpt;
    }
  }
  return indices;
}

/*! \brief The schedule error class when the padding size is invalid. */
class InvalidPaddingError : public ScheduleError {
 public:
  InvalidPaddingError(IRModule mod, Block block, Array<Integer> padding)
      : mod_(std::move(mod)), block_(std::move(block)), padding_(std::move(padding)) {}
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  String FastErrorString() const final {
    return "ScheduleError: The padding size for the block is invalid.";
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The padding for the block {0} are invalid. It should be a list of "
       << block_->iter_vars.size() << " non-negative integers. Got " << padding_;
    return os.str();
  }

  static void Check(const ScheduleState& self, const Block& block, Array<Integer> padding) {
    if (padding.size() != block->iter_vars.size()) {
      throw InvalidPaddingError(self->mod, block, padding);
    }
    for (const auto& pad : padding) {
      if (pad->value <= 0) {
        throw InvalidPaddingError(self->mod, block, padding);
      }
    }
  }

 private:
  IRModule mod_;
  Block block_;
  Array<Integer> padding_;
};

/*! \brief The schedule error class when the block body is not an Einsum pattern. */
class NonEinsumError : public ScheduleError {
 public:
  explicit NonEinsumError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  String FastErrorString() const final {
    return "ScheduleError: The block is not a computation of Einsum pattern.";
  }
  String DetailRenderTemplate() const final {
    return "The block {0} not a computation of Einsum pattern.";
  }

 private:
  IRModule mod_;
  Block block_;
};

/*! \brief Data structure that represents a Einsum computation. */
struct Einsum {
  // The output buffer
  Array<Buffer> output_buffers;
  // The indices of the output buffer
  Map<Buffer, Array<Var>> output_indices;
  // The input buffers
  Array<Buffer> input_buffers;
  // The indices of the input buffers
  Map<Buffer, Array<Var>> input_indices;
};

struct BufferPadding {
  Buffer buffer;
  Buffer padded_buffer;

  static BufferPadding FromBufferRegion(const BufferRegion& buffer_region,
                                        const Map<Var, PrimExpr>& iter_extents) {
    BufferPadding result;
    result.buffer = buffer_region->buffer;
    // Step 1. Calculate the shape of the padded buffer
    {
      Array<PrimExpr> shape;
      shape.reserve(buffer_region->region.size());
      for (const Range& range : buffer_region->region) {
        Var var = Downcast<Var>(range->min);
        PrimExpr extent = iter_extents[var];
        shape.push_back(extent);
      }
      result.padded_buffer = decl_buffer(shape, result.buffer->dtype, result.buffer->name + "_pad",
                                         result.buffer.scope());
    }
    LOG(INFO) << "buffer: " << result.buffer << ", old_shape = " << result.buffer->shape
              << ", new_shape = " << result.padded_buffer->shape;
    // padding.buffer = buffer;
    // padding.padded_buffer = padded_buffer;
    // return padding;
    return result;
  }

  Stmt MakeCopyBlock(bool is_read, arith::Analyzer* analyzer) {
    Array<Var> loop_vars;
    Array<Range> loop_doms;
    Array<IterVar> iter_vars;
    Array<Range> instance_dom;
    Array<PrimExpr> indices;
    int ndim = buffer->shape.size();
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim = padded_buffer->shape[i];
      Range dom = Range::FromMinExtent(IntImm(dim->dtype, 0), dim);
      loop_vars.push_back(Var("i" + std::to_string(i), dim->dtype));
      loop_doms.push_back(dom);
      IterVar iter_var(dom, Var("v" + std::to_string(i), dim->dtype), kDataPar);
      instance_dom.push_back(Range::FromMinExtent(iter_var->var, IntImm(dim->dtype, 1)));
      iter_vars.push_back(iter_var);
      indices.push_back(iter_var->var);
    }
    Stmt body{nullptr};
    if (is_read) {
      PrimExpr predicate = Bool(true);
      for (int i = 0; i < ndim; ++i) {
        if (!analyzer->CanProveEqual(buffer->shape[i], padded_buffer->shape[i])) {
          predicate = predicate && (indices[i] < buffer->shape[i]);
        }
      }
      PrimExpr rhs = BufferLoad(buffer, indices);
      body =
          BufferStore(padded_buffer, if_then_else(predicate, rhs, make_zero(rhs->dtype)), indices);
    } else {
      body = BufferStore(buffer, tvm::protected_write(BufferLoad(padded_buffer, indices)), indices);
    }
    body = BlockRealize(Array<PrimExpr>{loop_vars.begin(), loop_vars.end()}, Bool(true),
                        Block(iter_vars,                                                //
                              Array<BufferRegion>{BufferRegion(buffer, instance_dom)},  //
                              Array<BufferRegion>{BufferRegion(padded_buffer, instance_dom)},
                              padded_buffer->name, std::move(body)));
    for (int i = ndim - 1; i >= 0; --i) {
      body = For(loop_vars[i], loop_doms[i]->min, loop_doms[i]->extent, ForKind::kSerial,
                 std::move(body));
    }
    LOG(INFO) << "copy block:\n" << body;
    return body;
  }
};

Einsum ExtractEinsum(const ScheduleState& self, const Block& block) {
  Einsum result;
  std::unordered_set<const BufferNode*> buffer_used;
  int n_reads = block->reads.size();
  for (int i = 0; i < n_reads; ++i) {
    const Buffer& buffer = block->reads[i]->buffer;
    if (buffer_used.count(buffer.get()) != 0) {
      throw NonEinsumError(self->mod, block);
    }
    buffer_used.insert(buffer.get());
    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferAccess(block->reads[i])) {
      result.input_buffers.push_back(buffer);
      result.input_indices.Set(buffer, opt_indices.value());
    } else {
      throw NonEinsumError(self->mod, block);
    }
  }
  int n_writes = block->writes.size();
  for (int i = 0; i < n_writes; ++i) {
    const Buffer& buffer = block->writes[i]->buffer;
    if (buffer_used.count(buffer.get()) != 0) {
      throw NonEinsumError(self->mod, block);
    }
    buffer_used.insert(buffer.get());
    if (Optional<Array<Var>> opt_indices = CheckTrivialBufferAccess(block->writes[i])) {
      result.output_buffers.push_back(buffer);
      result.output_indices.Set(buffer, opt_indices.value());
    } else {
      throw NonEinsumError(self->mod, block);
    }
  }
  return result;
}

class BufferNotAllocatedInScopeError : public ScheduleError {
 public:
  explicit BufferNotAllocatedInScopeError(IRModule mod, Buffer buffer)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {}

  String FastErrorString() const final {
    return "ScheduleError: The buffer is not allocated as an intermediate buffer in current "
           "PrimFunc.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The buffer " << buffer_->name
       << " is not allocated as an intermediate buffer in current PrimFunc.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  Buffer buffer_;
};

/*! \brief The schedule error class when the producer block cannot be padded. */
class InvalidProducerError : public ScheduleError {
 public:
  explicit InvalidProducerError(IRModule mod, Block producer)
      : mod_(std::move(mod)), producer_(std::move(producer)) {}

  String FastErrorString() const final {
    return "ScheduleError: The producer block cannot be padded.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The producer block {0} cannot be padded. It should write to a single buffer and the "
          "body should be a BufferStore.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {producer_}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  Block producer_;
};

void PadEinsum(ScheduleState self, StmtSRef block_sref, Array<Integer> padding) {
  arith::Analyzer analyzer;
  // Step 1: Input checking and error handling
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  BlockRealize realize = GetBlockRealize(self, block_sref);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  InvalidPaddingError::Check(self, GetRef<Block>(block), padding);

  // Step 2. Extract the Einsum pattern
  Einsum einsum = ExtractEinsum(self, GetRef<Block>(block));

  // Step 3. Figure out the padding needed
  std::unordered_set<const VarNode*> padded_iters;
  Map<Var, PrimExpr> padded_iter_extents;
  for (int i = 0, n = padding.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimExpr dom = iter->dom->extent;
    PrimExpr new_dom = analyzer.Simplify(ceildiv(dom, padding[i]) * padding[i]);
    if (analyzer.CanProveEqual(new_dom, dom)) {
      padded_iter_extents.Set(iter->var, dom);
      padded_iter_extents.Set(Downcast<Var>(realize->iter_values[i]), dom);
    } else {
      padded_iter_extents.Set(iter->var, new_dom);
      padded_iter_extents.Set(Downcast<Var>(realize->iter_values[i]), new_dom);
      padded_iters.insert(iter->var.get());
    }
  }
  auto f_needs_padding = [&padded_iters](const Array<Range>& region) {
    for (const Range& range : region) {
      Var iter = Downcast<Var>(range->min);
      if (padded_iters.count(iter.get())) {
        return true;
      }
    }
    return false;
  };

  // Step 4. For each buffer, if it needs padding, create a new buffer and a new block
  for (const BufferRegion& buffer_region : block->reads) {
    if (f_needs_padding(buffer_region->region)) {
      BufferPadding bp = BufferPadding::FromBufferRegion(buffer_region, padded_iter_extents);
      bp.MakeCopyBlock(true, &analyzer);
    }
  }
  for (const BufferRegion& buffer_region : block->writes) {
    if (f_needs_padding(buffer_region->region)) {
      BufferPadding bp = BufferPadding::FromBufferRegion(buffer_region, padded_iter_extents);
      bp.MakeCopyBlock(false, &analyzer);
    }
  }
}

/******** Instruction Registration ********/

struct PadEinsumTraits : public UnpackedInstTraits<PadEinsumTraits> {
  static constexpr const char* kName = "PadEinsum";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Array<Integer> padding) {
    sch->PadEinsum(block, padding);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<Integer> padding) {
    PythonAPICall py("pad_einsum");
    py.Input("block", block);
    py.Input("padding", padding);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PadEinsumTraits);

}  // namespace tir
}  // namespace tvm
