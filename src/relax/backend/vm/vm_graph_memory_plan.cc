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
 * \file src/relax/backend/vm/vm_graph_memory_plan.cc
 * \brief Perform memory planning for memory reuse.
 */
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include <map>
#include <unordered_map>
#include <vector>

#include "../../../relay/transforms/pattern_utils.h"
#include "../../../support/arena.h"
#include "../../op/make_op.h"

namespace tvm {
namespace relax {

struct StorageToken {
  /*! \brief Reference counter */
  int ref_counter{0};
  /*! \brief number of bytes */
  int bytes{-1};
  /*! \brief The shape of the tensor */
  Array<PrimExpr> shape;
  /*! \brief The corresponding tensor dtype. */
  DataType dtype;
  /*! \brief The storage id */
  int storage_id{-1};
  /*! \brief The corresponding allocated storage */
  Var storage{nullptr};

  // Todo:
  // - take care of virtual device id
  // - think about symbolic dynamic case
  // - handle reshape

  std::string ToString() const {
    std::ostringstream os;
    os << "{storage_id: " << storage_id << ", bytes: " << bytes << ", shape: " << shape
       << ", dtype: " << dtype << "}";
    return os.str();
  }
};

// Todo: comments
// - only one can be non-empty, depending on `is_tuple`
struct TokenWrapper {
  TokenWrapper() : is_tuple(false), field_tokens{}, token(nullptr) {}

  TokenWrapper(bool is_tuple, std::vector<TokenWrapper> field_tokens, StorageToken* token) {
    if (is_tuple) {
      ICHECK(token == nullptr);
    } else {
      ICHECK_NOTNULL(token);
      ICHECK(field_tokens.empty());
    }

    this->is_tuple = is_tuple;
    this->field_tokens = std::move(field_tokens);
    this->token = token;
  }

  bool IsEmptyToken() const { return is_tuple == true && field_tokens.empty() && token == nullptr; }

  static std::vector<StorageToken*> FlattenTokens(const TokenWrapper& wrapper) {
    std::vector<StorageToken*> tokens;
    std::function<void(const TokenWrapper&)> f_visit = [&](const TokenWrapper& wrapper) {
      if (wrapper.is_tuple) {
        for (const TokenWrapper& field_wrapper : wrapper.field_tokens) {
          f_visit(field_wrapper);
        }
      } else {
        ICHECK_NOTNULL(wrapper.token);
        tokens.push_back(wrapper.token);
      }
    };

    f_visit(wrapper);
    return tokens;
  }

  void RemoveToken(const StorageToken* token_to_remove) {
    if (is_tuple) {
      for (TokenWrapper& wrapper : field_tokens) {
        wrapper.RemoveToken(token_to_remove);
      }
    } else if (token == token_to_remove) {
      // Set to an empty token.
      is_tuple = true;
      ICHECK(field_tokens.empty());
      token = nullptr;
    }
  }

  bool is_tuple;
  std::vector<TokenWrapper> field_tokens;
  StorageToken* token;
};

// - Nested functions are not considered yet.
// - Symbolic shape cases (ones with MatchShape) are completely not considered yet.
// - RuntimeDepShape is not allowed at this moment.
// - Reshape is not considered yet.
class StorageAllocatorInit : public ExprVisitor {
 public:
  // Entry function
  std::unordered_map<const ExprNode*, TokenWrapper> Init(const Function& func) {
    const TokenWrapper& body_tokens = GetTokens(func->body);
    // Mark the function body tokens as external referenced.
    ClearTokens(TokenWrapper::FlattenTokens(body_tokens));

    return this->token_map_;
  }

 private:
  void VisitBindingBlock_(const BindingBlockNode* block) final {
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    ICHECK(!block_stack_.empty());
    ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    const TokenWrapper& tokens = GetTokens(binding->value);
    ExprUsesTokens(binding->var.get(), tokens);
  }

  void VisitExpr_(const VarNode* var) final { ICHECK(token_map_.count(var)); }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      this->CreateToken(call);
      return;
    }

    ICHECK(!block_stack_.empty());
    const BindingBlockNode* cur_binding_block = block_stack_.back();
    for (const Expr& arg : call->args) {
      const TokenWrapper& wrapper = GetTokensWithAllocSiteCheck(arg, cur_binding_block);
      IncreaseRefCounter(wrapper);
    }

    // Todo: handle reshape separately
  }

  void VisitExpr_(const TupleNode* tuple) final {
    std::vector<TokenWrapper> tokens;
    for (const Expr& field : tuple->fields) {
      const TokenWrapper& field_tokens = GetTokens(field);
      tokens.push_back(field_tokens);
    }
    ExprUsesTokens(tuple, TokenWrapper(/*is_tuple=*/true,                   //
                                       /*field_tokens=*/std::move(tokens),  //
                                       /*token=*/nullptr));
  }

  void VisitExpr_(const TupleGetItemNode* tuple_item) final {
    ICHECK_GE(tuple_item->index, 0);

    const TokenWrapper& wrapper = GetTokens(tuple_item->tuple);
    ICHECK(wrapper.is_tuple);
    if (static_cast<int>(wrapper.field_tokens.size()) > tuple_item->index) {
      ExprUsesTokens(tuple_item, wrapper.field_tokens[tuple_item->index]);
    } else {
      ICHECK(wrapper.IsEmptyToken());
      token_map_[tuple_item] = no_tokens_;
    }
  }

  void VisitExpr_(const IfNode* if_node) final {
    const TokenWrapper& cond_tokens = GetTokens(if_node->cond);
    const TokenWrapper& then_tokens = GetTokens(if_node->true_branch);
    const TokenWrapper& else_tokens = GetTokens(if_node->false_branch);
    ClearTokens(TokenWrapper::FlattenTokens(cond_tokens));
    ClearTokens(TokenWrapper::FlattenTokens(then_tokens));
    ClearTokens(TokenWrapper::FlattenTokens(else_tokens));
  }

  void VisitExpr_(const SeqExprNode* seq) final {
    for (const BindingBlock& binding_block : seq->blocks) {
      this->VisitBindingBlock(binding_block);
    }
    const TokenWrapper& body_tokens = GetTokens(seq->body);
    ClearTokens(TokenWrapper::FlattenTokens(body_tokens));
  }

  TokenWrapper CreateToken(const CallNode* call) {
    ICHECK(!token_map_.count(call));

    // The impl guarantees that the input can only have DynTensorType
    const auto* ttype = call->checked_type().as<DynTensorTypeNode>();
    const auto* shape = call->shape().as<ShapeExprNode>();
    const auto* attrs = call->attrs.as<AllocTensorAttrs>();
    ICHECK_NOTNULL(ttype);
    ICHECK_NOTNULL(attrs);

    ICHECK(call->shape().same_as(call->args[0]));
    ICHECK(!ttype->IsUnknownDtype());
    ICHECK(ttype->dtype == attrs->dtype);

    if (ttype->IsUnknownNdim() || shape == nullptr) {
      token_map_[call] = no_tokens_;
      return no_tokens_;
    }
    // Does not support planning for symbolic shape at this moment.
    for (const PrimExpr& dim_len : shape->values) {
      if (!tir::is_const_int(dim_len)) {
        token_map_[call] = no_tokens_;
        return no_tokens_;
      }
    }

    auto* token = arena_->make<StorageToken>();
    token->dtype = ttype->dtype;
    token->shape = shape->values;
    ICHECK(!block_stack_.empty());
    token2block_[token] = block_stack_.back();

    TokenWrapper token_wrapper(/*is_tuple=*/false, /*field_tokens=*/{}, token);
    ExprUsesTokens(call, token_wrapper);
    // token_map_[call] = token_wrapper;
    return token_wrapper;
  }

  // The function is the place where recursive visit happens.
  const TokenWrapper& GetTokens(const Expr& expr) {
    this->VisitExpr(expr);
    auto it = token_map_.find(expr.get());
    if (it == token_map_.end()) {
      LOG(INFO) << "Tokens not created for " << expr->GetTypeKey() << ":\n" << PrettyPrint(expr);
      token_map_[expr.get()] = no_tokens_;
      return no_tokens_;
    }
    return it->second;
  }

  void ExprUsesTokens(const ExprNode* expr, const TokenWrapper& wrapper) {
    auto it = token_map_.insert({expr, wrapper});
    ICHECK(it.second == true);

    const std::vector<StorageToken*> tokens = TokenWrapper::FlattenTokens(wrapper);
    for (const StorageToken* token : tokens) {
      token2exprs_[token].push_back(expr);
    }
  }

  const TokenWrapper& GetTokensWithAllocSiteCheck(const Expr& expr,
                                                  const BindingBlockNode* cur_block) {
    const TokenWrapper& wrapper = GetTokens(expr);
    const std::vector<StorageToken*> tokens = TokenWrapper::FlattenTokens(wrapper);
    for (StorageToken* token : tokens) {
      auto it = token2block_.find(token);
      ICHECK(it != token2block_.end());
      if (it->second != cur_block) {
        ClearTokens({token});
      }
    }
    return token_map_[expr.get()];
  }

  void IncreaseRefCounter(const TokenWrapper& token_wrapper) {
    const std::vector<StorageToken*>& tokens = TokenWrapper::FlattenTokens(token_wrapper);
    for (StorageToken* token : tokens) {
      token->ref_counter += 1;
    }
  }

  void ClearTokens(const std::vector<StorageToken*>& tokens) {
    for (const StorageToken* token : tokens) {
      const std::vector<const ExprNode*>& exprs = token2exprs_[token];
      for (const ExprNode* expr : exprs) {
        token_map_[expr].RemoveToken(token);
      }
      token2exprs_.erase(token);
      token2block_.erase(token);
    }
  }

  /*! \brief The allocator */
  support::Arena* arena_;
  /*! \brief The binding block stack */
  std::vector<const BindingBlockNode*> block_stack_;
  /*! \brief The mapping from each Expr to its corresponding storage tokens */
  std::unordered_map<const ExprNode*, TokenWrapper> token_map_;
  /*! \brief The mapping from each token to the binding block where it is created */
  std::unordered_map<const StorageToken*, const BindingBlockNode*> token2block_;
  /*! \brief The mapping from each token to the Exprs that share this token */
  std::unordered_map<const StorageToken*, std::vector<const ExprNode*>> token2exprs_;
  /*! \brief An empty token map */
  const TokenWrapper no_tokens_;
};

/**
 * \brief Memory manager for flattened 1d memory (buffers)
 * TODO: refine
 */
class TokenAllocator1D {
 public:
  /*!
   * \brief Request a storage token for a given prototype.
   * \param prototype. The prototype storage token.
   * \return The result token.
   */
  StorageToken* Request(StorageToken* prototype) {
    ICHECK_EQ(prototype->storage_id, -1);

    // calculate the size;
    int size = GetMemorySize(prototype);
    // search memory block in [size / match_range_, size * match_range_)
    auto begin = available_pool_.lower_bound(size / match_range_);
    auto mid = available_pool_.lower_bound(size);
    auto end = available_pool_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageToken* available_token = it->second;
      ICHECK_EQ(available_token->ref_counter, 0);
      // Use exact matching strategy
      ICHECK_LE(size, available_token->bytes);
      available_token->ref_counter = prototype->ref_counter;
      // find a exact match, erase from map and return
      available_pool_.erase(it);
      return available_token;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageToken* available_token = it->second;
      ICHECK_EQ(available_token->ref_counter, 0);
      // Use exact matching strategy
      ICHECK_GE(size, available_token->bytes);
      available_token->bytes = size;
      available_token->ref_counter = prototype->ref_counter;
      // erase from map and return
      available_pool_.erase(it);
      return available_token;
    }
    return nullptr;
  }

  /*!
   * \brief Allocate a storage token by consuming prototype
   * \param prototype The prototype token.
   * \param size The size of memory being requested.
   */
  StorageToken* Alloc(StorageToken* prototype, int storage_id) {
    ICHECK_EQ(prototype->storage_id, -1);
    ICHECK(!prototype->storage.defined());
    int size = GetMemorySize(prototype);
    prototype->bytes = size;
    prototype->storage_id = storage_id;
    full_pool_.push_back(prototype);
    return prototype;
  }

  /*!
   * \brief Check if we can release token.
   * \param token The token to be released.
   */
  void Release(StorageToken* token) {
    ICHECK_GE(token->storage_id, 0);
    ICHECK_GE(token->bytes, 0);
    ICHECK_EQ(token->ref_counter, 0);
    ICHECK(!token->storage.defined());
    available_pool_.insert({token->bytes, token});
  }

 private:
  /*!
   * \brief ceil(size/word_size) to get number of words.
   * \param size The original size.
   * \param word_size The element size.
   */
  static int DivRoundUp(int size, int word_size) { return (size + word_size - 1) / word_size; }

  /*!
   * Todo: refine
   * \brief Get the memory requirement.
   * \param prototype The prototype token.
   * \return The required memory size.
   */
  int GetMemorySize(StorageToken* prototype) {
    ICHECK_EQ(prototype->storage_id, -1);
    if (prototype->bytes != -1) {
      return prototype->bytes;
    }

    int size = 1;
    for (const PrimExpr& dim_len : prototype->shape) {
      const int64_t* p_dim_len = tir::as_const_int(dim_len);
      ICHECK_NOTNULL(p_dim_len);
      size *= *p_dim_len;
    }
    size *= DivRoundUp(prototype->dtype.bits() * prototype->dtype.lanes(), 8);
    prototype->bytes = size;
    return size;
  }

 private:
  // scale used for rough match
  const int match_range_{16};
  // free list of storage entry
  std::multimap<int, StorageToken*> available_pool_;
  // all the storage resources available
  std::vector<StorageToken*> full_pool_;
  /*! \brief Number of storages */
  int n_storage_;
};

class StorageAllocator : public ExprVisitor {
 public:
  explicit StorageAllocator(std::unordered_map<const ExprNode*, TokenWrapper> token_map)
      : token_map_(std::move(token_map)) {}

  std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token;
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors;
  std::unordered_map<const BindingBlockNode*, std::vector<const StorageToken*>> block2tokens;

 private:
  void VisitBindingBlock_(const BindingBlockNode* block) final {
    block_stack_.push_back(block);
    ExprVisitor::VisitBindingBlock_(block);
    ICHECK(!block_stack_.empty());
    ICHECK(block_stack_.back() == block);
    block_stack_.pop_back();

    for (const StorageToken* token : block2tokens[block]) {
      ICHECK_EQ(token->ref_counter, 0);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    const TokenWrapper& tokens = GetTokens(binding->value);
    token_map_[binding->var.get()] = tokens;

    if (const CallNode* call_alloc_tensor = binding->value.as<CallNode>()) {
      auto it = alloc_tensor2token.find(call_alloc_tensor);
      if (it != alloc_tensor2token.end()) {
        auto it_insert = token2cur_tensor_.insert({it->second, binding->var});
        ICHECK(it_insert.second == true);
      }
    }
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call->op == alloc_tensor_op) {
      auto it = token_map_.find(call);
      ICHECK(it != token_map_.end());
      if (it->second.IsEmptyToken()) {
        token_map_[call] = no_tokens_;
        return;
      }

      const auto* attrs = call->attrs.as<AllocTensorAttrs>();
      ICHECK(!it->second.is_tuple);
      ICHECK(it->second.token != nullptr);
      StorageToken* new_token = this->RequestOrAlloc(it->second.token, attrs->runtime_device_index);
      // It doesn't make sense if a newly allocated tensor has 0 reference.
      ICHECK_GT(new_token->ref_counter, 0);

      alloc_tensor2token[call] = new_token;
      token_map_[call] = TokenWrapper(/*is_tuple=*/false, /*field_tokens=*/{}, /*token=*/new_token);
      ICHECK(!block_stack_.empty());
      block2tokens[block_stack_.back()].push_back(new_token);
      return;
    }

    for (const Expr& arg : call->args) {
      const TokenWrapper& wrapper = GetTokens(arg);
      const std::vector<StorageToken*> tokens = TokenWrapper::FlattenTokens(wrapper);
      for (StorageToken* token : tokens) {
        ICHECK_GT(token->ref_counter, 0);
        token->ref_counter -= 1;
        this->CheckForRelease(token, call);
      }
    }
  }

  void VisitExpr_(const TupleNode* tuple) final {
    std::vector<TokenWrapper> tokens;
    for (const Expr& field : tuple->fields) {
      const TokenWrapper& field_tokens = GetTokens(field);
      tokens.push_back(field_tokens);
    }
    token_map_[tuple] = TokenWrapper(/*is_tuple=*/true,                   //
                                     /*field_tokens=*/std::move(tokens),  //
                                     /*token=*/nullptr);
  }

  void VisitExpr_(const TupleGetItemNode* tuple_item) final {
    ICHECK_GE(tuple_item->index, 0);

    const TokenWrapper& wrapper = GetTokens(tuple_item->tuple);
    ICHECK(wrapper.is_tuple);
    if (static_cast<int>(wrapper.field_tokens.size()) > tuple_item->index) {
      token_map_[tuple_item] = wrapper.field_tokens[tuple_item->index];
    } else {
      ICHECK(wrapper.IsEmptyToken());
      token_map_[tuple_item] = no_tokens_;
    }
  }

  const TokenWrapper& GetTokens(const Expr& expr) {
    auto it = token_map_.find(expr.get());
    ICHECK(it != token_map_.end());
    return it->second;
  }

  StorageToken* RequestOrAlloc(StorageToken* prototype, int64_t virtual_device_idx) {
    StorageToken* token = allocator_.Request(prototype);
    if (token == nullptr) {
      token = allocator_.Alloc(prototype, this->n_storage_++);
    }
    ICHECK_NOTNULL(token);
    return token;
  }

  void CheckForRelease(StorageToken* token, const CallNode* release_site) {
    ICHECK_GE(token->storage_id, 0);
    ICHECK_GE(token->bytes, 0);
    ICHECK_GE(token->ref_counter, 0);
    ICHECK(!token->storage.defined());
    if (token->ref_counter == 0) {
      allocator_.Release(token);

      auto it = token2cur_tensor_.find(token);
      ICHECK(it != token2cur_tensor_.end());
      expr2killed_tensors[release_site].push_back(it->second);
      token2cur_tensor_.erase(it);
    }
  }

  /*! \brief The binding block stack */
  std::vector<const BindingBlockNode*> block_stack_;

  std::unordered_map<const ExprNode*, TokenWrapper> token_map_;
  /*! \brief An empty token map */
  const TokenWrapper no_tokens_;
  /*! \brief Number of allocated storages */
  int n_storage_{0};
  /*! \brief The 1D memory allocator */
  TokenAllocator1D allocator_;

  std::unordered_map<const StorageToken*, Var> token2cur_tensor_;
};

class StorageAllocationRewriter : public ExprMutator {
 public:
  explicit StorageAllocationRewriter(
      std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token,
      std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors,
      std::unordered_map<const BindingBlockNode*, std::vector<const StorageToken*>> block2tokens)
      : alloc_tensor2token_(std::move(alloc_tensor2token)),
        expr2killed_tensors_(std::move(expr2killed_tensors)),
        block2tokens_(std::move(block2tokens)) {}

 private:
  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }

    for (const StorageToken* token : block2tokens_[block]) {
      ICHECK(token->storage.defined());
      this->builder_->Emit(MakeMemKillStorage(token->storage));
    }

    BindingBlock new_block = builder_->EndBlock();
    return new_block;
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);
    ICHECK(!this->builder_->CurrentBlockIsDataFlow());

    auto it = expr2killed_tensors_.find(new_value.get());

    // fast path: reemit binding if nothing changes
    if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      ICHECK(it == expr2killed_tensors_.end());
      this->builder_->Emit(GetRef<VarBinding>(binding));
      return;
    }

    Var temp = WithShapeAndType(new_var, new_value->shape_, new_value->checked_type_);
    if (!temp.same_as(new_var)) {
      new_var = temp;
      this->var_remap_[binding->var->vid] = new_var;
    }
    this->builder_->Emit(VarBinding(new_var, new_value));

    if (it != expr2killed_tensors_.end()) {
      for (const Var& var : it->second) {
        Var new_var = Downcast<Var>(this->VisitExpr(var));
        this->builder_->Emit(MakeMemKillTensor(new_var));
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto it = alloc_tensor2token_.find(call);
    if (it != alloc_tensor2token_.end()) {
      StorageToken* token = it->second;
      const auto* attrs = call->attrs.as<AllocTensorAttrs>();
      if (!token->storage.defined()) {
        Constant size = relay::MakeConstantScalar(token->dtype, token->bytes);
        Call alloc_storage = Downcast<Call>(
            MakeAllocStorage(std::move(size), attrs->runtime_device_index, "global", token->dtype));
        token->storage = builder_->Emit(alloc_storage);
      }
      return MakeMemAllocTensor(token->storage, call->args[0], /*offset=*/0, attrs->dtype);
    }

    return ExprMutator::VisitExpr_(call);
  }

  std::unordered_map<const CallNode*, StorageToken*> alloc_tensor2token_;
  std::unordered_map<const ExprNode*, std::vector<Var>> expr2killed_tensors_;
  std::unordered_map<const BindingBlockNode*, std::vector<const StorageToken*>> block2tokens_;
};

Expr VMGraphMemoryPlan(Function func) {
  // First scan;
  std::unordered_map<const ExprNode*, TokenWrapper> token_map = StorageAllocatorInit().Init(func);
  // Second scan;
  StorageAllocator allocator(std::move(token_map));
  allocator(func);
  // Rewrite
  StorageAllocationRewriter rewriter(std::move(allocator.alloc_tensor2token),
                                     std::move(allocator.expr2killed_tensors),
                                     std::move(allocator.block2tokens));
  func = Downcast<Function>(rewriter(func));
  return func;
}

namespace transform {

Pass VMGraphMemoryPlan() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(VMGraphMemoryPlan(std::move(f)));
      };
  return CreateFunctionPass(pass_func, 0, "VMGraphMemoryPlan", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMGraphMemoryPlan").set_body_typed(VMGraphMemoryPlan);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
