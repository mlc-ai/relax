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
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/ir/module.h>

namespace tvm {

namespace tir{


PrimExpr ReLU(PrimExpr x) {
  return max(x, 0);
}

PrimExpr BiasAdd(PrimExpr x, PrimExpr bias) {
  return x + bias;
}


struct FuncMatcher : public StmtExprVisitor {

  void VisitStmt_(const BlockNode* op) final {
    if(fail_){
      return;
    }
    Array<BufferRegion> reads = op->reads;
    Array<BufferRegion> writes = op->writes;
    Array<BufferLoad> single_loads;
    for (const auto& read : op->reads) {
      Array<PrimExpr> indices;
      for (const auto& range : read->region) {
        ICHECK(is_one(range->extent));
        indices.push_back(range->min);
      }
      single_loads.push_back(BufferLoad(read->buffer, indices));
    }
    ICHECK(op->body.as<BufferStoreNode>());
    PrimExpr rhs = op->body.as<BufferStoreNode>()->value;
    // todo check if is cutlass
    if( is_matching_){
        if(const EvaluateNode* eval = op->body.as<EvaluateNode>()){
            if(const CallNode* call = eval->value.as<CallNode>()){
                
            }
        }
    }
    if (is_matching_ && single_loads.size() == 2) {
      PrimExpr composite = ReLU(BiasAdd(single_loads[0], single_loads[1]));
      if (!StructuralEqual()(composite, rhs)) {
        if(block_partition_.empty()){
          fail_ = true;
          return;
        }
        is_matching_ = false;
        allocs1_.erase(intermediate_buffer);
      }
    }
    StmtExprVisitor::VisitStmt_(op);

    }

    void VisitStmt_(const BufferStoreNode* op) final{
        if(is_matching_){
            allocs1_.insert(op->buffer);
        } else if (allocs1_.count(op->buffer)){
            fail_ = true;
        } else{
            allocs2_.insert(op->buffer);
        }
        if(is_matching_){
            intermediate_buffer = op->buffer;
        }else{
            input2_.insert(op->buffer);
        }
    }

    void VisitExpr_(const BufferLoadNode* op) final{
        if(is_matching_){
            input1_.insert(op->buffer);
        } else {
            input2_.insert(op->buffer);
        }

    }


    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs1_;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs2_;
    Map<Block, Bool> block_partition_;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input1_;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input2_;
    Buffer intermediate_buffer;
    bool is_matching_ = true;
    bool fail_ = false;
};

class BlockMasker: public StmtExprMutator {

    public:

    static Stmt Mask(Stmt stmt, const Map<Block, Bool>& block_partition, const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs, bool is_cutlass){
        BlockMasker masker(block_partition, allocs, is_cutlass);
        return masker(stmt);
    }

    private:
    BlockMasker(const Map<Block, Bool>& block_partition, const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs,  bool is_cutlass) : block_partition_(block_partition), allocs_(allocs), is_cutlass_(is_cutlass) {}

    Stmt VisitStmt_(const BlockNode* op) final{
        Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
        if(op->name_hint=="root"){
            return block;
        }
        ICHECK(block_partition_.count(GetRef<Block>(op)));
        bool block_is_cutlass = block_partition_[GetRef<Block>(op)]->value;
        Stmt body;
        if (!(is_cutlass_ ^ block_is_cutlass)) {
          body = block->body;
        } else {
          body = Evaluate(0);
        }
        Array<Buffer> alloc_buffers;
        for(const Buffer& b: block->alloc_buffers){
            if(allocs_.count(b)){
                alloc_buffers.push_back(b);
            }
        }
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*block.operator->());
        n->body = body;
        n->alloc_buffers = alloc_buffers;
        return Block(n);
    }
    Map<Block, Bool> block_partition_;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs_;
    bool is_cutlass_ = false;
};

std::pair<PrimFunc, Optional<PrimFunc>> SplitFunctions(PrimFunc func, std::vector<std::vector<int>>* param_partition){
    FuncMatcher matcher;
    matcher(func->body);
    if(matcher.fail_){
        return {func, NullOpt};
    }
    
    Stmt body1 = BlockMasker::Mask(func->body, matcher.block_partition_, matcher.allocs1_, true);
    Stmt body2 = BlockMasker::Mask(func->body, matcher.block_partition_, matcher.allocs2_, false);
    Array<Var> new_params1;
    std::vector<int> param_partition1;
    for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
        Var param = func->params[i];
      if (matcher.input1_.count(func->buffer_map[param])) {
        new_params1.push_back(param);
        if(i!=static_cast<int>(func->params.size())-1){
            param_partition1.push_back(i);
        }
      }
    }
    param_partition->push_back(param_partition1);

    new_params1.push_back(Var("output", DataType::Handle()));
    Map<Var, Buffer> new_buffer_map1;
    for (const auto& kv : func->buffer_map) {
      if (matcher.input1_.count(kv.second)) {
        new_buffer_map1.Set(kv.first, kv.second);
      }
    }
    new_buffer_map1.Set(new_params1.back(), matcher.intermediate_buffer);
    PrimFunc func1 = PrimFunc(new_params1, body1, func->ret_type, new_buffer_map1, NullOpt, func->attrs);


    //deal with second function
    Array<Var> new_params2;
    std::vector<int> param_partition2;
    new_params2.push_back(Var("input", DataType::Handle()));
    for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
        Var param = func->params[i];
      if (matcher.input2_.count(func->buffer_map[param])) {
        new_params2.push_back(param);
        if(i!=static_cast<int>(func->params.size())-1){
            param_partition2.push_back(i);
        }
      }
    }
    param_partition->push_back(param_partition2);
    Map<Var, Buffer> new_buffer_map2;
    new_buffer_map2.Set(new_params2[0], matcher.intermediate_buffer);
    for (const auto& kv : func->buffer_map) {
      if (matcher.input2_.count(kv.second)) {
        new_buffer_map2.Set(kv.first, kv.second);
      }
    }
    PrimFunc func2 = PrimFunc(new_params2, body2, func->ret_type, new_buffer_map2, NullOpt, func->attrs);
    return {func1, func2};
}
}  // namespace tir

namespace relax {

class SplitMutator: public ExprMutator{
    public:
    SplitMutator(const tvm::IRModule& mod): mod_(mod) {}
    static IRModule Transform(const IRModule& mod){
        SplitMutator mutator(mod);
        for (auto& kv : mod->functions) {
            if (auto* func = kv.second.as<FunctionNode>()) {
                Function new_func = Downcast<Function>(mutator(GetRef<Function>(func)));
                mutator.builder_->AddFunction(new_func, kv.first->name_hint);
            }
        }
        return mutator.builder_->GetContextIRModule();
    }
    private:
    inline Array<Expr> GetCallTIRArgs(Expr args){
        if(args.as<TupleNode>()){
            return args.as<TupleNode>()->fields;
        } else {
            return {args};
        }
    }

    Expr VisitExpr_(const CallNode* op) final {
        Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));
        static const Op& call_tir_op_ = Op::Get("relax.call_tir");
        if(call->op.same_as(call_tir_op_)){
            GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
            tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
            std::vector<std::vector<int>> param_partition;
            std::pair<tir::PrimFunc, Optional<tir::PrimFunc>> split_funcs = tir::SplitFunctions(func, &param_partition);
            if (!split_funcs.second.defined()) {
                return call;
            }
            tir::PrimFunc func1 = split_funcs.first;
            tir::PrimFunc func2 = split_funcs.second.value();
            ICHECK(param_partition.size() == 2);
            Array<Expr> args1;
            for (int p: param_partition[0]) {
                args1.push_back(GetCallTIRArgs(call->args[1])[p + 1]);
            }
            ShapeExpr shape1 (func1->buffer_map[func1->params.back()]->shape);
            GlobalVar gv1 = builder_->AddFunction(func1, "cutlass");
            Call call1(call_tir_op_, {gv1, Tuple(args1), shape1}, call->attrs, call->type_args);
            Var call_var1 = builder_->Emit(call1);
            Array<Expr> args2;
            args2.push_back(call_var1);
            for (int p : param_partition[1]) {
              args2.push_back(GetCallTIRArgs(call->args[1])[p + 1]);
            }
            GlobalVar gv2 = builder_->AddFunction(func2, "unfused_epilogue");
            Call call2(call_tir_op_, {gv2, Tuple(args2), call->args[2]}, call->attrs, call->type_args);
            return call2;
        }
    }

    tvm::IRModule mod_;
};

namespace transform {
  Pass SplitCutlass() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitMutator::Transform(m); };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"SplitCutlass",      //
                          /*required=*/{});
}
  TVM_REGISTER_GLOBAL("relax.transform.SplitCutlass").set_body_typed(SplitCutlass);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
