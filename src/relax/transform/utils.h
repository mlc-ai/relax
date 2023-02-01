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
 * \file src/relax/transform/utils.h
 * \brief Additional utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_TRANSFORM_UTILS_H_
#define TVM_RELAX_TRANSFORM_UTILS_H_

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <unordered_map>

#include "../../relay/analysis/graph_partitioner.h"

namespace tvm {
namespace relax {

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ::tvm::relax::ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ::tvm::relax::ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

  virtual OutputType VisitExpr_(const VarNode* vn) {
    ICHECK(memo_.count(GetRef<Expr>(vn)));
    return memo_[GetRef<Expr>(vn)];
  }

  virtual OutputType VisitBinding_(const VarBindingNode* binding) {
    ICHECK_EQ(memo_.count(binding->var), 0);
    auto v = VisitExpr(binding->value);
    memo_[binding->var] = v;
    return v;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ObjectPtrHash, ObjectPtrEqual> memo_;
};

/*!
 * \brief Remove unused global relax functions in an IRModule.
 * \param mod The target module
 * \param entry_functions list of entry functions
 * \return The updated module.
 */
TVM_DLL IRModule RemoveUnusedFunctions(IRModule mod, Array<runtime::String> entry_funcs);

/*!
 * \brief Get the external symbol of the Relax function name.
 *
 * \param func The provided function.
 * \return An external symbol.
 */
inline std::string GetExtSymbol(const Function& func) {
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(name_node.defined()) << "Fail to retrieve external symbol.";
  return std::string(name_node.value());
}

/*!
 * \brief Fuse ops or functions according to the given partition, and grouped them into a new
 * function.
 *
 * \param mod The input module.
 * \param partition A mapping from a subexpression to the containing group.
 * \param create_single_binding_function Whether or not to create a grouped function for a group
 *  containing a single binding.
 * \return A new module containing grouped functions.
 */
IRModule MakeGroupedFunctions(
    IRModule mod,
    const std::unordered_map<const Object*, relay::GraphPartitioner::Group*>& partition,
    bool create_single_binding_function = false);

/*!
 * \brief Check if the given StructInfo is a scalar tensor. The sinfo should be an instance of
 * TensorStructInfo; its shape must be ShapeExpr.
 * \param sinfo The StructInfo to be checked.
 * \return true if the given StructInfo is a scalar tensor.
 */
bool IsScalarTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a scalar tensor. Now the shape of the tensor expr must be
 * ShapeExpr.
 * \param expr The expr to be checked.
 * \return true if the given expr is a scalar tensor.
 */
bool IsScalarTensor(const Expr& expr);

/*!
 * \brief Check if the given StructInfo is a nested tensor StructInfo satisfying the given
 * condition f_condition.
 * \param sinfo The StructInfo to be checked.
 * \param f_condition The condition function for each leaf StructInfo with signature
 * `bool f_condition(TensorStructInfo)`.
 * \tparam FType The condition function type.
 * \return true if the given StructInfo is a nested tensor satisfying the given f_condition.
 */
template <typename FType>
bool IsNestedTensorConditioned(const StructInfo& sinfo, FType f_condition) {
  if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
    return f_condition(GetRef<TensorStructInfo>(tensor_sinfo));
  } else if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    return !std::any_of(
        tuple_sinfo->fields.begin(), tuple_sinfo->fields.end(),
        [&](const StructInfo& field) { return !IsNestedTensorConditioned(field, f_condition); });
  }
  return false;
}

/*!
 * \brief Check if the given StructInfo is a nested tensor.
 * \param sinfo The StructInfo to be checked.
 * \return true if the given StructInfo is a nested tensor.
 */
bool IsNestedTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a nested tensor.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor.
 */
bool IsNestedTensor(const Expr& expr);

/*!
 * \brief Check if the given StructInfo is a nested tensor of floating point dtype.
 * \param sinfo The StructInfo to be checked.
 * \return true if the given StructInfo is a nested tensor of floating point dtype.
 */
bool IsNestedFloatTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a nested tensor of floating point dtype.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor of floating point dtype.
 */
bool IsNestedFloatTensor(const Expr& expr);

// TODO(@bohan): implements some postorder function accepts a visitor closure
class VarReplacer : public ExprMutator {
 public:
  using VarMap = std::unordered_map<Id, Var, ObjectPtrHash, ObjectPtrEqual>;

  explicit VarReplacer(const VarMap& var_remap) : var_remap_(var_remap) {}

  static Expr Replace(const Expr& expr, const VarMap& var_remap) {
    VarReplacer replacer(var_remap);
    return replacer(expr);
  }

 private:
  Expr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var->vid);
    return it == var_remap_.end() ? var : it->second;
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var->vid);
    return it == var_remap_.end() ? var : it->second;
  }

  const VarMap& var_remap_;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
