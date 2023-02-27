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
#include <tvm/ir/expr.h>
#include <tvm/relax/distributed/global_info.h>
#include "./utils.h"
#include "../relax/utils.h"
namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<runtime::ShapeTuple>(
        "", [](runtime::ShapeTuple n, ObjectPath n_p, IRDocsifier d) -> Doc {
          int s = n.size();
          Array<ExprDoc> results;
          results.reserve(s);
          for (int i = 0; i < s; ++i) {
            results.push_back(d->AsDoc<ExprDoc>(Integer(n[i]), n_p->ArrayIndex(i)));
          }
          return TupleDoc(results);
        });


TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::distributed::DeviceMesh>(
        "", [](relax::distributed::DeviceMesh n, ObjectPath n_p, IRDocsifier d) -> Doc {

            bool has_relax_frame = false;
            const IRFrameNode* f = nullptr;
            for (const Frame& frame : d->frames) {
                if (const auto* relax_frame = frame.as<RelaxFrameNode>()) {
                    has_relax_frame = true;
                    break;
                } else if(const auto* ir_frame = frame.as<IRFrameNode>()){
                    f = ir_frame;
                }
            }
            if (!has_relax_frame || !f) {
                Array<ExprDoc> args;
                args.push_back(d->AsDoc<ExprDoc>(n->shape, n_p->Attr("shape")));
                if(n->device_range.defined()){
                    args.push_back(d->AsDoc<ExprDoc>(n->device_range, n_p->Attr("device_range")));
                } else{
                    args.push_back(d->AsDoc<ExprDoc>(n->device_ids, n_p->Attr("device_ids")));
                }
                return IR(d, "device_mesh")->Call(args);
            } else {
                for(const auto& kv: *f->global_infos){
                    for (int i = 0;i<static_cast<int>(kv.second.size());i++) {
                      if (kv.second[i].same_as(n)) {
                        std::stringstream ss;
                        ss << kv.first << "[" << i << "]";
                        return d->AsDoc<Doc>(String(ss.str()), n_p);
                      }
                    }
                }
                LOG(FATAL) << "Cannot find device mesh in global infos";
            }

        });

TVM_SCRIPT_REPR(relax::distributed::DeviceMeshNode, ReprPrintIR);

}  // namespace printer
}  // namespace script
}  // namespace tvm