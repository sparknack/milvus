// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <typeindex>
#include "index/ScalarIndex.h"
#include "tantivy-binding.h"

namespace milvus::index {
using JsonValueType = ::JsonExistValueType;

// Query capability shared by independent JSON-flat implementations.
class JsonFlatExecutorBase {
 public:
    virtual ~JsonFlatExecutorBase() = default;
    virtual TargetBitmap
    ExactPathExists(JsonValueType type = JsonValueType::Any) = 0;
    virtual void
    SetArrayQuery(bool) {
    }
};

class JsonFlatIndexBase {
 public:
    virtual ~JsonFlatIndexBase() = default;
    virtual std::string
    GetNestedPath() const = 0;
    virtual std::shared_ptr<IndexBase>
    CreateExecutor(std::string relative_path,
                   std::type_index type,
                   bool comparable) const = 0;

    template <typename T>
    std::shared_ptr<ScalarIndex<T>>
    create_executor(std::string path, bool comparable = true) const {
        auto executor = std::dynamic_pointer_cast<ScalarIndex<T>>(
            CreateExecutor(std::move(path), typeid(T), comparable));
        AssertInfo(executor != nullptr, "JSON flat executor type mismatch");
        return executor;
    }
};
}  // namespace milvus::index
