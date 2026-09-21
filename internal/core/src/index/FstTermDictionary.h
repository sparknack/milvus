// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace milvus::index {
// Immutable, segment-local term -> ordinal dictionary. Serialized bytes contain
// no process pointers. Only the dictionary is persisted here, not the postings.
class FstTermDictionary {
 public:
    static FstTermDictionary
    Build(const std::vector<std::string>& sorted_terms);
    // File must remain immutable for the lifetime of all copies of this object.
    // Verifies the complete checksum before exposing a reader (O(file size)).
    static FstTermDictionary
    MapFile(const std::string& path);
    std::optional<uint32_t>
    Lookup(std::string_view term) const;
    size_t
    Count() const;
    size_t
    ByteSize() const;
    bool
    IsMapped() const;
    std::string_view
    SerializedBytes() const;
    std::vector<std::pair<std::string, uint32_t>>
    Enumerate() const;
    using Visitor = std::function<void(std::string_view, uint32_t)>;
    // Views are valid only for the duration of the callback.
    void
    ForEachPrefix(std::string_view prefix, const Visitor& visitor) const;
    // False means unsupported/over-budget pattern, with no callbacks made.
    // Caller must validate LIKE syntax and use an equivalent fallback.
    bool
    ForEachLike(std::string_view pattern, const Visitor& visitor) const;
    // Whole-term Unicode optimal-string-alignment distance, including adjacent
    // transpositions at cost one. max_edits must be in [0, 2].
    void
    ForEachFuzzy(std::string_view term,
                 uint32_t max_edits,
                 const Visitor& visitor) const;

 private:
    struct State;
    std::shared_ptr<const State> state_;
};
}  // namespace milvus::index
