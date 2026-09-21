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

#include "segcore/TextColumnCache.h"
#include "common/EasyAssert.h"
#include <string_view>
#include <deque>
#include <vector>

namespace milvus::segcore {
// Shared sealed-index input path. Own pending references because source column
// pins may expire between callbacks; emitted text views are callback-local.
// NULL slots preserve row offsets and are never submitted to the LOB reader.
template <typename Enumerate, typename Read, typename Emit, typename Checkpoint>
void
VisitTextLobIndexInput(Enumerate&& enumerate,
                       Read&& read,
                       Emit&& emit,
                       Checkpoint&& checkpoint) {
    struct Entry {
        size_t offset;
        bool valid;
        size_t text_index;
    };
    std::vector<Entry> entries;
    std::deque<std::string> owned_refs;
    std::vector<milvus_storage::lob_column::EncodedRef> refs;
    auto flush = [&] {
        if (entries.empty())
            return;
        auto texts = read(refs);
        AssertInfo(texts.size() == refs.size(),
                   "TEXT LOB index input returned {} texts for {} refs",
                   texts.size(),
                   refs.size());
        for (const auto& entry : entries) {
            emit(entry.valid ? std::string_view(texts[entry.text_index])
                             : std::string_view{},
                 entry.offset,
                 entry.valid);
        }
        entries.clear();
        refs.clear();
        owned_refs.clear();
        checkpoint();
    };
    enumerate([&](std::string_view value, size_t offset, bool valid) {
        entries.push_back({offset, valid, refs.size()});
        if (valid) {
            owned_refs.emplace_back(value);
            const auto& ref = owned_refs.back();
            refs.push_back(MakeTextLobEncodedRef(ref.data(), ref.size()));
        }
        if (entries.size() >= kTextLobIndexBuildBatchSize)
            flush();
    });
    flush();
}
}  // namespace milvus::segcore
