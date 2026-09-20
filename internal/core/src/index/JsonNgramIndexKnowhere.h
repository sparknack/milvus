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
#include "index/NgramIndexKnowhere.h"
#include "index/JsonIndexBuilder.h"

namespace milvus::index {
// One explicitly selected JSON string path, not a flattened whole-JSON index.
class JsonNgramIndexKnowhere : public NgramIndexKnowhere {
 public:
    JsonNgramIndexKnowhere(size_t min_gram,
                           size_t max_gram,
                           std::string path,
                           proto::schema::FieldSchema schema)
        : NgramIndexKnowhere(min_gram, max_gram),
          max_gram_(max_gram),
          path_(std::move(path)),
          schema_(std::move(schema)) {
    }

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        size_t rows = 0;
        for (const auto& field : fields) rows += field->get_num_rows();
        std::vector<std::string> values(rows);
        std::vector<size_t> invalid;
        // A string predicate on a missing/null/non-string path is UNKNOWN.
        // Preserve the raw executor's validity so NOT cannot turn it into a hit.
        ProcessJsonFieldData<std::string>(
            fields,
            schema_,
            path_,
            JsonCastType::FromString("VARCHAR"),
            JsonCastFunction::FromString("unknown"),
            [&](const std::string* value, int64_t n, int64_t offset) {
                if (n > 0)
                    values[offset] = value[0];
                else
                    invalid.push_back(offset);
            },
            [](int64_t) {},
            [](int64_t) {},
            [](const Json&, const std::string&, simdjson::error_code) {});
        BuildWithNullOffsetsForUT(rows, values.data(), invalid);
    }

    void
    ExecutePhase2(const std::string& literal,
                  proto::plan::OpType op,
                  exec::SegmentExpr* segment,
                  TargetBitmap& candidates,
                  int64_t offset,
                  int64_t size) override {
        if ((op == OpType::InnerMatch && literal.size() <= max_gram_) ||
            candidates.none())
            return;
        AssertInfo(size >= 0 && candidates.size() == size_t(size),
                   "JSON ngram batch size mismatch");
        auto apply = [&](auto&& predicate) {
            // The existing JSON view API wraps these same padded chunk bytes.
            // Keep the pin alive and consume parser-owned string views before
            // the next parse; do not materialize Json wrappers for rejected rows.
            auto verify = [&](std::string_view raw) {
                Json json(raw);
                auto value = json.at<std::string_view>(path_);
                return !value.error() && predicate(value.value());
            };
            if (candidates.count() * 8 > size_t(size)) {
                auto batch = [&](const std::string_view* data,
                                 int64_t n,
                                 TargetBitmapView bits) {
                    for (auto i = bits.find_first(); i && *i < size_t(n);
                         i = bits.find_next(*i))
                        if (!verify(data[*i]))
                            bits[*i] = false;
                };
                segment->ProcessDataChunkForRange<std::string_view>(
                    batch, TargetBitmapView(candidates), offset, size);
            } else {
                segment->FilterStringCandidatesForRange(
                    verify, TargetBitmapView(candidates), offset, size);
            }
        };
        switch (op) {
            case OpType::InnerMatch:
                apply([&](auto s) {
                    return s.find(literal) != std::string_view::npos;
                });
                break;
            case OpType::PrefixMatch:
                apply([&](auto s) { return s.starts_with(literal); });
                break;
            case OpType::PostfixMatch:
                apply([&](auto s) { return s.ends_with(literal); });
                break;
            case OpType::Match: {
                LikePatternMatcher matcher(literal);
                apply([&](auto s)
                          __attribute__((flatten)) { return matcher(s); });
                break;
            }
            case OpType::RegexMatch: {
                PartialRegexMatcher matcher(literal);
                apply([&](auto s) { return matcher(s); });
                break;
            }
            default:
                ThrowInfo(Unsupported,
                          "unsupported JSON ngram phase2 operation");
        }
    }

 private:
    size_t max_gram_;
    std::string path_;
    proto::schema::FieldSchema schema_;
};
}  // namespace milvus::index
