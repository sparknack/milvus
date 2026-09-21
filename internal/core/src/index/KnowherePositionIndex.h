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
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include <type_traits>
#include "index/KnowherePackedVector.h"

namespace milvus::index {
// Resident, immutable after publication. LoadForPoC validates persisted streams
// before publication. Flat streams are shared by all terms; offsets are bytes.
class KnowherePositionIndex {
 public:
    struct Term {
        uint64_t doc_block_begin, pos_block_begin, positions;
        uint32_t docs;
    };
    struct DocBlock {
        uint64_t freq_offset, position_base;
    };
    struct DecodeStats {
        size_t freq_blocks = 0, position_blocks = 0;
    };
    static thread_local DecodeStats decode_stats_for_ut;
    void
    AppendTerm(const std::vector<uint32_t>& frequencies,
               const std::vector<uint32_t>& deltas);
    void
    Seal();
    size_t
    ByteSize() const;
    size_t
    LogicalBytes() const;

    // Private, versioned PoC format; no production index registration.
    std::vector<uint8_t>
    SerializeForPoC() const;
    void
    LoadForPoC(std::span<const uint8_t> bytes,
               const std::vector<uint32_t>& expected_dfs);

    class Reader {
     public:
        Reader(const KnowherePositionIndex& owner, size_t term)
            : owner_(owner), term_(owner.terms_.at(term)) {
        }
        // Materialize one document only. offset aligns query token positions.
        void
        Read(size_t posting_ordinal,
             uint64_t offset,
             std::vector<uint64_t>& output);

     private:
        const KnowherePositionIndex& owner_;
        Term term_;
        size_t freq_block_ = SIZE_MAX, pos_block_ = SIZE_MAX;
        std::array<uint64_t, 257> prefix_{};
        std::array<uint32_t, 256> deltas_{};
    };

 private:
    template <typename Record>
    struct Records {
        static constexpr size_t N = std::is_same_v<Record, Term> ? 4 : 2;
        std::array<KnowherePackedVector, N> columns;
        size_t
        size() const {
            return columns[0].size();
        }
        void
        reserve(size_t n) {
            for (auto& c : columns) c.reserve(n);
        }
        void
        push_back(Record value) {
            if constexpr (std::is_same_v<Record, Term>) {
                columns[0].push_back(value.doc_block_begin);
                columns[1].push_back(value.pos_block_begin);
                columns[2].push_back(value.positions);
                columns[3].push_back(value.docs);
            } else {
                columns[0].push_back(value.freq_offset);
                columns[1].push_back(value.position_base);
            }
        }
        Record
        operator[](size_t i) const {
            if constexpr (std::is_same_v<Record, Term>)
                return {columns[0][i],
                        columns[1][i],
                        columns[2][i],
                        uint32_t(columns[3][i])};
            else
                return {columns[0][i], columns[1][i]};
        }
        Record
        at(size_t i) const {
            (void)columns[0].at(i);
            return (*this)[i];
        }
        void
        Compact() {
            for (auto& c : columns) c.Compact();
        }
        size_t
        Bytes() const {
            size_t n = 0;
            for (auto& c : columns) n += c.Bytes();
            return n;
        }
        void
        Save(poc_io::Writer& w) const {
            for (auto& c : columns) c.Save(w);
        }
        void
        Load(poc_io::Reader& r, size_t n) {
            for (auto& c : columns) c.Load(r, n);
            if constexpr (std::is_same_v<Record, Term>)
                for (size_t i = 0; i < n; ++i)
                    poc_io::Check(columns[3][i] <= UINT32_MAX,
                                  "packed position DF overflow");
        }
    };
    Records<Term> terms_;
    Records<DocBlock> doc_blocks_;
    KnowherePackedVector position_offsets_;
    std::vector<uint8_t> frequencies_, positions_;
    bool sealed_ = false;
};

// Input positions are query-offset-aligned and ordered by posting DF, stably
// for equal DF, matching the pinned Tantivy fork's no-score phrase execution.
// Owned by one query and reused across candidate documents; never share between
// concurrent queries. Keeping capacity avoids per-document span allocations.
struct KnowherePhraseScratch {
    struct Span {
        uint64_t left, right;
    };
    std::vector<Span> spans, next, best;
};

bool
KnowherePhraseExists(const std::vector<std::vector<uint64_t>>& positions,
                     uint32_t slop,
                     KnowherePhraseScratch& scratch);

bool
KnowherePhraseExists(const std::vector<std::vector<uint64_t>>& positions,
                     uint32_t slop);
}  // namespace milvus::index
