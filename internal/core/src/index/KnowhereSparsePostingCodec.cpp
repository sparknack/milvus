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

#include "index/KnowhereSparsePostingCodec.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/EasyAssert.h"
#include "index/sparse/codec/adaptive.h"

// Private Knowhere ABI, deliberately confined to this translation unit.
extern "C" {
size_t
streamvbyte_encode_0124(const uint32_t*, uint32_t, uint8_t*);
size_t
streamvbyte_decode_0124(const uint8_t*, uint32_t*, uint32_t);
}

namespace milvus::index {
namespace {
uint32_t
ReadWord(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
void
WriteWord(std::vector<uint8_t>& out, size_t offset, uint32_t v) {
    std::memcpy(out.data() + offset, &v, sizeof(v));
}
}  // namespace

void
KnowhereSparsePostingCodec::Append(const uint32_t* ids,
                                   size_t count,
                                   std::vector<uint8_t>& out,
                                   Format format) {
    const uint32_t endian = 1;
    AssertInfo(*reinterpret_cast<const uint8_t*>(&endian) == 1,
               "PoC posting codec requires little endian");
    AssertInfo(count <= size_t(std::numeric_limits<int32_t>::max()),
               "PoC posting exceeds signed docID limit");
    AssertInfo(count == 0 || ids != nullptr, "missing posting docIDs");
    for (size_t i = 0; i < count; ++i) {
        AssertInfo(ids[i] <= uint32_t(std::numeric_limits<int32_t>::max()) &&
                       (i == 0 || ids[i] > ids[i - 1]),
                   "posting docIDs must be increasing signed row offsets");
    }
    uint32_t n = count;
    while (n >= 128) {
        out.push_back(n & 127);
        n >>= 7;
    }
    out.push_back(n | 128);
    if (count == 0)
        return;
    const size_t blocks = (count + 255) / 256;
    const size_t maxima = out.size();
    const size_t ends = maxima + blocks * 4;
    const size_t start = ends + (blocks - 1) * 4;
    out.resize(start);
    std::array<uint32_t, kBlockSize> deltas;
    // Both encode and decode implementations may use 16-byte SIMD accesses.
    std::array<uint8_t, kBlockSize * 4 + kBlockSize / 4 + kPadding> encoded{};
    uint32_t previous = UINT32_MAX;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t size = std::min(kBlockSize, count - b * kBlockSize);
        for (size_t j = 0; j < size; ++j) {
            uint32_t id = ids[b * kBlockSize + j];
            deltas[j] = id - previous - 1;
            previous = id;
        }
        WriteWord(out, maxima + b * 4, previous);
        if (format == Format::Adaptive) {
            knowhere::sparse::inverted::AdaptiveBlockCodec{}.encode_doc_ids(
                deltas.data(), size, out);
        } else {
            size_t length =
                streamvbyte_encode_0124(deltas.data(), size, encoded.data());
            out.insert(out.end(), encoded.data(), encoded.data() + length);
        }
        AssertInfo(out.size() - start <= UINT32_MAX,
                   "PoC posting block offsets exceed uint32");
        if (b + 1 < blocks)
            WriteWord(out, ends + b * 4, out.size() - start);
    }
}

KnowhereSparsePostingCodec::View::View(const uint8_t* data, Format format)
    : format_(format) {
    for (unsigned shift = 0;; shift += 7) {
        AssertInfo(shift <= 28, "invalid internal posting count");
        uint8_t byte = *data++;
        count_ |= uint32_t(byte & 127) << shift;
        if (byte & 128)
            break;
    }
    maxima_ = data;
    ends_ = maxima_ + Blocks() * 4;
    blocks_ = ends_ + (Blocks() ? Blocks() - 1 : 0) * 4;
}
uint32_t
KnowhereSparsePostingCodec::View::MaxDoc(size_t block) const {
    AssertInfo(block < Blocks(), "invalid posting block");
    return ReadWord(maxima_ + block * 4);
}
uint32_t
KnowhereSparsePostingCodec::View::EndOffset(size_t block) const {
    AssertInfo(block + 1 < Blocks(), "last posting block has no end offset");
    return ReadWord(ends_ + block * 4);
}
size_t
KnowhereSparsePostingCodec::View::DecodeBlock(size_t block,
                                              uint32_t* ids) const {
    AssertInfo(block < Blocks(), "invalid posting block");
    const size_t size =
        std::min(kBlockSize, size_t(count_) - block * kBlockSize);
    const auto* start = blocks_ + (block ? EndOffset(block - 1) : 0);
    if (format_ == Format::Adaptive)
        knowhere::sparse::inverted::AdaptiveBlockCodec{}.decode(start, ids, size);
    else
        streamvbyte_decode_0124(start, ids, size);
    uint32_t previous = block ? MaxDoc(block - 1) : UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        ids[i] += previous + 1;
        previous = ids[i];
    }
    return size;
}
}  // namespace milvus::index
