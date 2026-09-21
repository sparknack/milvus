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
#include "index/KnowherePoCIO.h"

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
    const bool short_list = format == Format::Adaptive && count <= kBlockSize;
    const size_t maxima = out.size();
    const size_t ends = maxima + (short_list ? 0 : blocks * 4);
    const size_t start = ends + (short_list ? 0 : (blocks - 1) * 4);
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
        if (!short_list)
            WriteWord(out, maxima + b * 4, previous);
        if (format != Format::StreamVByte) {
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

std::array<uint32_t, KnowhereSparsePostingCodec::kBlockSize>
KnowhereSparsePostingCodec::DecodeChecked(std::span<const uint8_t> bytes,
                                          size_t count,
                                          Format format) {
    using poc_io::Check;
    Check(count > 0 && count <= kBlockSize, "invalid persisted block count");
    Check(format == Format::Adaptive || format == Format::AdaptiveLegacy ||
              format == Format::StreamVByte,
          "unknown persisted block codec");
    size_t offset = 0;
    auto byte = [&]() {
        Check(offset < bytes.size(), "truncated persisted block header");
        return bytes[offset++];
    };
    auto skip = [&](size_t length) {
        Check(length <= bytes.size() - offset,
              "truncated persisted block payload");
        offset += length;
    };
    auto streamvbyte = [&]() {
        const size_t controls = (count + 3) / 4;
        Check(controls <= bytes.size() - offset,
              "truncated streamvbyte controls");
        const auto begin = offset;
        skip(controls);
        constexpr size_t lengths[4] = {0, 1, 2, 4};
        size_t payload = 0;
        for (size_t i = 0; i < count; ++i)
            payload += lengths[(bytes[begin + i / 4] >> (2 * (i % 4))) & 3];
        skip(payload);
    };
    if (format == Format::StreamVByte) {
        streamvbyte();
    } else {
        const uint8_t tag = byte();
        if (tag <= 32) {
            skip((count * tag + 7) / 8);
        } else if (tag >= 33 && tag <= 35) {
            skip(size_t{1} << (tag - 33));
        } else if (tag == 36) {
            streamvbyte();
        } else if (tag == 37) {
            const uint8_t token = byte(), bits = token & 31,
                          exceptions = token >> 5;
            if (bits) {
                skip((count * bits + 7) / 8);
            } else {
                // PFor uses ordinary continuation-high-bit vint (unlike the
                // outer posting count's stop-high-bit encoding).
                for (unsigned group = 0;; ++group) {
                    Check(group < 5, "overflowing adaptive base vint");
                    const uint8_t value = byte();
                    Check(group < 4 || (value & 0xf0) == 0,
                          "overflowing adaptive base vint");
                    if (!(value & 128))
                        break;
                }
            }
            int previous = -1;
            for (unsigned i = 0; i < exceptions; ++i) {
                const auto position = byte(), patch = byte();
                Check(position < count && position > previous,
                      "invalid adaptive exception position");
                Check(patch != 0 && (uint64_t(patch) << bits) <= UINT32_MAX,
                      "overflowing adaptive exception patch");
                previous = position;
            }
        } else {
            Check(false, "unknown adaptive encoding tag");
        }
    }
    Check(offset == bytes.size(), "trailing persisted block bytes");
    // The bounded structural pass precedes every trusted pointer-only decode.
    // A private padded copy also makes an isolated last block SIMD-readable.
    // Worst-case block is the adaptive streamvbyte tag + one control byte
    // per four integers + four payload bytes per integer. Avoid allocating a
    // temporary vector for each validated block during index loading.
    constexpr size_t max_block_bytes =
        1 + (kBlockSize + 3) / 4 + 4 * kBlockSize;
    Check(bytes.size() <= max_block_bytes,
          "persisted block exceeds codec bound");
    std::array<uint8_t, max_block_bytes + kPadding> padded{};
    std::copy(bytes.begin(), bytes.end(), padded.begin());
    std::array<uint32_t, kBlockSize> output{};
    const uint8_t* end;
    if (format != Format::StreamVByte) {
        end = knowhere::sparse::inverted::AdaptiveBlockCodec{}.decode(
            padded.data(), output.data(), count);
    } else {
        end = padded.data() +
              streamvbyte_decode_0124(padded.data(), output.data(), count);
    }
    Check(end == padded.data() + bytes.size(),
          "persisted decoder length mismatch");
    return output;
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
    short_ = format == Format::Adaptive && count_ <= kBlockSize;
    if (short_) {
        blocks_ = data;
        return;
    }
    maxima_ = data;
    ends_ = maxima_ + Blocks() * 4;
    blocks_ = ends_ + (Blocks() ? Blocks() - 1 : 0) * 4;
}
uint32_t
KnowhereSparsePostingCodec::View::MaxDoc(size_t block) const {
    AssertInfo(block < Blocks(), "invalid posting block");
    if (short_) {
        std::array<uint32_t, kBlockSize> ids;
        auto n = DecodeBlock(0, ids.data());
        return ids[n - 1];
    }
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
    if (format_ != Format::StreamVByte)
        knowhere::sparse::inverted::AdaptiveBlockCodec{}.decode(
            start, ids, size);
    else
        streamvbyte_decode_0124(start, ids, size);
    uint32_t previous = block ? MaxDoc(block - 1) : UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        ids[i] += previous + 1;
        previous = ids[i];
    }
    return size;
}
void
KnowhereSparsePostingCodec::Cursor::LoadBlock(size_t block) {
    block_ = block;
    size_ = view_.DecodeBlock(block, ids_.data());
    position_ = 0;
    started_ = true;
    ++decoded_blocks_;
}

uint32_t
KnowhereSparsePostingCodec::Cursor::Seek(uint32_t target) {
    if (ended_)
        return kEnd;
    if (target == kEnd || view_.Blocks() == 0) {
        ended_ = true;
        return kEnd;
    }
    if (started_ && target <= Doc())
        return Doc();
    // Dense candidate iteration usually asks for the next posting. Avoid a
    // block-directory lookup and binary search in that already decoded case.
    if (started_ && position_ + 1 < size_ && ids_[position_ + 1] >= target) {
        ++position_;
        return Doc();
    }
    size_t block = started_ ? block_ : 0;
    if (!view_.Short() && view_.MaxDoc(block) < target) {
        size_t lo = block + 1, hi = view_.Blocks();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (view_.MaxDoc(mid) < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        block = lo;
    }
    if (block == view_.Blocks()) {
        ended_ = true;
        return kEnd;
    }
    if (!started_ || block != block_)
        LoadBlock(block);
    // Directory-free short lists may seek past their last document.
    position_ = std::lower_bound(
                    ids_.begin() + position_, ids_.begin() + size_, target) -
                ids_.begin();
    if (position_ == size_) {
        ended_ = true;
        return kEnd;
    }
    return Doc();
}

uint32_t
KnowhereSparsePostingCodec::Cursor::Next() {
    if (ended_)
        return kEnd;
    if (!started_)
        return Seek(0);
    if (++position_ == size_) {
        if (block_ + 1 == view_.Blocks()) {
            ended_ = true;
            return kEnd;
        }
        LoadBlock(block_ + 1);
    }
    return Doc();
}
}  // namespace milvus::index
