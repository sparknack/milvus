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
#include "index/KnowherePoCIO.h"
#include "common/Types.h"

namespace milvus::index::poc_io {
inline std::span<const uint8_t> UnpackAdapter(std::span<const uint8_t> blob,
                                             std::string_view magic,
                                             uint32_t type) {
    uint32_t codec = 0;
    auto payload = Unpack(blob, magic, type, &codec);
    Check(codec == 0, "unsupported PoC adapter codec");
    return payload;
}
inline BinarySet ToBinarySet(std::vector<uint8_t> bytes) {
    Check(bytes.size() <= size_t(INT64_MAX), "PoC snapshot exceeds BinarySet domain");
    auto data = std::shared_ptr<uint8_t[]>(new uint8_t[bytes.size()]);
    std::copy(bytes.begin(), bytes.end(), data.get());
    BinarySet set;
    set.Append("knowhere_poc_v1", std::move(data), bytes.size());
    return set;
}
inline std::span<const uint8_t> FromBinarySet(const BinarySet& set) {
    const auto blob = set.GetByName("knowhere_poc_v1");
    Check(set.binary_map_.size() == 1 && blob && blob->size > 0 && blob->data,
          "invalid PoC BinarySet envelope");
    return {blob->data.get(), size_t(blob->size)};
}

inline void WriteBitmap(Writer& writer, const TargetBitmap& bitmap) {
    writer.U64(bitmap.size());
    std::vector<uint8_t> bytes((bitmap.size()+7)/8, 0);
    for (size_t i=0; i<bitmap.size(); ++i)
        if (bitmap[i]) bytes[i/8] |= uint8_t(1u << (i%8));
    writer.Bytes(bytes);
}
inline TargetBitmap ReadBitmap(Reader& reader) {
    const auto count = reader.U64();
    Check(count <= INT32_MAX, "PoC bitmap row domain exceeds INT32_MAX");
    const auto bytes = reader.Bytes();
    Check(bytes.size() == (count+7)/8, "PoC bitmap length mismatch");
    if (count%8)
        Check((bytes.back() >> (count%8)) == 0, "PoC bitmap has nonzero tail bits");
    TargetBitmap bitmap(count);
    for (size_t i=0; i<count; ++i)
        if (bytes[i/8] & uint8_t(1u << (i%8))) bitmap.set(i);
    return bitmap;
}
inline void CheckEqual(const TargetBitmap& a, const TargetBitmap& b,
                       std::string_view message) {
    Check(a.size()==b.size(), message);
    for (size_t i=0; i<a.size(); ++i) Check(bool(a[i])==bool(b[i]), message);
}
inline void CheckSubset(const TargetBitmap& a, const TargetBitmap& b,
                        std::string_view message) {
    Check(a.size()==b.size(), message);
    for (size_t i=0; i<a.size(); ++i) Check(!a[i] || b[i], message);
}
} // namespace milvus::index::poc_io
