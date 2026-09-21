// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#pragma once

#include "common/EasyAssert.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace milvus::index::poc_io {
inline void
Check(bool condition, std::string_view message) {
    if (!condition)
        ThrowInfo(ErrorCode::DataFormatBroken, "{}", message);
}

inline void
Require(bool condition, const char* message) {
    Check(condition, message);
}

// FNV-1a detects accidental corruption. Structural validation remains
// mandatory; this is deliberately not an authenticity or adversarial-integrity
// primitive.
inline uint64_t
Checksum(std::span<const uint8_t> bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}
class Writer {
 public:
    std::vector<uint8_t> data;
    void
    U8(uint8_t value) {
        data.push_back(value);
    }
    void
    U32(uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) U8(value >> shift);
    }
    void
    U64(uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) U8(value >> shift);
    }
    void
    Bytes(std::span<const uint8_t> value) {
        U64(value.size());
        data.insert(data.end(), value.begin(), value.end());
    }
    void
    String(std::string_view value) {
        Bytes({reinterpret_cast<const uint8_t*>(value.data()), value.size()});
    }
};
class Reader {
 public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {
    }
    size_t
    Remaining() const {
        return bytes_.size() - position_;
    }
    uint8_t
    U8() {
        Require(Remaining() >= 1, "truncated PoC byte");
        return bytes_[position_++];
    }
    uint32_t
    U32() {
        uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            value |= uint32_t(U8()) << shift;
        return value;
    }
    uint64_t
    U64() {
        uint64_t value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8)
            value |= uint64_t(U8()) << shift;
        return value;
    }
    std::span<const uint8_t>
    Bytes() {
        const uint64_t size = U64();
        Require(size <= Remaining(), "PoC byte range exceeds payload");
        auto result = bytes_.subspan(position_, size);
        position_ += size;
        return result;
    }
    std::string
    String() {
        auto value = Bytes();
        return std::string(reinterpret_cast<const char*>(value.data()),
                           value.size());
    }
    void
    Finish() const {
        Require(Remaining() == 0, "trailing PoC payload bytes");
    }

 private:
    std::span<const uint8_t> bytes_;
    size_t position_ = 0;
};

// Envelope: magic[8], version:u32, kind:u32, codec:u32, payload length:u64,
// payload bytes, checksum:u64 over every preceding byte. No native structs.
inline std::vector<uint8_t>
Pack(std::string_view magic,
     uint32_t type,
     uint32_t codec,
     std::span<const uint8_t> payload) {
    Require(magic.size() == 8, "PoC magic must have eight bytes");
    Writer writer;
    writer.data.insert(writer.data.end(), magic.begin(), magic.end());
    writer.U32(1);
    writer.U32(type);
    writer.U32(codec);
    writer.Bytes(payload);
    writer.U64(Checksum(writer.data));
    return std::move(writer.data);
}
inline std::span<const uint8_t>
Unpack(std::span<const uint8_t> blob,
       std::string_view magic,
       uint32_t type,
       uint32_t* codec = nullptr) {
    Require(magic.size() == 8 && blob.size() >= 36, "truncated PoC envelope");
    Require(std::equal(magic.begin(), magic.end(), blob.begin()),
            "wrong PoC magic");
    Reader checksum_reader(blob.last(8));
    Require(checksum_reader.U64() == Checksum(blob.first(blob.size() - 8)),
            "PoC checksum mismatch");
    Reader reader(blob.subspan(8, blob.size() - 16));
    Require(reader.U32() == 1, "unsupported PoC format version");
    Require(reader.U32() == type, "wrong PoC object type");
    const auto format = reader.U32();
    auto payload = reader.Bytes();
    reader.Finish();
    if (codec)
        *codec = format;
    return payload;
}

template <typename T>
constexpr uint32_t
TypeTag() {
    if constexpr (std::is_same_v<T, bool>)
        return 1;
    else if constexpr (std::is_same_v<T, int8_t>)
        return 2;
    else if constexpr (std::is_same_v<T, int16_t>)
        return 3;
    else if constexpr (std::is_same_v<T, int32_t>)
        return 4;
    else if constexpr (std::is_same_v<T, int64_t>)
        return 5;
    else if constexpr (std::is_same_v<T, uint32_t>)
        return 6;
    else if constexpr (std::is_same_v<T, uint64_t>)
        return 7;
    else if constexpr (std::is_same_v<T, float>)
        return 8;
    else if constexpr (std::is_same_v<T, double>)
        return 9;
    else if constexpr (std::is_same_v<T, std::string>)
        return 10;
    else {
        static_assert(!sizeof(T), "unsupported PoC persisted term type");
    }
}
}  // namespace milvus::index::poc_io
