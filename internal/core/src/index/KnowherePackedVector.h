// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>
#include "index/KnowherePoCIO.h"

namespace milvus::index {
// Immutable block frame-of-reference integers. No expanded resident copy.
// Every 128 values share a base and bit width; random access reads one value.
class KnowherePackedVector {
 public:
    static constexpr size_t kBlock = 128;
    void
    reserve(size_t n) {
        pending_.reserve(n);
    }
    void
    push_back(uint64_t v) {
        pending_.push_back(v);
    }
    size_t
    size() const {
        return packed_ ? count_ : pending_.size();
    }
    bool
    empty() const {
        return size() == 0;
    }
    uint64_t
    operator[](size_t i) const {
        if (!packed_)
            return pending_[i];
        const auto* p = data_.data() + offsets_[i / kBlock];
        uint64_t base = Read(p);
        unsigned bits = p[8];
        if (bits == 0)
            return base;
        const size_t bit = (i % kBlock) * bits;
        const unsigned shift = bit % 8;
        const auto* payload = p + 9 + bit / 8;
        uint64_t value;
        std::memcpy(
            &value, payload, 8);  // private read padding, never serialized
        if constexpr (std::endian::native != std::endian::little)
            value = Read(payload);
        value >>= shift;
        if (bits + shift > 64)
            value |= uint64_t(payload[8]) << (64 - shift);
        if (bits < 64)
            value &= (uint64_t{1} << bits) - 1;
        return base + value;
    }
    uint64_t
    at(size_t i) const {
        if (i >= size())
            throw std::out_of_range("packed ordinal");
        return (*this)[i];
    }
    uint64_t
    back() const {
        return (*this)[size() - 1];
    }
    void
    Compact() {
        if (packed_)
            return;
        count_ = pending_.size();
        for (size_t begin = 0; begin < count_; begin += kBlock) {
            size_t end = std::min(count_, begin + kBlock);
            auto [lo, hi] = std::minmax_element(pending_.begin() + begin,
                                                pending_.begin() + end);
            uint64_t base = *lo;
            unsigned bits = std::bit_width(*hi - base);
            offsets_.push_back(data_.size());
            for (unsigned b = 0; b < 8; ++b) data_.push_back(base >> (8 * b));
            data_.push_back(bits);
            const auto start = data_.size();
            data_.resize(start + ((end - begin) * bits + 7) / 8, 0);
            for (size_t i = begin; i < end; ++i) {
                uint64_t value = pending_[i] - base;
                size_t bit = (i - begin) * bits;
                for (unsigned done = 0; done < bits;) {
                    unsigned shift = (bit + done) % 8,
                             take = std::min<unsigned>(8 - shift, bits - done);
                    data_[start + (bit + done) / 8] |=
                        ((value >> done) & ((1u << take) - 1)) << shift;
                    done += take;
                }
            }
        }
        std::vector<uint64_t>().swap(pending_);
        data_.resize(data_.size() + 8, 0);
        offsets_.shrink_to_fit();
        data_.shrink_to_fit();
        packed_ = true;
    }
    size_t
    Bytes() const {
        return pending_.capacity() * 8 + offsets_.capacity() * 8 +
               data_.capacity();
    }
    void
    Save(poc_io::Writer& writer) const {
        writer.Bytes(std::span<const uint8_t>(
            data_.data(), data_.empty() ? 0 : data_.size() - 8));
    }
    void
    Load(poc_io::Reader& reader, size_t count) {
        using poc_io::Check;
        auto bytes = reader.Bytes();
        KnowherePackedVector next;
        next.count_ = count;
        next.packed_ = true;
        Check((count / kBlock + (count % kBlock != 0)) <= bytes.size() / 9,
              "truncated packed integer blocks");
        size_t offset = 0;
        for (size_t i = 0; i < count; i += kBlock) {
            Check(bytes.size() - offset >= 9,
                  "truncated packed integer header");
            const auto* p = bytes.data() + offset;
            unsigned bits = p[8];
            Check(bits <= 64, "invalid packed integer width");
            auto size = 9 + (std::min(kBlock, count - i) * bits + 7) / 8;
            Check(size <= bytes.size() - offset,
                  "truncated packed integer payload");
            next.offsets_.push_back(offset);
            offset += size;
        }
        Check(offset == bytes.size(), "trailing packed integer bytes");
        next.data_.reserve(bytes.size() + 8);
        next.data_.assign(bytes.begin(), bytes.end());
        next.data_.resize(next.data_.size() + 8, 0);
        // Reject overflow even for values that semantic callers do not use.
        for (size_t i = 0; i < count; ++i) {
            uint64_t base = Read(next.data_.data() + next.offsets_[i / kBlock]);
            Check(next[i] >= base, "overflowing packed integer value");
        }
        next.offsets_.shrink_to_fit();
        *this = std::move(next);
    }

 private:
    static uint64_t
    Read(const uint8_t* p) {
        uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
        return v;
    }
    bool packed_ = false;
    size_t count_ = 0;
    std::vector<uint64_t> pending_, offsets_;
    std::vector<uint8_t> data_;
};
}  // namespace milvus::index
