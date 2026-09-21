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
#include <bit>
#include <string>
#include <string_view>
#include <type_traits>
#include "index/KnowherePackedVector.h"
namespace milvus::index {
// Lossless bit-pattern storage: signed zero/NaN payloads are never quantized.
template <typename T>
class KnowhereTermStorage {
 public:
    void
    reserve(size_t n) {
        values_.reserve(n);
    }
    void
    push_back(T v) {
        if constexpr (std::is_same_v<T, bool>)
            values_.push_back(v);
        else if constexpr (std::is_same_v<T, float>)
            values_.push_back(std::bit_cast<uint32_t>(v));
        else if constexpr (std::is_same_v<T, double>)
            values_.push_back(std::bit_cast<uint64_t>(v));
        else
            values_.push_back(std::bit_cast<std::make_unsigned_t<T>>(v));
    }
    size_t
    size() const {
        return values_.size();
    }
    bool
    empty() const {
        return values_.empty();
    }
    T
    operator[](size_t i) const {
        auto v = values_[i];
        if constexpr (std::is_same_v<T, bool>)
            return v != 0;
        else if constexpr (std::is_same_v<T, float>)
            return std::bit_cast<float>(uint32_t(v));
        else if constexpr (std::is_same_v<T, double>)
            return std::bit_cast<double>(v);
        else
            return std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(v));
    }
    T
    at(size_t i) const {
        (void)values_.at(i);
        return (*this)[i];
    }
    T
    back() const {
        return (*this)[size() - 1];
    }
    void
    Compact() {
        values_.Compact();
    }
    size_t
    Bytes() const {
        return values_.Bytes();
    }
    void
    Save(poc_io::Writer& writer) const {
        values_.Save(writer);
    }
    void
    Load(poc_io::Reader& reader, size_t n) {
        values_.Load(reader, n);
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<T, bool>)
                poc_io::Check(values_[i] <= 1, "invalid packed bool");
            else if constexpr (sizeof(T) < 8)
                poc_io::Check(values_[i] < (uint64_t{1} << (sizeof(T) * 8)),
                              "packed term exceeds type width");
        }
    }

 private:
    KnowherePackedVector values_;
};
// Each 32-term group stores its common prefix/suffix once. Packed end offsets
// address the remaining middles. Query scratch is local: no mutable shared cache.
template <>
class KnowhereTermStorage<std::string> {
 public:
    static constexpr size_t kBlock = 32;
    void
    reserve(size_t n) {
        pending_.reserve(n);
    }
    void
    push_back(std::string_view value) {
        pending_.emplace_back(value);
    }
    size_t
    size() const {
        return sealed_ ? ends_.size() : pending_.size();
    }
    bool
    empty() const {
        return size() == 0;
    }
    void
    Get(size_t i, std::string& out) const {
        if (!sealed_) {
            out = pending_[i];
            return;
        }
        const auto& b = blocks_[i / kBlock];
        auto end = ends_[i];
        auto begin = i % kBlock ? ends_[i - 1] : b.offset + b.prefix + b.suffix;
        const char* p = data_.empty() ? "" : data_.data();
        out.assign(p + b.offset, b.prefix);
        out.append(p + begin, end - begin);
        out.append(p + b.offset + b.prefix, b.suffix);
    }
    std::string
    operator[](size_t i) const {
        std::string out;
        Get(i, out);
        return out;
    }
    std::string
    at(size_t i) const {
        if (i >= size())
            throw std::out_of_range("term ordinal");
        return (*this)[i];
    }
    std::string
    back() const {
        return (*this)[size() - 1];
    }
    void
    Compact() {
        if (sealed_)
            return;
        for (size_t begin = 0; begin < pending_.size(); begin += kBlock) {
            auto end = std::min(pending_.size(), begin + kBlock);
            const auto& first = pending_[begin];
            size_t prefix = first.size(), suffix = 0;
            for (size_t i = begin + 1; i < end; ++i) {
                prefix = std::min(prefix, pending_[i].size());
                size_t j = 0;
                while (j < prefix && first[j] == pending_[i][j]) ++j;
                prefix = j;
            }
            suffix = first.size() - prefix;
            for (size_t i = begin + 1; i < end; ++i) {
                const auto& term = pending_[i];
                suffix = std::min(suffix, term.size() - prefix);
                size_t j = 0;
                while (j < suffix &&
                       first[first.size() - 1 - j] == term[term.size() - 1 - j])
                    ++j;
                suffix = j;
            }
            // Source strings are bounded by addressable memory; lengths are
            // stored as u64 on wire and in the block descriptor.
            blocks_.push_back({data_.size(), prefix, suffix});
            data_.insert(data_.end(), first.begin(), first.begin() + prefix);
            data_.insert(data_.end(), first.end() - suffix, first.end());
            for (size_t i = begin; i < end; ++i) {
                const auto& term = pending_[i];
                data_.insert(
                    data_.end(), term.begin() + prefix, term.end() - suffix);
                ends_.push_back(data_.size());
            }
        }
        ends_.Compact();
        std::vector<std::string>().swap(pending_);
        blocks_.shrink_to_fit();
        data_.shrink_to_fit();
        sealed_ = true;
    }
    size_t
    Bytes() const {
        return ends_.Bytes() + blocks_.capacity() * sizeof(Block) +
               data_.capacity();
    }
    void
    Save(poc_io::Writer& writer) const {
        writer.Bytes(
            {reinterpret_cast<const uint8_t*>(data_.data()), data_.size()});
        ends_.Save(writer);
        for (const auto& b : blocks_) {
            writer.U64(b.offset);
            writer.U64(b.prefix);
            writer.U64(b.suffix);
        }
    }
    void
    Load(poc_io::Reader& reader, size_t n) {
        using poc_io::Check;
        auto bytes = reader.Bytes();
        ends_.Load(reader, n);
        auto blocks = (n + kBlock - 1) / kBlock;
        Check(blocks <= reader.Remaining() / 24, "truncated term blocks");
        size_t previous = 0;
        for (size_t b = 0; b < blocks; ++b) {
            Block block{reader.U64(), reader.U64(), reader.U64()};
            Check(block.offset == previous && block.offset <= bytes.size(),
                  "invalid term block offset");
            Check(block.prefix <= bytes.size() - previous,
                  "invalid term prefix");
            previous += block.prefix;
            Check(block.suffix <= bytes.size() - previous,
                  "invalid term suffix");
            previous += block.suffix;
            for (size_t i = b * kBlock; i < std::min(n, (b + 1) * kBlock);
                 ++i) {
                Check(ends_[i] >= previous && ends_[i] <= bytes.size(),
                      "invalid term end");
                previous = ends_[i];
            }
            blocks_.push_back(block);
        }
        Check(previous == bytes.size(), "unreferenced term bytes");
        data_.assign(bytes.begin(), bytes.end());
        blocks_.shrink_to_fit();
        sealed_ = true;
    }

 private:
    struct Block {
        uint64_t offset, prefix, suffix;
    };
    bool sealed_ = false;
    std::vector<std::string> pending_;
    KnowherePackedVector ends_;
    std::vector<Block> blocks_;
    std::vector<char> data_;
};
}  // namespace milvus::index
