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
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <cstdint>

namespace milvus::index {
// Immutable after the owning core is published. String views refer directly to
// one contiguous pool; no per-term std::string objects or heap allocations.
template <typename T>
class KnowhereTermStorage {
 public:
    void
    reserve(size_t n) {
        values_.reserve(n);
    }
    void
    push_back(const T& value) {
        values_.push_back(value);
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
        return values_[i];
    }
    T
    at(size_t i) const {
        return values_.at(i);
    }
    T
    back() const {
        return values_.back();
    }
    void
    Compact() {
        values_.shrink_to_fit();
    }
    size_t
    Bytes() const {
        if constexpr (std::is_same_v<T, bool>)
            return (values_.capacity() + 7) / 8;
        else
            return values_.capacity() * sizeof(T);
    }

 private:
    std::vector<T> values_;
};
template <>
class KnowhereTermStorage<std::string> {
 public:
    void
    reserve(size_t n) {
        ends_.reserve(n);
    }
    void
    push_back(std::string_view value) {
        bytes_.insert(bytes_.end(), value.begin(), value.end());
        ends_.push_back(bytes_.size());
    }
    size_t
    size() const {
        return ends_.size();
    }
    bool
    empty() const {
        return ends_.empty();
    }
    std::string_view
    operator[](size_t i) const {
        size_t begin = i ? ends_[i - 1] : 0;
        // The empty-string-only dictionary has a zero-length pool.
        return {bytes_.empty() ? "" : bytes_.data() + begin, ends_[i] - begin};
    }
    std::string_view
    at(size_t i) const {
        (void)ends_.at(i);
        return (*this)[i];
    }
    std::string_view
    back() const {
        return (*this)[size() - 1];
    }
    void
    Compact() {
        ends_.shrink_to_fit();
        bytes_.shrink_to_fit();
    }
    size_t
    Bytes() const {
        return ends_.capacity() * sizeof(uint64_t) + bytes_.capacity();
    }

 private:
    std::vector<uint64_t> ends_;
    std::vector<char> bytes_;
};
}  // namespace milvus::index
