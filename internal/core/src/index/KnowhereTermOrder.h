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
#include <cstdint>
#include <type_traits>

namespace milvus::index {
// Match Tantivy's f64 term order, including signed NaN payloads. Its FLOAT
// writer and query binding both promote f32 to f64 before creating a term.
// This is index ordering, not IEEE arithmetic comparison semantics.
template <typename T>
struct KnowhereTermOrder {
    static uint64_t
    Key(T value) requires(std::is_floating_point_v<T>) {
        double promoted = static_cast<double>(value);
        if (promoted == 0)
            promoted = 0;  // -0 and +0 share one term.
        const auto bits = std::bit_cast<uint64_t>(promoted);
        constexpr uint64_t sign = uint64_t{1} << 63;
        return (bits & sign) ? ~bits : bits ^ sign;
    }
    bool
    operator()(const T& a, const T& b) const {
        if constexpr (std::is_floating_point_v<T>)
            return Key(a) < Key(b);
        else
            return a < b;
    }
    static bool
    Equal(const T& a, const T& b) {
        if constexpr (std::is_floating_point_v<T>)
            return Key(a) == Key(b);
        else
            return a == b;
    }
};
}  // namespace milvus::index
