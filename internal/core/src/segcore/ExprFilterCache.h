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

#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/Types.h"

namespace milvus::segcore {

// LRU cache for filter expression bitset results in sealed segments.
// Used by two-stage search to avoid re-executing the same filter in Stage 2.
//
// Key: serialized expr plan bytes (std::string, exact match — no hash collision risk)
// Value: (TargetBitmap, valid_bitset) — the filter result, moved in and out (zero-copy)
// Semantics: Put moves data in, Take moves data out and deletes entry (one-shot).
//   Stale entries (Stage 1 wrote but Stage 2 never ran) are naturally evicted by LRU.
// Thread-safe: all operations are protected by a mutex.
class ExprFilterCache {
 public:
    struct CacheEntry {
        TargetBitmap bitset;
        TargetBitmap valid_bitset;
    };

    explicit ExprFilterCache(size_t capacity) : capacity_(capacity) {
    }

    // Move a filter result into the cache. If cache is full, evicts the LRU entry.
    void
    Put(const std::string& expr_key,
        TargetBitmap&& bitset,
        TargetBitmap&& valid_bitset) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(expr_key);
        if (it != map_.end()) {
            // Update existing entry and move to front
            it->second->second = {std::move(bitset), std::move(valid_bitset)};
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
            return;
        }
        // Evict if full
        if (map_.size() >= capacity_) {
            auto last = std::prev(lru_list_.end());
            map_.erase(last->first);
            lru_list_.pop_back();
        }
        lru_list_.emplace_front(
            expr_key,
            CacheEntry{std::move(bitset), std::move(valid_bitset)});
        map_[expr_key] = lru_list_.begin();
    }

    // Move a cached filter result out and delete the entry (one-shot semantics).
    // Returns nullopt on miss.
    // This is the expected usage pattern for two-stage search:
    //   Stage 1 Put() → Stage 2 Take() → entry gone, memory freed.
    std::optional<std::pair<TargetBitmap, TargetBitmap>>
    Take(const std::string& expr_key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(expr_key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        auto& entry = it->second->second;
        // Move out data, then delete entry
        auto result = std::make_pair(std::move(entry.bitset),
                                     std::move(entry.valid_bitset));
        lru_list_.erase(it->second);
        map_.erase(it);
        return result;
    }

    void
    Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
        lru_list_.clear();
    }

    size_t
    Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.size();
    }

 private:
    using ListType = std::list<std::pair<std::string, CacheEntry>>;
    size_t capacity_;
    mutable std::mutex mutex_;
    ListType lru_list_;  // front = MRU, back = LRU
    std::unordered_map<std::string, ListType::iterator> map_;
};

}  // namespace milvus::segcore
