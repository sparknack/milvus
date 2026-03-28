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

#include <gtest/gtest.h>

#include "segcore/ExprFilterCache.h"

using namespace milvus;
using namespace milvus::segcore;

TEST(ExprFilterCacheTest, PutAndTake) {
    ExprFilterCache cache(16);

    TargetBitmap bitset(100);
    bitset.set(0);
    bitset.set(50);
    TargetBitmap valid(100);

    std::string key = "expr_plan_bytes_1";
    cache.Put(key, std::move(bitset), std::move(valid));

    // Take: move out + auto delete
    auto result = cache.Take(key);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->first.size(), 100);
    ASSERT_TRUE(result->first[0]);
    ASSERT_TRUE(result->first[50]);
    ASSERT_FALSE(result->first[1]);

    // Second Take should miss (one-shot semantics)
    auto result2 = cache.Take(key);
    ASSERT_FALSE(result2.has_value());
}

TEST(ExprFilterCacheTest, MissReturnsNullopt) {
    ExprFilterCache cache(16);
    auto result = cache.Take("nonexistent_key");
    ASSERT_FALSE(result.has_value());
}

TEST(ExprFilterCacheTest, LRUEviction) {
    ExprFilterCache cache(4);  // 4 entries

    // Fill 4 entries
    for (int i = 0; i < 4; i++) {
        cache.Put(
            "key_" + std::to_string(i), TargetBitmap(10), TargetBitmap(10));
    }

    // Insert 5th, should evict key_0 (LRU tail)
    cache.Put("key_100", TargetBitmap(10), TargetBitmap(10));

    ASSERT_FALSE(cache.Take("key_0").has_value());  // evicted
    ASSERT_TRUE(cache.Take("key_1").has_value());
    ASSERT_TRUE(cache.Take("key_2").has_value());
    ASSERT_TRUE(cache.Take("key_3").has_value());
    ASSERT_TRUE(cache.Take("key_100").has_value());
}


TEST(ExprFilterCacheTest, PutOverwrite) {
    ExprFilterCache cache(16);

    TargetBitmap b1(100);
    b1.set(0);
    cache.Put("key", std::move(b1), TargetBitmap(100));

    // Overwrite
    TargetBitmap b2(100);
    b2.set(99);
    cache.Put("key", std::move(b2), TargetBitmap(100));

    auto result = cache.Take("key");
    ASSERT_TRUE(result.has_value());
    ASSERT_FALSE(result->first[0]);   // b1's bit 0 gone
    ASSERT_TRUE(result->first[99]);   // b2's bit 99 present
}

TEST(ExprFilterCacheTest, Clear) {
    ExprFilterCache cache(16);
    cache.Put("key", TargetBitmap(10), TargetBitmap(10));
    cache.Clear();
    ASSERT_FALSE(cache.Take("key").has_value());
}

TEST(ExprFilterCacheTest, SizeTracking) {
    ExprFilterCache cache(16);
    ASSERT_EQ(cache.Size(), 0);

    cache.Put("a", TargetBitmap(10), TargetBitmap(10));
    ASSERT_EQ(cache.Size(), 1);

    cache.Put("b", TargetBitmap(10), TargetBitmap(10));
    ASSERT_EQ(cache.Size(), 2);

    // Take removes entry
    cache.Take("a");
    ASSERT_EQ(cache.Size(), 1);

    cache.Clear();
    ASSERT_EQ(cache.Size(), 0);
}
