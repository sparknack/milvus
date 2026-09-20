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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace milvus::index {
// Knowhere-derived docID-only format, not a Knowhere serialized sparse index.
// No value payload or singleton shortcut. Integers use native little endian.
class KnowhereSparsePostingCodec {
 public:
    enum class Format { StreamVByte, Adaptive };
    static constexpr size_t kBlockSize = 256;
    static constexpr size_t kPadding = 16;
    static void
    Append(const uint32_t* ids, size_t count, std::vector<uint8_t>& bytes,
           Format format = Format::StreamVByte);

    // Trusted in-process output of Append only; not a persisted-data parser.
    // The backing allocation must have kPadding readable bytes after the blob.
    class View {
     public:
        explicit View(const uint8_t* data, Format format = Format::StreamVByte);
        uint32_t
        Count() const {
            return count_;
        }
        size_t
        Blocks() const {
            return (size_t(count_) + 255) / 256;
        }
        uint32_t
        MaxDoc(size_t block) const;
        uint32_t
        EndOffset(size_t block) const;
        size_t
        DecodeBlock(size_t block, uint32_t* ids) const;

     private:
        Format format_;
        uint32_t count_ = 0;
        const uint8_t* maxima_ = nullptr;
        const uint8_t* ends_ = nullptr;
        const uint8_t* blocks_ = nullptr;
    };
};
}  // namespace milvus::index
