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

#include "index/InvertedIndexKnowhere.h"
#include "index/JsonIndexBuilder.h"
#include "index/JsonKnowherePoCIO.h"
#include <bit>
#include <cmath>
#include <map>

namespace milvus::index {
// Resident, one-path scalar JSON PoC. Extraction and casting deliberately use
// the same production converter as JsonScalarIndexWrapper; no Tantivy base.
template <typename T>
class JsonPathIndexKnowhere : public InvertedIndexKnowhere<T> {
 public:
    JsonPathIndexKnowhere(const JsonCastType& cast_type,
                          std::string path,
                          const proto::schema::FieldSchema& schema,
                          JsonCastFunction cast_function =
                              JsonCastFunction::FromString("unknown"))
        : InvertedIndexKnowhere<T>(
              KnowhereSparsePostingCodec::Format::Adaptive),
          cast_type_(cast_type),
          path_(std::move(path)),
          schema_(schema),
          cast_function_(cast_function) {
        if (cast_type.data_type() == JsonCastType::DataType::ARRAY)
            ThrowInfo(Unsupported,
                      "JSON path scalar PoC does not support ARRAY casts");
    }

    std::vector<uint8_t>
    SerializeForPoC() const override {
        poc_io::Writer out;
        out.String(path_);
        out.String(cast_type_.ToString());
        out.String(schema_.SerializeAsString());
        out.U8(cast_function_.match<double>());
        out.Bytes(InvertedIndexKnowhere<T>::SerializeForPoC());
        json_poc_io::WriteBitmap(out, valid_);
        json_poc_io::WriteBitmap(out, exists_);
        out.U8(bool(nan_));
        if (nan_)
            out.Bytes(nan_->SerializeForPoC());
        return poc_io::Pack("KWJSONP1", poc_io::TypeTag<T>(), 0, out.data);
    }
    void
    LoadForPoC(std::span<const uint8_t> bytes) override {
        uint32_t codec = 0;
        poc_io::Reader in(
            poc_io::Unpack(bytes, "KWJSONP1", poc_io::TypeTag<T>(), &codec));
        poc_io::Check(codec == 0, "unsupported JSON wrapper codec");
        poc_io::Check(in.String() == path_, "JSON path mismatch");
        poc_io::Check(in.String() == cast_type_.ToString(),
                      "JSON cast type mismatch");
        json_poc_io::CheckSchema(in, schema_);
        poc_io::Check(in.U8() == uint8_t(cast_function_.match<double>()),
                      "JSON cast function mismatch");
        const auto base_bytes = in.Bytes();
        auto valid = json_poc_io::ReadBitmap(in);
        auto exists = json_poc_io::ReadBitmap(in);
        const auto has_nan = in.U8();
        poc_io::Check(has_nan <= 1, "invalid JSON NaN marker");
        std::unique_ptr<
            InvertedIndexKnowhereCore<uint64_t, TargetBitmap, OpType>>
            next_nan;
        if (has_nan) {
            poc_io::Check((std::is_same_v<T, double>),
                          "NaN postings on non-double JSON path");
            next_nan = std::make_unique<
                InvertedIndexKnowhereCore<uint64_t, TargetBitmap, OpType>>();
            next_nan->LoadForPoC(in.Bytes());
        }
        in.Finish();
        InvertedIndexKnowhere<T> staged(
            KnowhereSparsePostingCodec::Format::Adaptive);
        staged.LoadForPoC(base_bytes);
        const auto rows = size_t(staged.Count());
        poc_io::Check(valid.size() == rows && exists.size() == rows,
                      "JSON bitmap row count mismatch");
        // This adapter's snapshot deliberately keeps NaNs in its side stream;
        // the generic scalar core also supports NaN, so enforce that split here.
        if constexpr (std::is_same_v<T, double>) {
            for (size_t term = 0; term < staged.CoreForUT()->TermCount();
                 ++term)
                poc_io::Check(!std::isnan(staged.CoreForUT()->Term(term)),
                              "NaN in JSON ordinary stream");
        }
        auto ordinary = json_poc_io::PostingRows(*staged.CoreForUT(), true);
        json_poc_io::CheckEqual(
            ordinary, staged.IsNotNull(), "JSON scalar core validity mismatch");
        if (next_nan) {
            poc_io::Check(next_nan->Count() == rows &&
                              next_nan->TermCount() != 0 &&
                              next_nan->IsNull().none(),
                          "invalid JSON NaN row domain");
            for (size_t term = 0; term < next_nan->TermCount(); ++term) {
                const auto key = next_nan->Term(term);
                poc_io::Check(
                    key < OrderedDouble(
                              -std::numeric_limits<double>::infinity()) ||
                        key > OrderedDouble(
                                  std::numeric_limits<double>::infinity()),
                    "ordinary term in JSON NaN stream");
            }
            auto special = json_poc_io::PostingRows(*next_nan, true);
            auto overlap = special.clone();
            overlap &= ordinary;
            poc_io::Check(overlap.none(), "JSON row is both ordinary and NaN");
            ordinary |= special;
        }
        json_poc_io::CheckEqual(
            ordinary, valid, "JSON typed validity differs from postings");
        json_poc_io::CheckSubset(
            valid, exists, "JSON valid scalar is marked nonexistent");
        this->SwapStateForPoC(staged);
        nan_ = std::move(next_nan);
        valid_ = std::move(valid);
        exists_ = std::move(exists);
        ComputeByteSize();
    }

    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        auto converted = ConvertJsonToTypedFieldData<T>(
            fields, schema_, path_, cast_type_, cast_function_);
        auto& field = converted.field_data;
        const auto n = field->get_num_rows();
        FixedVector<T> values(n);
        std::vector<size_t> invalid;
        TargetBitmap exists(n, true), valid(n, true);
        std::map<uint64_t, std::vector<uint32_t>> nan_postings;
        for (size_t row = 0; row < n; ++row) {
            if (field->is_valid(row)) {
                values[row] = *static_cast<const T*>(field->RawValue(row));
                if constexpr (std::is_same_v<T, double>) {
                    if (std::isnan(values[row])) {
                        nan_postings[OrderedDouble(values[row])].push_back(row);
                        invalid.push_back(row);
                    }
                }
            } else {
                invalid.push_back(row);
                valid.reset(row);
            }
        }
        for (auto row : converted.non_exist_offsets) exists.reset(row);
        std::unique_ptr<
            InvertedIndexKnowhereCore<uint64_t, TargetBitmap, OpType>>
            next_nan;
        if (!nan_postings.empty()) {
            next_nan = std::make_unique<
                InvertedIndexKnowhereCore<uint64_t, TargetBitmap, OpType>>(
                KnowhereSparsePostingCodec::Format::Adaptive);
            next_nan->BuildFromPostings(n, nan_postings);
        }
        // All allocations precede publication. Base construction is transactional;
        // bitmap moves below cannot allocate. Rebuilds require exclusive access.
        InvertedIndexKnowhere<T>::BuildWithNullOffsetsForUT(
            n, values.data(), invalid);
        nan_ = std::move(next_nan);
        exists_ = std::move(exists);
        valid_ = std::move(valid);
        ComputeByteSize();
    }

    // JSON extraction must own every build so its two validity domains cannot
    // become stale after a caller supplies only typed scalar values.
    void
    Build(size_t, const T*, const bool* = nullptr) override {
        ThrowInfo(Unsupported, "JSON path PoC requires JSON FieldData build");
    }
    void
    BuildWithRawDataForUT(size_t, const void*, const Config& = {}) override {
        ThrowInfo(Unsupported, "JSON path PoC requires JSON FieldData build");
    }
    TargetBitmap
    IsNotNull() override {
        return valid_.clone();
    }
    const TargetBitmap
    IsNull() override {
        auto result = valid_.clone();
        result.flip();
        return result;
    }
    const TargetBitmap
    NotIn(size_t n, const T* values) override {
        auto result = In(n, values);
        result.flip();
        result &= valid_;
        return result;
    }
    const TargetBitmap
    In(size_t n, const T* values) override {
        if constexpr (std::is_same_v<T, double>) {
            bool has_nan = false;
            for (size_t i = 0; i < n; ++i) has_nan |= std::isnan(values[i]);
            if (has_nan) {
                std::vector<double> finite;
                std::vector<uint64_t> special;
                for (size_t i = 0; i < n; ++i)
                    if (std::isnan(values[i]))
                        special.push_back(OrderedDouble(values[i]));
                    else
                        finite.push_back(values[i]);
                auto result =
                    InvertedIndexKnowhere<T>::In(finite.size(), finite.data());
                if (nan_)
                    result |= nan_->In(special.size(), special.data());
                return result;
            }
        }
        return InvertedIndexKnowhere<T>::In(n, values);
    }
    const TargetBitmap
    Range(const T& value, OpType op) override {
        if constexpr (std::is_same_v<T, double>) {
            if (op != OpType::LessThan && op != OpType::LessEqual &&
                op != OpType::GreaterThan && op != OpType::GreaterEqual)
                return InvertedIndexKnowhere<T>::Range(T{}, op);
            if (!nan_ && !std::isnan(value))
                return InvertedIndexKnowhere<T>::Range(value, op);
            TargetBitmap result(this->Count());
            if (std::isnan(value)) {
                // The ordinary domain lies strictly between negative/positive NaNs.
                const bool less =
                    op == OpType::LessThan || op == OpType::LessEqual;
                const bool greater =
                    op == OpType::GreaterThan || op == OpType::GreaterEqual;
                if ((less && !std::signbit(value)) ||
                    (greater && std::signbit(value)))
                    result = InvertedIndexKnowhere<T>::IsNotNull();
            } else {
                result =
                    InvertedIndexKnowhere<T>::CoreForUT()->Range(value, op);
            }
            if (nan_)
                result |= nan_->Range(OrderedDouble(value), op);
            return result;
        }
        return InvertedIndexKnowhere<T>::Range(value, op);
    }
    const TargetBitmap
    Range(const T& lo, bool li, const T& hi, bool ui) override {
        if constexpr (std::is_same_v<T, double>) {
            if (!nan_ && !std::isnan(lo) && !std::isnan(hi))
                return InvertedIndexKnowhere<T>::Range(lo, li, hi, ui);
            TargetBitmap result(this->Count());
            if ((!std::isnan(lo) || std::signbit(lo)) &&
                (!std::isnan(hi) || !std::signbit(hi))) {
                const double lower =
                    std::isnan(lo) ? -std::numeric_limits<double>::infinity()
                                   : lo;
                const double upper =
                    std::isnan(hi) ? std::numeric_limits<double>::infinity()
                                   : hi;
                result = InvertedIndexKnowhere<T>::CoreForUT()->Range(
                    lower, std::isnan(lo) || li, upper, std::isnan(hi) || ui);
            }
            if (nan_)
                result |=
                    nan_->Range(OrderedDouble(lo), li, OrderedDouble(hi), ui);
            return result;
        }
        return InvertedIndexKnowhere<T>::Range(lo, li, hi, ui);
    }
    JsonCastType
    GetCastType() const override {
        return cast_type_;
    }
    TargetBitmap
    Exists() override {
        return exists_.clone();
    }
    void
    ComputeByteSize() override {
        InvertedIndexKnowhere<T>::ComputeByteSize();
        this->cached_byte_size_ += exists_.size_in_bytes() +
                                   valid_.size_in_bytes() +
                                   (nan_ ? nan_->ByteSize() : 0);
    }

 private:
    // Same sortable IEEE-754 mapping as Tantivy common::f64_to_u64, including
    // signed NaNs and payloads. Keep ordinary scalar core's NaN contract intact.
    static uint64_t
    OrderedDouble(double value) {
        if (value == 0)
            value = 0;
        const auto bits = std::bit_cast<uint64_t>(value);
        constexpr uint64_t sign = uint64_t{1} << 63;
        return (bits & sign) ? ~bits : bits ^ sign;
    }
    std::unique_ptr<InvertedIndexKnowhereCore<uint64_t, TargetBitmap, OpType>>
        nan_;
    // Hide nonvirtual scalar-only PoC entry points on the JSON adapter.
    using InvertedIndexKnowhere<T>::BuildWithNullOffsetsForUT;
    using InvertedIndexKnowhere<T>::BuildFromPostingsForPoC;
    JsonCastType cast_type_;
    std::string path_;
    proto::schema::FieldSchema schema_;
    JsonCastFunction cast_function_;
    TargetBitmap exists_;
    TargetBitmap valid_;
};
}  // namespace milvus::index
