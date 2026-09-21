// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "index/InvertedIndexKnowhere.h"
#include "index/JsonFlatIndexBase.h"
#include "index/KnowherePoCAdapterIO.h"
#include "common/Json.h"
#include <atomic>
#include <map>
#include <typeindex>
#include <unordered_set>

namespace milvus::index {
namespace knowhere_json_flat {
template <typename T>
using Core = InvertedIndexKnowhereCore<T, TargetBitmap, OpType>;
template <typename T>
struct Column {
    std::map<T, std::vector<uint32_t>> pending;
    // One checkpoint per 64 terms. Boundary sums use the core's existing DF
    // array, instead of duplicating an 8-byte prefix value for every term.
    static constexpr size_t kDFBlock = 64;
    std::vector<size_t> docfreq_prefix;
    size_t total_docfreq = 0;
    Core<T> core{KnowhereSparsePostingCodec::Format::Adaptive};
    void
    Add(const T& value, uint32_t row) {
        auto& ids = pending[value];
        if (ids.empty() || ids.back() != row)
            ids.push_back(row);
    }
    void
    Finish(size_t rows) {
        core.BuildFromPostings(rows, pending);
        RebuildFrequencies();
        decltype(pending)().swap(pending);
    }
    void
    RebuildFrequencies() {
        if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
            docfreq_prefix.clear();
            docfreq_prefix.reserve(core.TermCount() / kDFBlock + 1);
            docfreq_prefix.push_back(0);
            total_docfreq = 0;
            for (size_t i = 0; i < core.TermCount(); ++i) {
                poc_io::Check(core.DocFreq(i) <= SIZE_MAX - total_docfreq,
                              "JSON flat DF prefix overflow");
                total_docfreq += core.DocFreq(i);
                if ((i + 1) % kDFBlock == 0)
                    docfreq_prefix.push_back(total_docfreq);
            }
        }
    }
    size_t
    Prefix(size_t end) const {
        if (end == core.TermCount())
            return total_docfreq;
        const size_t block = end / kDFBlock;
        size_t sum = docfreq_prefix[block];
        for (size_t i = block * kDFBlock; i < end; ++i) sum += core.DocFreq(i);
        return sum;
    }
    size_t
    Bytes() const {
        return core.ByteSize() + docfreq_prefix.capacity() * sizeof(size_t);
    }
};
struct Columns {
    Column<bool> booleans;
    Column<int64_t> integers;
    Column<uint64_t> unsigneds;
    Column<double> doubles;
    Column<std::string> strings;
    void
    Finish(size_t rows) {
        booleans.Finish(rows);
        integers.Finish(rows);
        unsigneds.Finish(rows);
        doubles.Finish(rows);
        strings.Finish(rows);
    }
    size_t
    Bytes() const {
        return booleans.Bytes() + integers.Bytes() + unsigneds.Bytes() +
               doubles.Bytes() + strings.Bytes();
    }
};
struct Path {
    explicit Path(size_t rows)
        : exists(rows),
          arrays(rows),
          scalar_bool(rows),
          scalar_numeric(rows),
          scalar_string(rows) {
    }
    Columns scalar, elements;
    bool scalar_single_value = true;
    TargetBitmap exists, arrays, scalar_bool, scalar_numeric, scalar_string;
    size_t
    Bytes() const {
        return scalar.Bytes() + elements.Bytes() + exists.size_in_bytes() * 5;
    }
};
struct Snapshot {
    size_t rows = 0;
    TargetBitmap row_valid;
    std::map<std::string, std::shared_ptr<Path>> paths;
    mutable std::atomic<size_t> calls{0}, range_complements{0};
};
inline std::string
Escape(std::string_view key) {
    std::string out;
    for (char c : key)
        out += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
    return out;
}
// Keep I64/U64/F64 terms in separate physical dictionaries. Exact comparisons
// use an extended 64-bit integer mantissa; callers explicitly request double
// rounding where the existing raw JSON evaluator uses it. Unsupported targets
// fail at compile time instead of silently changing numeric comparisons.
static_assert(std::numeric_limits<long double>::digits >= 64);
template <typename V>
std::pair<size_t, size_t>
NumericBounds(const Core<V>& core,
              long double lo,
              bool li,
              long double hi,
              bool ui,
              bool round_to_double = false) {
    if (std::isnan(lo) || std::isnan(hi) || lo > hi)
        return {0, 0};
    auto bound = [&](long double value, bool upper) {
        size_t a = 0, b = core.TermCount();
        while (a < b) {
            auto mid = a + (b - a) / 2;
            auto term = round_to_double
                            ? static_cast<long double>(
                                  static_cast<double>(core.Term(mid)))
                            : static_cast<long double>(core.Term(mid));
            if (term < value || (upper && term == value))
                a = mid + 1;
            else
                b = mid;
        }
        return a;
    };
    auto begin = bound(lo, !li), end = bound(hi, ui);
    return {begin, std::max(begin, end)};
}
}  // namespace knowhere_json_flat

template <typename T>
class JsonFlatKnowhereExecutor : public InvertedIndexKnowhere<T>,
                                 public JsonFlatExecutorBase {
 public:
    JsonFlatKnowhereExecutor(
        std::shared_ptr<const knowhere_json_flat::Snapshot> snapshot,
        std::shared_ptr<const knowhere_json_flat::Path> path,
        bool comparable)
        : InvertedIndexKnowhere<T>(
              KnowhereSparsePostingCodec::Format::Adaptive),
          snapshot_(std::move(snapshot)),
          path_(std::move(path)),
          comparable_(comparable) {
    }
    int64_t
    Count() override {
        return snapshot_->rows;
    }
    int64_t
    Size() override {
        return Count();
    }
    void
    SetArrayQuery(bool value) override {
        array_query_ = value;
    }
    TargetBitmap
    Exists() override {
        ++snapshot_->calls;
        return path_ ? path_->exists.clone() : TargetBitmap(Count());
    }
    TargetBitmap
    ExactPathExists(JsonValueType type) override {
        if (!path_)
            return TargetBitmap(Count());
        if (array_query_)
            return path_->arrays.clone();
        if (type == JsonValueType::Bool)
            return path_->scalar_bool.clone();
        if (type == JsonValueType::Numeric)
            return path_->scalar_numeric.clone();
        if (type == JsonValueType::String)
            return path_->scalar_string.clone();
        auto result = path_->scalar_bool.clone();
        result |= path_->scalar_numeric;
        result |= path_->scalar_string;
        return result;
    }
    TargetBitmap
    IsNotNull() override {
        if (!comparable_ && !array_query_)
            return snapshot_->row_valid.clone();
        if constexpr (std::is_same_v<T, bool>)
            return ExactPathExists(JsonValueType::Bool);
        else if constexpr (std::is_same_v<T, std::string>)
            return ExactPathExists(JsonValueType::String);
        else
            return ExactPathExists(JsonValueType::Numeric);
    }
    const TargetBitmap
    IsNull() override {
        auto r = IsNotNull();
        r.flip();
        return r;
    }
    const TargetBitmap
    In(size_t n, const T* values) override {
        ++snapshot_->calls;
        TargetBitmap result(Count());
        if (!path_)
            return result;
        const auto& cols = array_query_ ? path_->elements : path_->scalar;
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<T, std::string>)
                cols.strings.core.DecodeInto(
                    cols.strings.core.Lookup(values[i]), result);
            else if constexpr (std::is_same_v<T, bool>)
                cols.booleans.core.DecodeInto(
                    cols.booleans.core.Lookup(values[i]), result);
            else
                Numeric(cols,
                        static_cast<long double>(values[i]),
                        true,
                        static_cast<long double>(values[i]),
                        true,
                        result);
        }
        return result;
    }
    const TargetBitmap
    NotIn(size_t n, const T* values) override {
        auto result = In(n, values);
        result.flip();
        result &= IsNotNull();
        return result;
    }
    const TargetBitmap
    Range(const T& value, OpType op) override {
        ++snapshot_->calls;
        TargetBitmap result(Count());
        if (!path_)
            return result;
        const auto& cols = array_query_ ? path_->elements : path_->scalar;
        if constexpr (std::is_same_v<T, std::string>)
            return cols.strings.core.Range(value, op);
        else if constexpr (std::is_same_v<T, bool>)
            return cols.booleans.core.Range(value, op);
        else {
            const long double inf =
                                  std::numeric_limits<long double>::infinity(),
                              v = value;
            switch (op) {
                case OpType::LessThan:
                    Numeric(cols, -inf, false, v, false, result, false, true);
                    break;
                case OpType::LessEqual:
                    Numeric(cols, -inf, false, v, true, result, false, true);
                    break;
                case OpType::GreaterThan:
                    Numeric(cols, v, false, inf, false, result, false, true);
                    break;
                case OpType::GreaterEqual:
                    Numeric(cols, v, true, inf, false, result, false, true);
                    break;
                default:
                    ThrowInfo(OpTypeInvalid,
                              "invalid flat numeric range operator");
            }
            return result;
        }
    }
    const TargetBitmap
    Range(const T& lo, bool li, const T& hi, bool ui) override {
        ++snapshot_->calls;
        TargetBitmap result(Count());
        if (!path_)
            return result;
        const auto& cols = array_query_ ? path_->elements : path_->scalar;
        if constexpr (std::is_same_v<T, std::string>)
            return cols.strings.core.Range(lo, li, hi, ui);
        else if constexpr (std::is_same_v<T, bool>)
            return cols.booleans.core.Range(lo, li, hi, ui);
        else {
            Numeric(cols, lo, li, hi, ui, result, true, true);
            return result;
        }
    }
    bool
    SupportPatternMatch() const override {
        return std::is_same_v<T, std::string>;
    }
    const TargetBitmap
    PatternMatch(const std::string& pattern, OpType op) override {
        ++snapshot_->calls;
        TargetBitmap result(Count());
        if constexpr (!std::is_same_v<T, std::string>)
            return InvertedIndexKnowhere<T>::PatternMatch(pattern, op);
        else {
            auto run = [&](const std::string& prefix, auto&& match) {
                if (!path_)
                    return;
                const auto& core =
                    (array_query_ ? path_->elements : path_->scalar)
                        .strings.core;
                core.ForEachStringPrefix(
                    prefix, [&](std::string_view term, uint32_t id) {
                        if (match(term))
                            core.DecodeInto(id, result);
                    });
            };
            if (op == OpType::RegexMatch) {
                // Compile before checking missing paths so malformed input
                // cannot change behavior depending on data presence.
                PartialRegexMatcher matcher(pattern);
                run("", [&](auto value) { return matcher(value); });
            } else if (op == OpType::PrefixMatch) {
                run(pattern, [](auto) { return true; });
            } else {
                std::string like;
                switch (op) {
                    case OpType::Match:
                        like = pattern;
                        break;
                    case OpType::InnerMatch:
                        like = "%" + EscapeLikePattern(pattern) + "%";
                        break;
                    case OpType::PostfixMatch:
                        like = "%" + EscapeLikePattern(pattern);
                        break;
                    default:
                        return ScalarIndex<T>::PatternMatch(pattern, op);
                }
                RegexMatcher matcher(PatternMatchTranslator{}(like));
                run(extract_fixed_prefix_from_pattern(like),
                    [&](auto value) { return matcher(value); });
            }
            return result;
        }
    }

 private:
    void
    Numeric(const knowhere_json_flat::Columns& cols,
            long double lo,
            bool li,
            long double hi,
            bool ui,
            TargetBitmap& result,
            bool binary_bounds = false,
            bool allow_complement = false) {
        // Compute identical bounds before selecting positive or complement
        // execution. Typed numeric conversion semantics are shared by both.
        const auto ib = knowhere_json_flat::NumericBounds(
            cols.integers.core, lo, li, hi, ui, std::is_floating_point_v<T>);
        const auto ub = knowhere_json_flat::NumericBounds(
            cols.unsigneds.core,
            lo,
            li,
            hi,
            ui,
            std::is_floating_point_v<T> || binary_bounds);
        const auto db = knowhere_json_flat::NumericBounds(
            cols.doubles.core, lo, li, hi, ui);
        const auto total = cols.integers.core.TermCount() +
                           cols.unsigneds.core.TermCount() +
                           cols.doubles.core.TermCount();
        const auto matched = (ib.second - ib.first) + (ub.second - ub.first) +
                             (db.second - db.first);
        auto frequencies = [](const auto& column, auto bounds) {
            const size_t hits =
                column.Prefix(bounds.second) - column.Prefix(bounds.first);
            return std::pair<size_t, size_t>{hits, column.total_docfreq - hits};
        };
        const auto idf = frequencies(cols.integers, ib);
        const auto udf = frequencies(cols.unsigneds, ub);
        const auto ddf = frequencies(cols.doubles, db);
        const size_t matched_df = idf.first + udf.first + ddf.first;
        const size_t outside_df = idf.second + udf.second + ddf.second;
        // A scalar path has one value per row. ARRAY paths can have both an
        // in-range and out-of-range value, so subtracting their postings would
        // discard valid hits. In() also always retains positive union semantics.
        const bool complement =
            allow_complement && !array_query_ && path_->scalar_single_value &&
            matched > total - matched && outside_df <= matched_df;
        TargetBitmap outside;
        if (complement) {
            ++snapshot_->range_complements;
            result = path_->scalar_numeric.clone();
            outside = TargetBitmap(Count());
        }
        auto decode = [&](const auto& core, auto bounds) {
            if (complement) {
                for (size_t i = 0; i < bounds.first; ++i)
                    core.DecodeInto(i, outside);
                for (size_t i = bounds.second; i < core.TermCount(); ++i)
                    core.DecodeInto(i, outside);
            } else {
                for (size_t i = bounds.first; i < bounds.second; ++i)
                    core.DecodeInto(i, result);
            }
        };
        decode(cols.integers.core, ib);
        decode(cols.unsigneds.core, ub);
        decode(cols.doubles.core, db);
        if (complement) {
            outside.flip();
            result &= outside;
        }
    }
    std::shared_ptr<const knowhere_json_flat::Snapshot> snapshot_;
    std::shared_ptr<const knowhere_json_flat::Path> path_;
    bool comparable_, array_query_ = false;
};

class JsonFlatIndexKnowhere : public InvertedIndexKnowhere<std::string>,
                              public JsonFlatIndexBase {
 public:
    explicit JsonFlatIndexKnowhere(std::string root_path = "")
        : InvertedIndexKnowhere<std::string>(
              KnowhereSparsePostingCodec::Format::Adaptive),
          root_path_(std::move(root_path)) {
    }
    std::vector<uint8_t>
    SerializeForPoC() const override {
        poc_io::Writer writer;
        writer.String(root_path_);
        writer.U64(snapshot_ ? snapshot_->rows : 0);
        if (snapshot_)
            poc_io::WriteBitmap(writer, snapshot_->row_valid);
        else
            poc_io::WriteBitmap(writer, TargetBitmap(0));
        writer.U64(snapshot_ ? snapshot_->paths.size() : 0);
        if (snapshot_) {
            for (const auto& [path, store] : snapshot_->paths) {
                writer.String(path);
                writer.U8(store->scalar_single_value);
                poc_io::WriteBitmap(writer, store->exists);
                poc_io::WriteBitmap(writer, store->arrays);
                poc_io::WriteBitmap(writer, store->scalar_bool);
                poc_io::WriteBitmap(writer, store->scalar_numeric);
                poc_io::WriteBitmap(writer, store->scalar_string);
                auto columns = [&](const knowhere_json_flat::Columns& values) {
                    writer.Bytes(values.booleans.core.SerializeForPoC());
                    writer.Bytes(values.integers.core.SerializeForPoC());
                    writer.Bytes(values.unsigneds.core.SerializeForPoC());
                    writer.Bytes(values.doubles.core.SerializeForPoC());
                    writer.Bytes(values.strings.core.SerializeForPoC());
                };
                columns(store->scalar);
                columns(store->elements);
            }
        }
        return poc_io::Pack("KHJFLAT1", 1, 1, writer.data);
    }
    void
    LoadForPoC(std::span<const uint8_t> bytes) override {
        using poc_io::Check;
        uint32_t codec = 0;
        poc_io::Reader reader(poc_io::Unpack(bytes, "KHJFLAT1", 1, &codec));
        Check(codec == 1, "JSON flat persistence requires adaptive format");
        const auto root = reader.String();
        auto canonical = [](std::string_view path) {
            if (!path.empty() && path.front() != '/')
                return false;
            for (size_t i = 0; i < path.size(); ++i) {
                if (path[i] != '~')
                    continue;
                if (++i == path.size() || (path[i] != '0' && path[i] != '1'))
                    return false;
            }
            return true;
        };
        Check(canonical(root) && root == root_path_,
              "JSON flat persisted root mismatch");
        auto next = std::make_shared<knowhere_json_flat::Snapshot>();
        const auto rows = reader.U64();
        Check(rows <= INT32_MAX,
              "JSON flat persisted row domain exceeds limit");
        next->rows = rows;
        next->row_valid = poc_io::ReadBitmap(reader);
        Check(next->row_valid.size() == rows,
              "JSON flat row validity domain mismatch");
        const uint64_t path_count = reader.U64();
        // Each path requires a string length, flag, five bitmap headers and
        // ten length-prefixed core envelopes. Bound before allocating entries.
        Check(path_count <= reader.Remaining() / (8 + 1 + 5 * 16 + 10 * 44),
              "JSON flat path count exceeds payload");
        std::string previous;
        for (size_t i = 0; i < path_count; ++i) {
            auto path = reader.String();
            Check(canonical(path) && (i == 0 ? path.empty() : previous < path),
                  "JSON flat paths must be canonical, ordered and rooted");
            previous = path;
            auto store = std::make_shared<knowhere_json_flat::Path>(0);
            const auto single = reader.U8();
            Check(single <= 1, "invalid JSON flat single-value flag");
            store->scalar_single_value = single;
            store->exists = poc_io::ReadBitmap(reader);
            store->arrays = poc_io::ReadBitmap(reader);
            store->scalar_bool = poc_io::ReadBitmap(reader);
            store->scalar_numeric = poc_io::ReadBitmap(reader);
            store->scalar_string = poc_io::ReadBitmap(reader);
            for (const auto* mask : {&store->exists,
                                     &store->arrays,
                                     &store->scalar_bool,
                                     &store->scalar_numeric,
                                     &store->scalar_string}) {
                poc_io::CheckSubset(*mask,
                                    next->row_valid,
                                    "JSON flat mask exceeds valid rows");
            }
            TargetBitmap scalar_seen(rows);
            bool duplicate_scalar = false;
            auto column = [&](auto& values, bool scalar) {
                values.core.LoadForPoC(reader.Bytes());
                Check(values.core.Count() == rows &&
                          values.core.IsNull().count() == 0 &&
                          values.core.PostingFormatForPoC() ==
                              KnowhereSparsePostingCodec::Format::Adaptive,
                      "JSON flat column domain/format mismatch");
                TargetBitmap coverage(rows);
                for (size_t term = 0; term < values.core.TermCount(); ++term) {
                    values.core.ForEachDoc(term, [&](size_t row) {
                        coverage.set(row);
                        if (scalar) {
                            duplicate_scalar |= bool(scalar_seen[row]);
                            scalar_seen.set(row);
                        }
                    });
                }
                // Prefix sums are query acceleration metadata, reconstructed
                // from validated DF without rebuilding/recompressing postings.
                values.RebuildFrequencies();
                poc_io::CheckSubset(coverage,
                                    scalar ? store->exists : store->arrays,
                                    "JSON flat posting shape/exists mismatch");
                if (!scalar)
                    poc_io::CheckSubset(coverage,
                                        store->exists,
                                        "JSON flat array term is absent");
                return coverage;
            };
            auto scalar_bool = column(store->scalar.booleans, true);
            auto scalar_numeric = column(store->scalar.integers, true);
            scalar_numeric |= column(store->scalar.unsigneds, true);
            scalar_numeric |= column(store->scalar.doubles, true);
            auto scalar_string = column(store->scalar.strings, true);
            poc_io::CheckEqual(scalar_bool,
                               store->scalar_bool,
                               "JSON flat boolean validity mismatch");
            poc_io::CheckEqual(scalar_numeric,
                               store->scalar_numeric,
                               "JSON flat numeric validity mismatch");
            poc_io::CheckEqual(scalar_string,
                               store->scalar_string,
                               "JSON flat string validity mismatch");
            Check(!single || !duplicate_scalar,
                  "JSON flat scalar flag hides multiple values");
            for (size_t row = 0; row < rows; ++row) {
                Check(unsigned(bool(scalar_bool[row])) +
                              unsigned(bool(scalar_numeric[row])) +
                              unsigned(bool(scalar_string[row])) +
                              unsigned(bool(store->arrays[row])) <=
                          1,
                      "JSON flat scalar and array shapes overlap");
            }
            column(store->elements.booleans, false);
            column(store->elements.integers, false);
            column(store->elements.unsigneds, false);
            column(store->elements.doubles, false);
            column(store->elements.strings, false);
            next->paths.emplace(std::move(path), std::move(store));
        }
        reader.Finish();
        for (const auto& [path, store] : next->paths) {
            if (path.empty())
                continue;
            auto parent = next->paths.find(path.substr(0, path.rfind('/')));
            Check(parent != next->paths.end(),
                  "JSON flat persisted path has no parent");
            poc_io::CheckSubset(store->exists,
                                parent->second->exists,
                                "JSON flat child exists outside parent");
        }
        snapshot_ = std::move(next);
        ComputeByteSize();
    }
    std::string
    GetNestedPath() const override {
        return root_path_;
    }
    JsonCastType
    GetCastType() const override {
        return JsonCastType::FromString("JSON");
    }
    int64_t
    Count() override {
        return snapshot_ ? snapshot_->rows : 0;
    }
    int64_t
    Size() override {
        return Count();
    }
    size_t
    RangeComplementCallsForUT() const {
        return snapshot_ ? snapshot_->range_complements.load() : 0;
    }
    size_t
    QueryCalls() const {
        return snapshot_ ? snapshot_->calls.load() : 0;
    }
    TargetBitmap
    IsNotNull() override {
        return snapshot_ ? snapshot_->row_valid.clone() : TargetBitmap(0);
    }
    const TargetBitmap
    IsNull() override {
        auto r = IsNotNull();
        r.flip();
        return r;
    }
    // Allocated payload by category, excluding allocator/map node overhead.
    std::map<std::string, size_t>
    MemoryForPoC() const {
        std::map<std::string, size_t> out;
        if (!snapshot_)
            return out;
        out["masks"] = snapshot_->row_valid.size_in_bytes();
        for (const auto& [path, store] : snapshot_->paths) {
            out["paths"] += path.capacity();
            out["masks"] += store->exists.size_in_bytes() * 5;
            auto column = [&](const auto& c) {
                const auto m = c.core.Memory();
                out["terms"] += m.terms;
                out["posting_metadata"] += m.metadata;
                out["postings"] += m.postings;
                out["null_offsets"] += m.nulls;
                out["df_prefix"] +=
                    c.docfreq_prefix.capacity() * sizeof(size_t);
            };
            for (const auto* cols : {&store->scalar, &store->elements}) {
                column(cols->booleans);
                column(cols->integers);
                column(cols->unsigneds);
                column(cols->doubles);
                column(cols->strings);
            }
        }
        return out;
    }
    void
    ComputeByteSize() override {
        this->cached_byte_size_ = sizeof(*this);
        if (!snapshot_)
            return;
        this->cached_byte_size_ += snapshot_->row_valid.size_in_bytes();
        for (const auto& [path, store] : snapshot_->paths)
            this->cached_byte_size_ += path.capacity() + store->Bytes();
    }
    std::shared_ptr<IndexBase>
    CreateExecutor(std::string path,
                   std::type_index type,
                   bool comparable) const override {
        AssertInfo(snapshot_ != nullptr,
                   "flat executor requires a completed build");
        auto it = snapshot_->paths.find(path);
        std::shared_ptr<const knowhere_json_flat::Path> data =
            it == snapshot_->paths.end() ? nullptr : it->second;
        if (type == typeid(bool))
            return std::make_shared<JsonFlatKnowhereExecutor<bool>>(
                snapshot_, data, comparable);
        if (type == typeid(int64_t))
            return std::make_shared<JsonFlatKnowhereExecutor<int64_t>>(
                snapshot_, data, comparable);
        if (type == typeid(double))
            return std::make_shared<JsonFlatKnowhereExecutor<double>>(
                snapshot_, data, comparable);
        if (type == typeid(std::string))
            return std::make_shared<JsonFlatKnowhereExecutor<std::string>>(
                snapshot_, data, comparable);
        ThrowInfo(Unsupported, "unsupported JSON flat query type");
    }
    void
    Build(size_t, const std::string*, const bool* = nullptr) override {
        ThrowInfo(Unsupported, "JSON flat PoC requires JSON FieldData build");
    }
    void
    BuildWithRawDataForUT(size_t, const void*, const Config& = {}) override {
        ThrowInfo(Unsupported, "JSON flat PoC requires JSON FieldData build");
    }
    void
    BuildWithFieldData(const std::vector<FieldDataPtr>& fields) override {
        auto next = std::make_shared<knowhere_json_flat::Snapshot>();
        for (const auto& field : fields) next->rows += field->get_num_rows();
        next->row_valid = TargetBitmap(next->rows);
        size_t row = 0;
        for (const auto& field : fields) {
            AssertInfo(field->get_data_type() == DataType::JSON,
                       "flat builder requires JSON FieldData");
            for (int64_t i = 0; i < field->get_num_rows(); ++i, ++row) {
                if (!field->is_valid(i))
                    continue;
                next->row_valid.set(row);
                const auto* json =
                    static_cast<const milvus::Json*>(field->RawValue(i));
                AssertInfo(json != nullptr, "valid JSON row has no payload");
                // A default/zero-byte Json is an absent value. dom_doc()
                // returns a default result for it, not a traversable tape.
                if (json->size() == 0)
                    continue;
                auto document = json->dom_doc();
                auto root = document.value();
                if (root_path_.empty())
                    Visit(*next, "", root, row);
                else {
                    auto value = root.at_pointer(root_path_);
                    if (!value.error())
                        Visit(*next, "", value.value(), row);
                }
            }
        }
        for (auto& [path, store] : next->paths) {
            store->scalar.Finish(next->rows);
            store->elements.Finish(next->rows);
        }
        snapshot_ = std::move(next);
        ComputeByteSize();
    }

 private:
    using InvertedIndexKnowhere<std::string>::BuildFromPostingsForPoC;
    using InvertedIndexKnowhere<std::string>::BuildWithNullOffsetsForUT;
    static bool
    Add(knowhere_json_flat::Columns& cols,
        simdjson::dom::element value,
        uint32_t row) {
        using E = simdjson::dom::element_type;
        switch (value.type()) {
            case E::BOOL:
                cols.booleans.Add(value.get_bool().value(), row);
                return true;
            case E::INT64:
                cols.integers.Add(value.get_int64().value(), row);
                return true;
            case E::UINT64:
                cols.unsigneds.Add(value.get_uint64().value(), row);
                return true;
            case E::DOUBLE:
                cols.doubles.Add(value.get_double().value(), row);
                return true;
            case E::STRING:
                cols.strings.Add(std::string(value.get_string().value()), row);
                return true;
            default:
                return false;
        }
    }
    static bool
    ContainsScalar(simdjson::dom::element value) {
        using E = simdjson::dom::element_type;
        if (value.type() == E::NULL_VALUE)
            return false;
        if (value.type() == E::OBJECT) {
            auto object = value.get_object().value();
            for (auto field : object)
                if (ContainsScalar(field.value))
                    return true;
            return false;
        }
        if (value.type() == E::ARRAY) {
            auto array = value.get_array().value();
            for (auto item : array)
                if (ContainsScalar(item))
                    return true;
            return false;
        }
        return true;
    }
    static bool
    Visit(knowhere_json_flat::Snapshot& state,
          const std::string& path,
          simdjson::dom::element value,
          uint32_t row) {
        auto& ptr = state.paths[path];
        if (!ptr)
            ptr = std::make_shared<knowhere_json_flat::Path>(state.rows);
        auto& entry = *ptr;
        using E = simdjson::dom::element_type;
        bool nonempty = false;
        if (value.type() == E::OBJECT) {
            // Own the object/array view before range-for. In C++20 a
            // reference returned by value() does not extend the temporary
            // simdjson_result lifetime through the loop.
            auto object = value.get_object().value();
            std::unordered_set<std::string_view> seen;
            for (auto field : object) {
                if (!seen.insert(field.key).second) {
                    // at_pointer binds the first duplicate key. However,
                    // ancestor exist() scans every child for nonempty content.
                    nonempty = ContainsScalar(field.value) || nonempty;
                    continue;
                }
                nonempty =
                    Visit(state,
                          path + "/" + knowhere_json_flat::Escape(field.key),
                          field.value,
                          row) ||
                    nonempty;
            }
        } else if (value.type() == E::ARRAY) {
            entry.arrays.set(row);
            size_t index = 0;
            auto array = value.get_array().value();
            for (auto item : array) {
                Add(entry.elements, item, row);
                nonempty = Visit(state,
                                 path + "/" + std::to_string(index++),
                                 item,
                                 row) ||
                           nonempty;
            }
        } else {
            nonempty = Add(entry.scalar, value, row);
            if (nonempty &&
                (entry.scalar_numeric[row] || entry.scalar_bool[row] ||
                 entry.scalar_string[row]))
                entry.scalar_single_value = false;
            if (value.type() == E::BOOL)
                entry.scalar_bool.set(row);
            else if (value.type() == E::STRING)
                entry.scalar_string.set(row);
            else if (nonempty)
                entry.scalar_numeric.set(row);
        }
        if (nonempty)
            entry.exists.set(row);
        return nonempty;
    }
    std::string root_path_;
    std::shared_ptr<const knowhere_json_flat::Snapshot> snapshot_;
};
}  // namespace milvus::index
