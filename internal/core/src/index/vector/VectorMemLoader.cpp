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

#include "index/vector/VectorMemLoader.h"
#include "index/vector/VectorLoadUtils.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/Utils.h"
#include "index/Meta.h"
#include "index/vector/VectorIndexReader.h"
#include "index/vector/VectorIndexValidDataUtils.h"
#include "index/vector/VectorLoadResource.h"
#include "knowhere/binaryset.h"
#include "knowhere/comp/index_param.h"
#include "nlohmann/json.hpp"
#include "storage/artifact/LocalDirectory.h"

namespace milvus::index {
namespace {

using vector_load_params::ArtifactState;
using vector_load_params::DimensionSource;
using vector_load_params::LoadBackend;
using vector_load_params::NormalizedLoadMetadata;
using vector_load_params::ParseIdMapMmapMetadata;
using vector_load_params::ParseNormalizedLoadMetadata;
using vector_load_params::ParseVectorShapeMetadata;
using vector_load_params::ValidateLoadedDimension;
using vector_load_params::ValidateLoadedShape;

using RuntimeParams = NormalizedLoadMetadata;

RuntimeParams
ParseRuntimeParams(const Config& params) {
    auto result = ParseNormalizedLoadMetadata(params, LoadBackend::Memory);
    if (VectorUsesDiskLoad(result.index_type, result.version)) {
        ThrowInfo(UnexpectedError,
                  "disk-load vector index {} was routed to vector_mem",
                  result.index_type);
    }
    ParseVectorShapeMetadata(params, result);
    ParseIdMapMmapMetadata(params, result);
    return result;
}

struct EntryPlan {
    ArtifactState state{ArtifactState::Normal};
    std::vector<std::string> all_names;
    std::vector<std::string> engine_names;
    bool has_validity{false};
    bool has_emb_meta{false};
    bool has_emb_raw{false};
};

EntryPlan
PlanEntries(storage::FileSource& source, const RuntimeParams& params) {
    EntryPlan plan;
    plan.all_names = source.EntryNames();
    std::unordered_set<std::string> unique;
    unique.reserve(plan.all_names.size());
    bool has_valid_count = false;
    bool has_valid_data = false;
    bool has_empty_offsets = false;
    for (const auto& name : plan.all_names) {
        if (name.empty() || !unique.insert(name).second) {
            ThrowInfo(DataFormatBroken,
                      "vector artifact has an empty or duplicate entry name");
        }
        if (name == VALID_DATA_COUNT_KEY) {
            has_valid_count = true;
        } else if (name == VALID_DATA_KEY) {
            has_valid_data = true;
        } else if (name == EMPTY_EMB_LIST_OFFSETS_KEY) {
            has_empty_offsets = true;
        } else if (name == knowhere::meta::EMB_LIST_META) {
            plan.has_emb_meta = true;
        } else if (name == knowhere::meta::EMB_LIST_RAW_INDEX) {
            plan.has_emb_raw = true;
        } else {
            plan.engine_names.push_back(name);
        }
    }
    if (has_valid_count != has_valid_data) {
        ThrowInfo(DataFormatBroken,
                  "nullable vector valid_data entries are incomplete");
    }
    plan.has_validity = has_valid_count;

    if (has_empty_offsets) {
        if (params.elem_type == DataType::NONE || !plan.engine_names.empty() ||
            plan.has_emb_meta || plan.has_emb_raw) {
            ThrowInfo(DataFormatBroken,
                      "empty embedding-list artifact contains incompatible "
                      "engine entries");
        }
        plan.state = ArtifactState::EmptyEmbeddingList;
        return plan;
    }

    if (plan.engine_names.empty()) {
        if (!plan.has_validity || plan.has_emb_meta || plan.has_emb_raw) {
            ThrowInfo(DataFormatBroken,
                      "vector artifact has no loadable engine state");
        }
        plan.state = ArtifactState::AllNull;
        return plan;
    }

    if (params.elem_type == DataType::NONE) {
        if (plan.has_emb_meta || plan.has_emb_raw) {
            ThrowInfo(DataFormatBroken,
                      "ordinary vector artifact contains embedding-list "
                      "sidecars");
        }
    } else if (!plan.has_emb_meta) {
        ThrowInfo(DataFormatBroken,
                  "embedding-list vector artifact has no EMB_LIST_META entry");
    }
    return plan;
}

void
AppendReadEntry(storage::FileSource& source,
                const std::string& name,
                knowhere::BinarySet& entries) {
    auto bytes = source.ReadEntry(name);
    if (bytes.size() >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        ThrowInfo(DataFormatBroken,
                  "vector artifact entry {} exceeds BinarySet size",
                  name);
    }
    const auto size = static_cast<int64_t>(bytes.size());
    auto owner = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
    std::shared_ptr<uint8_t[]> data(owner, owner->data());
    entries.Append(name, std::move(data), size);
}

knowhere::BinarySet
ReadEntries(storage::FileSource& source,
            const std::vector<std::string>& names) {
    knowhere::BinarySet entries;
    for (const auto& name : names) {
        AppendReadEntry(source, name, entries);
    }
    return entries;
}

using detail::EmptyEmbeddingListState;

EmptyEmbeddingListState
DecodeEmptyEmbeddingList(const knowhere::BinarySet& entries) {
    const auto entry =
        entries.GetByName(std::string(EMPTY_EMB_LIST_OFFSETS_KEY));
    constexpr auto header_size = detail::kEmptyEmbeddingListHeaderSize;
    if (entry == nullptr || entry->size < 0 ||
        static_cast<uint64_t>(entry->size) < header_size ||
        entry->data == nullptr) {
        ThrowInfo(DataFormatBroken,
                  "empty embedding-list offset entry is invalid");
    }
    auto result = detail::DecodeEmptyEmbeddingListPayload(
        entry->data.get(), static_cast<uint64_t>(entry->size));
    if (!detail::IsValidEmptyEmbeddingListOffsets(result.offsets)) {
        ThrowInfo(DataFormatBroken,
                  "empty embedding-list offsets are not an all-zero prefix "
                  "sum");
    }
    if (!detail::IsValidEmptyEmbeddingListDimension(result.dim)) {
        ThrowInfo(DataFormatBroken,
                  "empty embedding-list dimension {} is invalid",
                  result.dim);
    }
    return result;
}

void
SetWarmup(Config& config, storage::WarmupPolicy warmup) {
    switch (warmup) {
        case storage::WarmupPolicy::Disable:
            config[WARMUP] = "disable";
            return;
        case storage::WarmupPolicy::Sync:
            config[WARMUP] = "sync";
            return;
        case storage::WarmupPolicy::Async:
            config[WARMUP] = "async";
            return;
    }
    ThrowInfo(UnexpectedError, "unknown vector warmup policy");
}

[[noreturn]] void
ThrowDeserializeError(knowhere::Status status) {
    ThrowInfo(KnowhereStatusToErrorCode(status),
              "failed to deserialize vector index: status {} ({})",
              static_cast<int>(status),
              knowhere::Status2String(status));
}

// Publish the persisted validity into the engine's id map. This must happen
// BEFORE the engine is deserialized: knowhere derives both mapping directions
// from the bitmap inside Deserialize, and a metadata-only artifact never
// reaches Deserialize at all, so its map is finalized explicitly.
RestoredIdMap
RestoreValidity(storage::FileSource& source,
                const EntryPlan& plan,
                const RuntimeParams& params,
                KnowhereEngine& engine,
                const std::string& mmap_path_prefix,
                knowhere::BinarySet* existing = nullptr) {
    if (!plan.has_validity) {
        return {};
    }
    knowhere::BinarySet local;
    auto& entries = existing == nullptr ? local : *existing;
    if (!entries.Contains(VALID_DATA_COUNT_KEY)) {
        AppendReadEntry(source, VALID_DATA_COUNT_KEY, entries);
        AppendReadEntry(source, VALID_DATA_KEY, entries);
    }
    const auto mmap_flags =
        mmap_path_prefix.empty() ? IdMapMmapFlags{} : params.id_map_mmap;
    return RestoreIdMapFromBinarySet(
        entries, engine.native_index.GetIdMap(), mmap_flags, mmap_path_prefix);
}

struct OpenedMemState {
    explicit OpenedMemState(const RuntimeParams& params)
        : engine(params.physical_type,
                 params.elem_type,
                 params.index_type,
                 params.metric_type,
                 params.version) {
    }

    KnowhereEngine engine;
};

void
PopulateState(OpenedMemState& state,
              storage::FileSource& source,
              const storage::LoadOptions& opts,
              const RuntimeParams& params,
              const EntryPlan& plan) {
    auto config = params.knowhere_config;
    config.erase(MMAP_FILE_PATH);
    config.erase(EMB_LIST_META_PATH);
    config.erase(EMB_LIST_RAW_INDEX_PATH);
    SetWarmup(config, opts.warmup);

    // The id map's derived arrays may be file-backed, and knowhere removes
    // each backing file with its mapping, so the staging directory only has to
    // outlive the engine. A metadata-only artifact owns no engine file, so it
    // creates that directory on its own when mmap was requested.
    const auto id_map_mmap_requested = params.id_map_mmap.Any();
    auto stage_id_map_mmap_dir = [&]() -> std::string {
        if (!plan.has_validity || !id_map_mmap_requested) {
            return {};
        }
        auto local_files = storage::LocalDirectory::CreateOwned(
            opts.mmap_dir_path, "vector_id_map_XXXXXX", "vector id map mmap");
        auto path = local_files->Path();
        state.engine.backing_owner = std::move(local_files);
        return path;
    };

    if (plan.state == ArtifactState::EmptyEmbeddingList) {
        auto entries = ReadEntries(source, plan.all_names);
        auto empty = DecodeEmptyEmbeddingList(entries);
        state.engine.SetDim(empty.dim);
        state.engine.SetEmptyEmbListOffsets(std::move(empty.offsets));
        const auto restored = RestoreValidity(source,
                                              plan,
                                              params,
                                              state.engine,
                                              stage_id_map_mmap_dir(),
                                              &entries);
        if (restored.has_valid_data) {
            FinalizeRestoredIdMap(state.engine.native_index.Node(),
                                  "empty embedding-list vector load");
        }
        ValidateLoadedDimension(params,
                                state.engine.Dim(),
                                DimensionSource::PersistedArtifact,
                                LoadBackend::Memory);
    } else if (plan.state == ArtifactState::AllNull) {
        if (!params.runtime_dim.has_value()) {
            ThrowInfo(UnexpectedError,
                      "validity-only vector artifact requires runtime dim");
        }
        state.engine.SetDim(*params.runtime_dim);
        const auto restored = RestoreValidity(
            source, plan, params, state.engine, stage_id_map_mmap_dir());
        if (restored.has_valid_data) {
            FinalizeRestoredIdMap(state.engine.native_index.Node(),
                                  "all-null nullable vector load");
        }
        ValidateLoadedDimension(params,
                                state.engine.Dim(),
                                DimensionSource::RuntimeMetadata,
                                LoadBackend::Memory);
    } else {
        const bool mmap =
            opts.enable_mmap &&
            KnowhereMmapSupported(state.engine.KnowhereIndexType());
        if (mmap) {
            auto local_files = storage::LocalDirectory::CreateOwned(
                opts.mmap_dir_path, "vector_mem_XXXXXX", "vector mmap");
            const auto& directory = local_files->Path();
            state.engine.backing_owner = std::move(local_files);
            const auto main_path =
                (std::filesystem::path(directory) / "index").string();
            source.ReadEntriesToLocalFile(plan.engine_names, main_path);
            config[ENABLE_MMAP] = true;
            if (params.elem_type != DataType::NONE) {
                const auto meta_path =
                    (std::filesystem::path(directory) / EMB_LIST_META_FILE_NAME)
                        .string();
                source.ReadEntryToLocalFile(knowhere::meta::EMB_LIST_META,
                                            meta_path);
                config[EMB_LIST_META_PATH] = meta_path;
                if (plan.has_emb_raw) {
                    const auto raw_path = (std::filesystem::path(directory) /
                                           EMB_LIST_RAW_INDEX_FILE_NAME)
                                              .string();
                    source.ReadEntryToLocalFile(
                        knowhere::meta::EMB_LIST_RAW_INDEX, raw_path);
                    config[EMB_LIST_RAW_INDEX_PATH] = raw_path;
                }
            }
            // Restore before deserializing: Deserialize is what derives the
            // dense id arrays from the validity bitmap.
            RestoreValidity(source, plan, params, state.engine, directory);
            const auto status = state.engine.native_index.DeserializeFromFile(
                main_path, config);
            // The knowhere deserialize API has no OpContext entrance. Remote
            // reads and local materialization observe the context captured by
            // FileSource; the borrowed opts.op_ctx is never retained here.
            if (status != knowhere::Status::success) {
                ThrowDeserializeError(status);
            }
        } else {
            config[ENABLE_MMAP] = false;
            auto entries = ReadEntries(source, plan.all_names);
            RestoreValidity(
                source, plan, params, state.engine, std::string{}, &entries);
            const auto status =
                state.engine.native_index.Deserialize(entries, config);
            if (status != knowhere::Status::success) {
                ThrowDeserializeError(status);
            }
            entries.clear();
        }
        state.engine.SetDim(state.engine.native_index.Dim());
        ValidateLoadedDimension(params,
                                state.engine.Dim(),
                                DimensionSource::PersistedArtifact,
                                LoadBackend::Memory);
    }

    ValidateLoadedShape(params, plan.state, state.engine, LoadBackend::Memory);
}

}  // namespace

ReaderCaps
VectorMemLoader::DeriveCaps(const Config& index_meta) {
    (void)ParseRuntimeParams(index_meta);
    // ReaderCaps is scalar-shaped. Family/type metadata identifies this as a
    // vector reader; no vector capability expansion is implied here.
    return {};
}

IIndexReaderBasePtr
VectorMemLoader::Open(storage::FileSource& source,
                      const storage::LoadOptions& opts) {
    const auto params = ParseRuntimeParams(opts.params);
    if (source.Gen() != storage::Generation::V1V2) {
        ThrowInfo(Unsupported,
                  "in-memory vector indexes have no V3 persisted format");
    }
    const auto plan = PlanEntries(source, params);
    if (plan.has_validity && params.id_map_mmap.Any() &&
        opts.mmap_dir_path.empty()) {
        ThrowInfo(UnexpectedError,
                  "nullable vector mmap mapping requires a staging parent");
    }
    OpenedMemState state(params);
    PopulateState(state, source, opts, params, plan);
    return std::make_unique<VectorIndexReader>(std::move(state.engine));
}

}  // namespace milvus::index
