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

#include <concepts>
#include <variant>
#include "folly/CancellationToken.h"
#include "folly/coro/Task.h"
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "common/Types.h"
#include "index/contracts/build/IArtifactBuilder.h"
#include "index/IndexLoadPlan.h"
#include "storage/IndexEntryFormat.h"
#include "index/contracts/query/IIndexReaderBase.h"
#include "index/contracts/query/ReaderCaps.h"
#include "storage/artifact/FileSource.h"
#include "storage/artifact/LoadOptions.h"

// Family-level loader / builder registries. Selection parameters identify a
// family; the selected implementation parses its own remaining parameters.

namespace milvus::index {

// "inverted" / "bitmap" / "sort" / "marisa" / "text" / "ngram" /
// "json_flat" / "rtree" / "fmindex" / vector families...
//
// A canonical registry key. Load planning derives it from runtime parameters
// and, for the scalar hybrid family, the persisted selector.
using IndexFamily = std::string;

struct LoaderEntry {
    using DeriveCapsFn = ReaderCaps (*)(const Config&);
    using OpenFn = IIndexReaderBasePtr (*)(storage::FileSource&,
                                           const storage::LoadOptions&);

    using OpenAsyncFn = folly::coro::Task<IIndexReaderBasePtr> (*)(
        storage::FileSource&, const storage::LoadOptions&);

    using PlanPackedFn = IndexLoadPlan (*)(const storage::IndexEntryDirectory&,
                                           const nlohmann::json&,
                                           const storage::LoadOptions&);
    using FinishPackedSyncFn =
        IIndexReaderBasePtr (*)(IndexLoadPlan&, const storage::LoadOptions&);
    // Async finalizers yield between batches; false preserves inline execution.
    using FinishPackedAsyncFn =
        folly::coro::Task<IIndexReaderBasePtr> (*)(IndexLoadPlan&,
                                                   const storage::LoadOptions&,
                                                   bool,
                                                   folly::CancellationToken);
    using FinishPackedFn =
        std::variant<FinishPackedSyncFn, FinishPackedAsyncFn>;

    // Derive capabilities from load-time metadata without opening payloads.
    // Inventory uses this before pinning and compares it with Reader::Caps()
    // after open.
    DeriveCapsFn derive_caps{nullptr};

    // Open persisted bytes into one uniquely owned reader. mmap providers may
    // parse bounded metadata while leaving bulk payloads file-backed.
    OpenFn open{nullptr};

    // Packed loaders plan final destinations before any payload read. The
    // orchestrator retains the plan through initialization and commits only
    // after a complete reader has been constructed. Null for legacy-only families.
    PlanPackedFn plan_packed{nullptr};
    FinishPackedFn finish_packed{FinishPackedSyncFn{nullptr}};

    // Legacy transport suspends; the orchestrator supplies the loading executor.
    OpenAsyncFn open_async{nullptr};

    explicit operator bool() const noexcept {
        return derive_caps != nullptr && open != nullptr;
    }
};

template <typename Provider>
concept StaticLoaderProvider =
    requires(const Config& params,
             storage::FileSource& source,
             const storage::LoadOptions& options) {
        { Provider::kFamily } -> std::convertible_to<std::string_view>;
        { Provider::DeriveCaps(params) } -> std::same_as<ReaderCaps>;
        {
            Provider::Open(source, options)
            } -> std::same_as<IIndexReaderBasePtr>;
    };

// Family-keyed registry of stateless load function pairs. Load planning uses
// the selected entry to derive metadata-only capabilities before opening any
// payload.
class LoaderRegistry {
 public:
    static LoaderRegistry&
    Instance();

    template <StaticLoaderProvider Provider>
    void
    Register() {
        LoaderEntry entry{&Provider::DeriveCaps, &Provider::Open};
        if constexpr (requires {
                          &Provider::PlanPacked;
                          &Provider::FinishPacked;
                      }) {
            entry.plan_packed = &Provider::PlanPacked;
            entry.finish_packed = &Provider::FinishPacked;
        }
        if constexpr (requires(storage::FileSource & source,
                               const storage::LoadOptions& options) {
                          Provider::OpenAsync(source, options);
                      }) {
            entry.open_async = [](storage::FileSource& source,
                                  const storage::LoadOptions& options) {
                return Provider::OpenAsync(source, options);
            };
        }
        RegisterEntry(Provider::kFamily, entry);
    }

    // An empty entry means unknown family; callers decide how to report it.
    LoaderEntry
    Lookup(const IndexFamily& family) const;

 private:
    LoaderRegistry() = default;

    void
    RegisterEntry(std::string_view family, LoaderEntry entry);
};

// Family-specific build knobs, opaque to the builder registry.
//
// `Config` is `nlohmann::json` (`common/Types.h:673`). This is deliberately a
// bag at the builder registry boundary only: the registry finds the factory;
// each factory immediately parses its own typed parameters.
using BuildParams = Config;

// Input-typed builder registry. Family factories parse their own configuration;
// the registry does not interpret it or erase the complete input shape. The
// scalar hybrid family registers a build-time selector rather than a reader.
// Load-resource estimates are separate free functions, not mutable registry
// state.
template <typename Input>
class BuilderRegistry {
 public:
    using Factory = std::function<std::unique_ptr<IArtifactBuilder<Input>>(
        const BuildParams&)>;

    static BuilderRegistry&
    Instance();

    void
    Register(IndexFamily family, Factory factory);

    // Null when the family is unknown or does not build this input shape.
    std::unique_ptr<IArtifactBuilder<Input>>
    Create(const IndexFamily& family, const BuildParams& params) const;

 private:
    BuilderRegistry() = default;
};

// Pre-load resource estimation is implemented by the free functions in
// index/LoadResource.h; it is not registry or reader state.

}  // namespace milvus::index
