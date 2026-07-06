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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "segcore/storagev2translator/ManifestGroupTranslatorV2.h"

#include "arrow/api.h"
#include "cachinglayer/Utils.h"
#include "common/Consts.h"
#include "common/FieldMeta.h"
#include "common/GroupChunk.h"
#include "common/Schema.h"
#include "folly/ScopeGuard.h"
#include "gtest/gtest.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/async_read_options.h"
#include "pb/common.pb.h"
#include "segcore/storagev2translator/AsyncLoadPipeline.h"
#include "segcore/storagev2translator/ManifestGroupTranslator.h"
#include "segcore/storagev2translator/StorageV2Config.h"
#include "storage/EntryStreamUtils.h"
#include "test_utils/Constants.h"
#include "test_utils/DataGen.h"
#include "test_utils/ManifestTestUtil.h"

using namespace milvus;
using namespace milvus::segcore;
using namespace milvus::segcore::storagev2translator;

namespace {

class MetadataFieldChunkReader final : public milvus_storage::api::ChunkReader {
 public:
    explicit MetadataFieldChunkReader(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(std::move(batch)) {
    }

    size_t
    total_number_of_chunks() const override {
        return 1;
    }

    arrow::Result<std::vector<int64_t>>
    get_chunk_indices(const std::vector<int64_t>& row_indices) override {
        return row_indices;
    }

    arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    get_chunk(int64_t) override {
        return batch_;
    }

    arrow::Result<std::vector<std::shared_ptr<arrow::RecordBatch>>>
    get_chunks(const std::vector<int64_t>& chunk_indices,
               size_t /*parallelism*/) override {
        return std::vector<std::shared_ptr<arrow::RecordBatch>>(
            chunk_indices.size(), batch_);
    }

    folly::SemiFuture<arrow::Result<milvus_storage::api::RecordBatchVector>>
    get_chunks_async(
        const std::vector<int64_t>& chunk_indices,
        const milvus_storage::api::AsyncReadOptions& /*options*/) override {
        return folly::makeSemiFuture(
            arrow::Result<milvus_storage::api::RecordBatchVector>(
                milvus_storage::api::RecordBatchVector(chunk_indices.size(),
                                                       batch_)));
    }

    arrow::Result<std::vector<uint64_t>>
    get_chunk_size() override {
        return std::vector<uint64_t>{64};
    }

    arrow::Result<std::vector<uint64_t>>
    get_chunk_rows() override {
        return std::vector<uint64_t>{static_cast<uint64_t>(batch_->num_rows())};
    }

 private:
    std::shared_ptr<arrow::RecordBatch> batch_;
};

std::shared_ptr<arrow::RecordBatch>
MakeMetadataFieldRecordBatch(FieldId field_id) {
    arrow::Int64Builder builder;
    AssertInfo(builder.Append(10).ok(), "append failed");
    AssertInfo(builder.Append(20).ok(), "append failed");
    std::shared_ptr<arrow::Array> values;
    AssertInfo(builder.Finish(&values).ok(), "finish failed");

    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append(milvus_storage::ARROW_FIELD_ID_KEY,
                     std::to_string(field_id.get()));
    auto schema = arrow::schema({arrow::field(
        "json_path_int64", arrow::int64(), true, std::move(metadata))});
    return arrow::RecordBatch::Make(schema, values->length(), {values});
}

}  // namespace

TEST(StorageV2ConfigTest, AsyncLoadDefaultsToEnabledAndCanBeToggled) {
    const bool old_enabled = StorageV2AsyncLoadEnabled();
    auto restore_enabled = folly::makeGuard(
        [old_enabled]() { SetStorageV2AsyncLoadEnabled(old_enabled); });

    EXPECT_TRUE(StorageV2AsyncLoadEnabled());

    SetStorageV2AsyncLoadEnabled(false);
    EXPECT_FALSE(StorageV2AsyncLoadEnabled());

    SetStorageV2AsyncLoadEnabled(true);
    EXPECT_TRUE(StorageV2AsyncLoadEnabled());
}

TEST(ManifestGroupTranslatorV2Test, ResolvesFieldIdFromArrowMetadata) {
    constexpr int64_t kSegmentId = 77;
    auto field_id = FieldId(START_JSON_STATS_FIELD_ID);
    auto reader = std::make_shared<MetadataFieldChunkReader>(
        MakeMetadataFieldRecordBatch(field_id));

    std::unordered_map<FieldId, FieldMeta> field_metas;
    field_metas.emplace(field_id,
                        FieldMeta(FieldName("json_path_int64"),
                                  field_id,
                                  DataType::INT64,
                                  true,
                                  std::nullopt));

    ManifestGroupTranslatorV2 translator(
        kSegmentId,
        GroupChunkType::JSON_KEY_STATS,
        /*column_group_index=*/0,
        std::move(reader),
        field_metas,
        /*use_mmap=*/false,
        /*mmap_populate=*/false,
        /*mmap_dir_path=*/"",
        /*num_fields=*/1,
        milvus::proto::common::LoadPriority::LOW,
        /*eager_load=*/true,
        /*warmup_policy=*/"",
        /*cache_key_suffix=*/std::to_string(field_id.get()));

    auto cells = translator.get_cells(nullptr, {0});

    ASSERT_EQ(cells.size(), 1);
    ASSERT_NE(cells[0].second, nullptr);
    EXPECT_TRUE(cells[0].second->HasChunk(field_id));
    EXPECT_EQ(cells[0].second->RowNums(), 2);
}

class ManifestGroupTranslatorV2ParityTest
    : public ::testing::TestWithParam<bool> {
 protected:
    void
    SetUp() override {
        schema_ = CreateTestSchema();
        base_path_ = "manifest_translator_v2_parity_test";
        v1_mmap_dir_ = TestMmapPath + "/manifest_translator_v2_v1_mmap/";
        v2_mmap_dir_ = TestMmapPath + "/manifest_translator_v2_v2_mmap/";
        std::filesystem::create_directories(v1_mmap_dir_);
        std::filesystem::create_directories(v2_mmap_dir_);

        test_data_ = std::make_unique<milvus::test::V3SegmentTestData>(
            schema_, n_batch_, per_batch_, dim_, TestLocalPath, base_path_);
    }

    ~ManifestGroupTranslatorV2ParityTest() override {
        std::filesystem::remove_all(v1_mmap_dir_);
        std::filesystem::remove_all(v2_mmap_dir_);
    }

    std::unique_ptr<ManifestGroupTranslator>
    MakeV1Translator(int64_t cg_index, bool use_mmap) {
        auto chunk_reader = test_data_->CreateChunkReader(cg_index);
        auto field_metas = test_data_->GetFieldMetas(cg_index);
        return std::make_unique<ManifestGroupTranslator>(
            segment_id_,
            GroupChunkType::DEFAULT,
            cg_index,
            std::move(chunk_reader),
            field_metas,
            use_mmap,
            /*mmap_populate=*/true,
            v1_mmap_dir_,
            field_metas.size(),
            milvus::proto::common::LoadPriority::LOW,
            /*eager_load=*/true,
            /*warmup_policy=*/"");
    }

    std::unique_ptr<ManifestGroupTranslatorV2>
    MakeV2Translator(int64_t cg_index, bool use_mmap) {
        auto chunk_reader = test_data_->CreateChunkReader(cg_index);
        auto field_metas = test_data_->GetFieldMetas(cg_index);
        return std::make_unique<ManifestGroupTranslatorV2>(
            segment_id_,
            GroupChunkType::DEFAULT,
            cg_index,
            std::move(chunk_reader),
            field_metas,
            use_mmap,
            /*mmap_populate=*/true,
            v2_mmap_dir_,
            field_metas.size(),
            milvus::proto::common::LoadPriority::LOW,
            /*eager_load=*/true,
            /*warmup_policy=*/"");
    }

    SchemaPtr schema_;
    std::unique_ptr<milvus::test::V3SegmentTestData> test_data_;
    std::string base_path_;
    std::string v1_mmap_dir_;
    std::string v2_mmap_dir_;
    int64_t segment_id_ = 0;
    int64_t n_batch_ = 8;
    int64_t per_batch_ = 1000;
    int64_t dim_ = 128;
};

TEST_P(ManifestGroupTranslatorV2ParityTest, MatchesV1ForStorageV3Cells) {
    auto use_mmap = GetParam();
    auto v1 = MakeV1Translator(/*cg_index=*/0, use_mmap);
    auto v2 = MakeV2Translator(/*cg_index=*/0, use_mmap);

    ASSERT_EQ(v2->num_cells(), v1->num_cells());
    ASSERT_EQ(v2->key(), v1->key());

    for (size_t i = 0; i < v1->num_cells(); ++i) {
        EXPECT_EQ(v2->cell_id_of(i), v1->cell_id_of(i));
        EXPECT_EQ(v2->estimated_byte_size_of_cell(i),
                  v1->estimated_byte_size_of_cell(i));
    }

    std::vector<cachinglayer::cid_t> cids;
    for (size_t i = v1->num_cells(); i > 0; --i) {
        cids.push_back(static_cast<cachinglayer::cid_t>(i - 1));
    }

    auto v1_cells = v1->get_cells(nullptr, cids);
    auto v2_cells = v2->get_cells(nullptr, cids);
    ASSERT_EQ(v2_cells.size(), v1_cells.size());
    for (size_t i = 0; i < cids.size(); ++i) {
        EXPECT_EQ(v2_cells[i].first, v1_cells[i].first);
        EXPECT_NE(v2_cells[i].second, nullptr);
    }
}

TEST_P(ManifestGroupTranslatorV2ParityTest,
       BuildsStorageV3LoadUnitsFromRequestedCells) {
    auto use_mmap = GetParam();
    auto v2 = MakeV2Translator(/*cg_index=*/0, use_mmap);
    ASSERT_GE(v2->num_cells(), 1);

    auto* meta = static_cast<GroupCTMeta*>(v2->meta());
    std::vector<cachinglayer::cid_t> cids;
    for (size_t i = v2->num_cells(); i > 0; --i) {
        cids.push_back(static_cast<cachinglayer::cid_t>(i - 1));
    }

    auto units = BuildStorageV3LoadUnitsForCells(
        *meta, cids, [](int64_t memory_size) { return memory_size + 17; });

    ASSERT_EQ(units.size(), cids.size());
    for (size_t i = 0; i < cids.size(); ++i) {
        auto cid = cids[i];
        auto [start, end] = meta->get_row_group_range(cid);
        EXPECT_EQ(units[i].cid, cid);
        EXPECT_EQ(units[i].rg_offset, static_cast<int64_t>(start));
        EXPECT_EQ(units[i].rg_count, static_cast<int64_t>(end - start));
        EXPECT_EQ(units[i].memory_size, meta->chunk_memory_size_[cid]);
        EXPECT_EQ(units[i].loading_overhead_size,
                  meta->chunk_memory_size_[cid] + 17);
    }
}

TEST_P(ManifestGroupTranslatorV2ParityTest, LoadsCellsWithTinyAdmissionBudget) {
    auto& budget =
        milvus::storage::TransientMemoryBudget::GetLoadTransientBudget();
    auto old_capacity = budget.CapacityBytes();
    auto restore_budget = folly::makeGuard(
        [&budget, old_capacity]() { budget.SetCapacityBytes(old_capacity); });
    budget.SetCapacityBytes(1);

    auto use_mmap = GetParam();
    auto v2 = MakeV2Translator(/*cg_index=*/0, use_mmap);
    ASSERT_GT(v2->num_cells(), 0);

    std::vector<cachinglayer::cid_t> cids;
    for (size_t i = 0; i < v2->num_cells(); ++i) {
        cids.push_back(static_cast<cachinglayer::cid_t>(i));
    }

    auto cells = v2->get_cells(nullptr, cids);

    ASSERT_EQ(cells.size(), cids.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        EXPECT_EQ(cells[i].first, cids[i]);
        EXPECT_NE(cells[i].second, nullptr);
    }
}

TEST_P(ManifestGroupTranslatorV2ParityTest, ReturnsEmptyForEmptyCellRequest) {
    auto use_mmap = GetParam();
    auto v2 = MakeV2Translator(/*cg_index=*/0, use_mmap);

    auto cells = v2->get_cells(nullptr, {});

    EXPECT_TRUE(cells.empty());
}

INSTANTIATE_TEST_SUITE_P(ManifestGroupTranslatorV2,
                         ManifestGroupTranslatorV2ParityTest,
                         testing::Bool());
