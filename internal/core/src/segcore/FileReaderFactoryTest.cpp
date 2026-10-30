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

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/util/future.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <parquet/encryption/encryption.h>
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/metadata.h"
#include "segcore/memory_planner.h"

namespace milvus::segcore {
namespace {
using milvus_storage::ARROW_FIELD_ID_KEY;
using milvus_storage::FieldID;
using milvus_storage::FieldIDList;
using milvus_storage::GROUP_FIELD_ID_LIST_META_KEY;
using milvus_storage::GroupFieldIDList;
using milvus_storage::ROW_GROUP_META_KEY;
using milvus_storage::RowGroupMetadata;
using milvus_storage::RowGroupMetadataVector;
using milvus_storage::STORAGE_VERSION_KEY;

struct ReadCounters {
    std::atomic<int> footer_reads{0};
    std::atomic<bool> fail_read{false};
};

class CountingBufferReader : public arrow::io::RandomAccessFile {
 public:
    CountingBufferReader(std::shared_ptr<arrow::Buffer> data,
                         std::shared_ptr<ReadCounters> counters)
        : reader_(std::move(data)), counters_(std::move(counters)) {
    }

    arrow::Status
    Close() override {
        return reader_.Close();
    }
    bool
    closed() const override {
        return reader_.closed();
    }
    arrow::Result<int64_t>
    Tell() const override {
        return reader_.Tell();
    }
    arrow::Status
    Seek(int64_t position) override {
        return reader_.Seek(position);
    }
    arrow::Result<int64_t>
    GetSize() override {
        return reader_.GetSize();
    }
    arrow::Result<int64_t>
    Read(int64_t nbytes, void* out) override {
        return reader_.Read(nbytes, out);
    }
    arrow::Result<std::shared_ptr<arrow::Buffer>>
    Read(int64_t nbytes) override {
        return reader_.Read(nbytes);
    }

    arrow::Result<std::shared_ptr<arrow::Buffer>>
    ReadAt(int64_t position, int64_t nbytes) override {
        ARROW_RETURN_NOT_OK(BeforeRead(position, nbytes));
        return reader_.ReadAt(position, nbytes);
    }
    arrow::Result<int64_t>
    ReadAt(int64_t position, int64_t nbytes, void* out) override {
        ARROW_RETURN_NOT_OK(BeforeRead(position, nbytes));
        return reader_.ReadAt(position, nbytes, out);
    }
    arrow::Future<std::shared_ptr<arrow::Buffer>>
    ReadAsync(const arrow::io::IOContext&,
              int64_t position,
              int64_t nbytes) override {
        return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(
            ReadAt(position, nbytes));
    }

 private:
    arrow::Status
    BeforeRead(int64_t position, int64_t nbytes) {
        if (position + nbytes >= reader_.buffer()->size()) {
            ++counters_->footer_reads;
        }
        if (counters_->fail_read.exchange(false)) {
            return arrow::Status::IOError("injected read failure");
        }
        return arrow::Status::OK();
    }
    arrow::io::BufferReader reader_;
    std::shared_ptr<ReadCounters> counters_;
};

class CountingFileSystem : public arrow::fs::LocalFileSystem {
 public:
    using LocalFileSystem::OpenInputFile;
    arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
    OpenInputFile(const std::string& path) override {
        if (fail_open.exchange(false)) {
            return arrow::Status::IOError("injected open failure");
        }
        return std::make_shared<CountingBufferReader>(files.at(path), counters);
    }
    std::map<std::string, std::shared_ptr<arrow::Buffer>> files;
    std::shared_ptr<ReadCounters> counters = std::make_shared<ReadCounters>();
    std::atomic<bool> fail_open{false};
};

// Real packed Parquet bytes, with four distinguishable row groups.
arrow::Result<std::shared_ptr<arrow::Buffer>>
MakePackedFile(
    int64_t base = 0,
    std::shared_ptr<parquet::FileEncryptionProperties> encryption = nullptr) {
    arrow::Int64Builder values;
    for (int64_t i = 0; i < 16; ++i) {
        ARROW_RETURN_NOT_OK(values.Append(base + i));
    }
    ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
    RowGroupMetadataVector row_groups;
    for (int64_t i = 0; i < 4; ++i) {
        row_groups.Add(RowGroupMetadata(32, 4, i * 4));
    }
    GroupFieldIDList groups(
        std::vector<FieldIDList>{FieldIDList(std::vector<FieldID>{100})});
    auto schema = arrow::schema(
        {arrow::field(
            "value",
            arrow::int64(),
            false,
            arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))},
        arrow::key_value_metadata(
            {ROW_GROUP_META_KEY,
             STORAGE_VERSION_KEY,
             GROUP_FIELD_ID_LIST_META_KEY},
            {row_groups.Serialize(), "1.0.0", groups.Serialize()}));
    auto table = arrow::Table::Make(schema, {array});
    ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::BufferOutputStream::Create());
    parquet::WriterProperties::Builder properties;
    if (encryption) {
        properties.encryption(std::move(encryption));
    }
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(
        *table,
        arrow::default_memory_pool(),
        sink,
        4,
        properties.build(),
        parquet::ArrowWriterProperties::Builder().store_schema()->build()));
    return sink->Finish();
}

class FileReaderFactoryTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        auto data = MakePackedFile();
        ASSERT_TRUE(data.ok()) << data.status();
        fs->files["first"] = *data;
        data = MakePackedFile(100);
        ASSERT_TRUE(data.ok()) << data.status();
        fs->files["second"] = *data;
    }
    std::shared_ptr<CountingFileSystem> fs =
        std::make_shared<CountingFileSystem>();
};

TEST_F(FileReaderFactoryTest, ConcurrentBatchesReadOneFooterPerFile) {
    auto factory = MakeFileReaderFactory({"first", "second"}, fs);
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < 16; ++i) {
        tasks.emplace_back(std::async(std::launch::async, [&, i, ready] {
            ready.wait();
            auto result = factory(i % 2, i % 4, 1, 32);
            ASSERT_TRUE(result.ok()) << result.status();
            ASSERT_EQ(result->size(), 1);
            const auto& table = (*result)[0];
            EXPECT_EQ(table->num_rows(), 4);
            EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(
                          table->column(0)->chunk(0))
                          ->Value(0),
                      (i % 2) * 100 + (i % 4) * 4);
        }));
    }
    start.set_value();
    for (auto& task : tasks) task.get();
    EXPECT_EQ(fs->counters->footer_reads, 2);
    // Sequential batches continue using the same metadata, not just concurrent ones.
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(factory(0, i, 1, 32).ok());
    EXPECT_EQ(fs->counters->footer_reads, 2);
}

TEST_F(FileReaderFactoryTest, FailedFirstOpenIsNotCached) {
    auto factory = MakeFileReaderFactory({"first"}, fs);
    fs->fail_open = true;
    auto failed = factory(0, 0, 1, 32);
    ASSERT_FALSE(failed.ok());
    EXPECT_TRUE(failed.status().IsIOError()) << failed.status();
    EXPECT_EQ(fs->counters->footer_reads, 0);
    ASSERT_TRUE(factory(0, 0, 1, 32).ok());
    ASSERT_TRUE(factory(0, 1, 1, 32).ok());
    EXPECT_EQ(fs->counters->footer_reads, 1);
}

TEST_F(FileReaderFactoryTest, FailedFooterReadIsNotCached) {
    auto factory = MakeFileReaderFactory({"first"}, fs);
    fs->counters->fail_read = true;
    auto failed = factory(0, 0, 1, 32);
    ASSERT_FALSE(failed.ok());
    EXPECT_TRUE(failed.status().IsIOError()) << failed.status();
    ASSERT_TRUE(factory(0, 0, 1, 32).ok());
    EXPECT_EQ(fs->counters->footer_reads, 2);
}

TEST_F(FileReaderFactoryTest, DataFailureDoesNotPoisonCachedMetadata) {
    auto factory = MakeFileReaderFactory({"first"}, fs);
    ASSERT_TRUE(factory(0, 0, 1, 32).ok());
    fs->counters->fail_read = true;
    auto failed = factory(0, 1, 1, 32);
    ASSERT_FALSE(failed.ok());
    EXPECT_TRUE(failed.status().IsIOError()) << failed.status();
    ASSERT_TRUE(factory(0, 1, 1, 32).ok());
    EXPECT_EQ(fs->counters->footer_reads, 1);
}

TEST_F(FileReaderFactoryTest, NewFactoryHasItsOwnMetadataLifetime) {
    {
        auto factory = MakeFileReaderFactory({"first"}, fs);
        ASSERT_TRUE(factory(0, 0, 1, 32).ok());
    }
    auto replacement = MakePackedFile(1000);
    ASSERT_TRUE(replacement.ok()) << replacement.status();
    fs->files["first"] = *replacement;
    auto factory = MakeFileReaderFactory({"first"}, fs);
    auto result = factory(0, 0, 1, 32);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(
                  (*result)[0]->column(0)->chunk(0))
                  ->Value(0),
              1000);
    EXPECT_EQ(fs->counters->footer_reads, 2);
}

TEST_F(FileReaderFactoryTest, CorruptFooterRemainsAnError) {
    fs->files["broken"] = arrow::Buffer::FromString("not parquet");
    auto factory = MakeFileReaderFactory({"broken"}, fs);
    EXPECT_FALSE(factory(0, 0, 1, 32).ok());
    EXPECT_FALSE(factory(0, 0, 1, 32).ok());
}

}  // namespace
}  // namespace milvus::segcore
