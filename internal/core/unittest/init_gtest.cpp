// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "cachinglayer/Manager.h"
#include "common/common_type_c.h"
#include "exec/expression/function/init_c.h"
#include "folly/init/Init.h"
#include "milvus-storage/filesystem/fs.h"
#include "segcore/arrow_fs_c.h"
#include "pb/cgo_msg.pb.h"
#include "pb/clustering.pb.h"
#include "pb/common.pb.h"
#include "pb/index_cgo_msg.pb.h"
#include "pb/plan.pb.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "storage/MmapManager.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "storage/Util.h"
#include "index/Meta.h"
#include "test_utils/Constants.h"

std::string TestLocalPath;
std::string TestRemotePath;
std::string TestMmapPath;

int
main(int argc, char** argv) {
    // Diagnostic branch only: isolate the CI crash and preserve its stack.
    if (std::filesystem::path(argv[0]).filename() != "all_tests") {
        return 0;
    }
    bool listing = false;
    for (int i = 1; i < argc; ++i) {
        listing |= std::string(argv[i]).starts_with("--gtest_list_tests");
    }
    if (!listing && std::getenv("PR55513_GDB_CHILD") == nullptr) {
        setenv("PR55513_GDB_CHILD", "1", 1);
        std::vector<std::string> arguments{
            "gdb", "--batch", "--return-child-result",
            "-ex", "set pagination off",
            "-ex", "set print thread-events off",
            "-ex", "run",
            "-ex", "thread apply all bt 25", "--args"};
        for (int i = 0; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        arguments.emplace_back("--gtest_filter=AsyncIndexEntryReaderTest.CancelledMetadataAdmissionPreservesErrorCode");
        arguments.emplace_back("--gtest_repeat=1000");
        arguments.emplace_back("--gtest_break_on_failure");
        std::vector<char*> command;
        for (auto& argument : arguments) {
            command.push_back(argument.data());
        }
        command.push_back(nullptr);
        execvp(command[0], command.data());
        std::perror("Cannot launch GDB for PR #55513 reproduction");
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    folly::Init follyInit(&argc, &argv, false);

    // Force eager initialization of all protobuf descriptors.
    // Protobuf 5.x uses lazy descriptor initialization via absl::call_once.
    // Under ASAN, this lazy init can intermittently fail with
    // "Check failed: file != nullptr" in AssignDescriptorsImpl.
    // Initializing descriptors here (single-threaded, before any tests)
    // eliminates the flaky failure.
    ::milvus::proto::common::SegmentStats::descriptor();
    ::milvus::proto::schema::CollectionSchema::descriptor();
    ::milvus::proto::plan::PlanNode::descriptor();
    ::milvus::proto::segcore::RetrieveResults::descriptor();
    ::milvus::proto::indexcgo::TypeParams::descriptor();
    ::milvus::proto::cgo::SerializedIndexFileInfo::descriptor();
    ::milvus::proto::clustering::StorageConfig::descriptor();

    // Determine the base directory for test output.
    // Priority: MILVUS_TEST_ROOT_DIR env var > compile-time MILVUS_CPPUT_OUTPUT_DIR
    std::string base_dir;
    if (auto* env = std::getenv("MILVUS_TEST_ROOT_DIR")) {
        base_dir = env;
    } else {
        base_dir = MILVUS_CPPUT_OUTPUT_DIR;
    }

    // Compute shard-aware test paths to avoid conflicts in parallel test runs
    int shard_index = 0;
    int total_shards = 1;
    if (auto* env = std::getenv("GTEST_SHARD_INDEX")) {
        shard_index = std::atoi(env);
    }
    if (auto* env = std::getenv("GTEST_TOTAL_SHARDS")) {
        total_shards = std::atoi(env);
    }
    std::srand(std::time(nullptr));
    int random = std::rand();
    // Ensure random % total_shards == shard_index so different shards use different paths
    random = random - (random % total_shards) + shard_index;
    TestLocalPath =
        base_dir + "/test_" + std::to_string(random) + "/local_data/";
    TestRemotePath =
        base_dir + "/test_" + std::to_string(random) + "/remote_data/";
    TestMmapPath = base_dir + "/test_" + std::to_string(random) + "/mmap_data/";

    // Ensure test directories exist before any Init calls
    std::filesystem::create_directories(TestLocalPath);
    std::filesystem::create_directories(TestRemotePath);
    std::filesystem::create_directories(TestMmapPath);

    // Initialize expression function factory (registers aggregate functions like count, sum, min, max)
    InitExecExpressionFunctionFactory();

    // Test-suite master switch: MILVUS_USE_ARROW_FS_CHUNK_MANAGER=1 runs
    // every test that obtains a chunk manager via CreateChunkManager (the
    // remote chunk manager singleton below, DataGen, indexbuilder, ...) on
    // the ArrowFileSystem backed IO path instead of the legacy one.
    if (auto* env = std::getenv("MILVUS_USE_ARROW_FS_CHUNK_MANAGER")) {
        std::string value(env);
        milvus::storage::SetUseArrowFileSystemChunkManager(value == "1" ||
                                                           value == "true");
    }

    milvus::storage::LocalChunkManagerSingleton::GetInstance().Init(
        TestLocalPath);
    milvus::storage::StorageConfig local_config;
    local_config.storage_type = "local";
    local_config.root_path = TestRemotePath;
    milvus::storage::RemoteChunkManagerSingleton::GetInstance().Init(
        local_config);
    milvus::storage::MmapManager::GetInstance().Init(
        {"willneed",
         TestMmapPath,
         uint64_t{2} * 1024 * 1024 * 1024,
         uint64_t{4} * 1024 * 1024,
         false});

    CStorageConfig arrow_fs_config = {};
    arrow_fs_config.root_path = TestLocalPath.c_str();
    arrow_fs_config.storage_type = "local";
    auto c_status = InitArrowFileSystem(arrow_fs_config);
    if (c_status.error_code != 0) {
        throw std::runtime_error("Failed to init arrow filesystem");
    }

    static const int64_t mb = 1024 * 1024;

    milvus::cachinglayer::Manager::ConfigureTieredStorage(
        {CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable,
         CacheWarmupPolicy::CacheWarmupPolicy_Disable},
        {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb},
        true,
        true,
        {10, true, 30},
        std::chrono::milliseconds(0),
        std::chrono::milliseconds(-1));

    return RUN_ALL_TESTS();
}
