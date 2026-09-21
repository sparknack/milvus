// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#pragma once
#include "common/Types.h"
#include "index/KnowherePoCAdapterIO.h"
#include "pb/schema.pb.h"
#include <google/protobuf/util/message_differencer.h>
#include <climits>

namespace milvus::index::json_poc_io {
using poc_io::CheckEqual;
using poc_io::CheckSubset;
using poc_io::ReadBitmap;
using poc_io::WriteBitmap;
inline void
CheckSchema(poc_io::Reader& in, const proto::schema::FieldSchema& expected) {
    proto::schema::FieldSchema actual;
    const auto bytes = in.Bytes();
    poc_io::Check(bytes.size() <= INT_MAX &&
                      actual.ParseFromArray(bytes.data(), bytes.size()),
                  "invalid JSON field schema");
    poc_io::Check(
        google::protobuf::util::MessageDifferencer::Equals(actual, expected),
        "JSON field schema mismatch");
}
template <typename Core>
TargetBitmap
PostingRows(const Core& core, bool single_value) {
    TargetBitmap rows(core.Count());
    uint64_t frequency = 0;
    for (size_t term = 0; term < core.TermCount(); ++term) {
        core.DecodeInto(term, rows);
        frequency += core.DocFreq(term);
    }
    if (single_value)
        poc_io::Check(frequency == rows.count(),
                      "JSON scalar row has multiple terms");
    return rows;
}
}  // namespace milvus::index::json_poc_io
