// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hnsw_rabitq_streamer.h"
#include <memory>
#include <gtest/gtest.h>
#include "zvec/ailego/container/params.h"
#include "zvec/core/framework/index_framework.h"
#include "zvec/core/framework/index_streamer.h"
#include "hnsw_rabitq_streamer.h"
#include "rabitq_converter.h"
#include "rabitq_reformer.h"
#include "test_index_provider.h"

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 128;

class HnswRabitqStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswRabitqStreamerTest::dir_("hnswRabitqStreamerTest");
shared_ptr<IndexMeta> HnswRabitqStreamerTest::index_meta_ptr_;

void HnswRabitqStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswRabitqStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswRabitqStreamerTest, TestBuildAndSearch) {
  auto holder = make_shared<FloatIndexProvider>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  RabitqConverter converter;
  converter.init(*index_meta_ptr_, ailego::Params());
  ASSERT_EQ(converter.train(holder), 0);
  std::shared_ptr<RabitqReformer> reformer;
  ASSERT_EQ(converter.to_reformer(&reformer), 0);
  IndexStreamer::Pointer streamer =
      std::make_shared<HnswRabitqStreamer>(holder, reformer);


  ailego::Params params;
  params.set("proxima.hnsw.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.streamer.scaling_factor", 5U);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/AddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  for (auto it = holder->create_iterator(); it->is_valid(); it->next()) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    ASSERT_EQ(0,
              streamer->add_impl(it->key(), it->data(), query_meta, context));
  }
  streamer->flush(0UL);

  // Perform search verification
  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  context->set_topk(10);
  ASSERT_EQ(0, streamer->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  // reopen and load reformer from storage
  ASSERT_EQ(0, streamer->close());
  IndexStreamer::Pointer new_streamer =
      std::make_shared<HnswRabitqStreamer>(holder);
  ASSERT_EQ(0, new_streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, new_streamer->open(storage));
}

// TEST_F(HnswRabitqStreamerTest, TestAddVector) {
//   // Build initial index
//   IndexBuilder::Pointer builder =
//       IndexFactory::CreateBuilder("HnswRabitqBuilder");
//   ASSERT_NE(builder, nullptr);

//   auto holder =
//       make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
//   size_t doc_cnt = 500UL;
//   for (size_t i = 0; i < doc_cnt; i++) {
//     NumericalVector<float> vec(dim);
//     for (size_t j = 0; j < dim; ++j) {
//       vec[j] = static_cast<float>(i * dim + j) / 500.0f;
//     }
//     ASSERT_TRUE(holder->emplace(i, vec));
//   }

//   ailego::Params build_params;
//   build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

//   ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
//   ASSERT_EQ(0, builder->train(holder));
//   ASSERT_EQ(0, builder->build(holder));

//   auto dumper = IndexFactory::CreateDumper("FileDumper");
//   string path = _dir + "/TestAddVector";
//   ASSERT_EQ(0, dumper->create(path));
//   ASSERT_EQ(0, builder->dump(dumper));
//   ASSERT_EQ(0, dumper->close());

//   // Load and add vectors
//   IndexStreamer::Pointer streamer =
//       IndexFactory::CreateStreamer("HnswRabitqStreamer");
//   ASSERT_NE(streamer, nullptr);

//   ailego::Params stream_params;
//   ASSERT_EQ(0, streamer->init(*_index_meta_ptr, stream_params));

//   auto storage = IndexFactory::CreateStorage("FileStorage");
//   ASSERT_EQ(0, storage->open(path, false));
//   ASSERT_EQ(0, streamer->open(storage));

//   // Add new vectors
//   size_t new_doc_cnt = 100UL;
//   for (size_t i = doc_cnt; i < doc_cnt + new_doc_cnt; i++) {
//     NumericalVector<float> vec(dim);
//     for (size_t j = 0; j < dim; ++j) {
//       vec[j] = static_cast<float>(i * dim + j) / 500.0f;
//     }

//     IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
//     auto context = streamer->create_context();
//     ASSERT_EQ(0, streamer->add(i, vec.data(), query_meta, context));
//   }

//   // Verify added vectors can be searched
//   NumericalVector<float> query(dim);
//   for (size_t j = 0; j < dim; ++j) {
//     query[j] = static_cast<float>(doc_cnt * dim + j) / 500.0f;
//   }

//   IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
//   auto context = streamer->create_context();
//   context->set_topk(10);

//   ASSERT_EQ(0, streamer->search(query.data(), query_meta, context));

//   auto &result = context->result();
//   ASSERT_GT(result.size(), 0UL);

//   ASSERT_EQ(0, streamer->close());
// }

// TEST_F(HnswRabitqStreamerTest, TestSearchWithDifferentEf) {
//   // Build index
//   IndexBuilder::Pointer builder =
//       IndexFactory::CreateBuilder("HnswRabitqBuilder");
//   ASSERT_NE(builder, nullptr);

//   auto holder =
//       make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
//   size_t doc_cnt = 1000UL;
//   for (size_t i = 0; i < doc_cnt; i++) {
//     NumericalVector<float> vec(dim);
//     for (size_t j = 0; j < dim; ++j) {
//       vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
//     }
//     ASSERT_TRUE(holder->emplace(i, vec));
//   }

//   ailego::Params build_params;
//   build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

//   ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
//   ASSERT_EQ(0, builder->train(holder));
//   ASSERT_EQ(0, builder->build(holder));

//   auto dumper = IndexFactory::CreateDumper("FileDumper");
//   string path = _dir + "/TestSearchWithDifferentEf";
//   ASSERT_EQ(0, dumper->create(path));
//   ASSERT_EQ(0, builder->dump(dumper));
//   ASSERT_EQ(0, dumper->close());

//   // Test with different ef values
//   std::vector<uint32_t> ef_values = {50, 100, 200};

//   for (auto ef : ef_values) {
//     IndexStreamer::Pointer streamer =
//         IndexFactory::CreateStreamer("HnswRabitqStreamer");
//     ASSERT_NE(streamer, nullptr);

//     ailego::Params stream_params;
//     stream_params.set("proxima.hnsw.streamer.ef", ef);

//     ASSERT_EQ(0, streamer->init(*_index_meta_ptr, stream_params));

//     auto storage = IndexFactory::CreateStorage("FileStorage");
//     ASSERT_EQ(0, storage->open(path, true));
//     ASSERT_EQ(0, streamer->open(storage));

//     NumericalVector<float> query(dim);
//     for (size_t j = 0; j < dim; ++j) {
//       query[j] = static_cast<float>(j) / 1000.0f;
//     }

//     IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
//     auto context = streamer->create_context();
//     context->set_topk(10);

//     ASSERT_EQ(0, streamer->search(query.data(), query_meta, context));

//     auto &result = context->result();
//     ASSERT_GT(result.size(), 0UL);
//     ASSERT_LE(result.size(), 10UL);

//     ASSERT_EQ(0, streamer->close());
//   }
// }

// TEST_F(HnswRabitqStreamerTest, TestBruteForceSearch) {
//   // Build index
//   IndexBuilder::Pointer builder =
//       IndexFactory::CreateBuilder("HnswRabitqBuilder");
//   ASSERT_NE(builder, nullptr);

//   auto holder =
//       make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
//   size_t doc_cnt = 500UL;
//   for (size_t i = 0; i < doc_cnt; i++) {
//     NumericalVector<float> vec(dim);
//     for (size_t j = 0; j < dim; ++j) {
//       vec[j] = static_cast<float>(i * dim + j) / 500.0f;
//     }
//     ASSERT_TRUE(holder->emplace(i, vec));
//   }

//   ailego::Params build_params;
//   build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

//   ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
//   ASSERT_EQ(0, builder->train(holder));
//   ASSERT_EQ(0, builder->build(holder));

//   auto dumper = IndexFactory::CreateDumper("FileDumper");
//   string path = _dir + "/TestBruteForceSearch";
//   ASSERT_EQ(0, dumper->create(path));
//   ASSERT_EQ(0, builder->dump(dumper));
//   ASSERT_EQ(0, dumper->close());

//   // Load and brute force search
//   IndexStreamer::Pointer streamer =
//       IndexFactory::CreateStreamer("HnswRabitqStreamer");
//   ASSERT_NE(streamer, nullptr);

//   ailego::Params stream_params;
//   ASSERT_EQ(0, streamer->init(*_index_meta_ptr, stream_params));

//   auto storage = IndexFactory::CreateStorage("FileStorage");
//   ASSERT_EQ(0, storage->open(path, true));
//   ASSERT_EQ(0, streamer->open(storage));

//   NumericalVector<float> query(dim);
//   for (size_t j = 0; j < dim; ++j) {
//     query[j] = static_cast<float>(j) / 500.0f;
//   }

//   IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
//   auto context = streamer->create_context();
//   context->set_topk(10);

//   ASSERT_EQ(0, streamer->search_bf(query.data(), query_meta, context));

//   auto &result = context->result();
//   ASSERT_GT(result.size(), 0UL);
//   ASSERT_LE(result.size(), 10UL);

//   ASSERT_EQ(0, streamer->close());
// }

}  // namespace core
}  // namespace zvec
