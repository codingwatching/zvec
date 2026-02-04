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

#include "hnsw_rabitq_searcher.h"
#include <ailego/container/vector.h>
#include <gtest/gtest.h>
#include "core/framework/index_framework.h"
#include "hnsw_rabitq_builder.h"

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 128;

class HnswRabitqSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string HnswRabitqSearcherTest::_dir("hnswRabitqSearcherTest");
shared_ptr<IndexMeta> HnswRabitqSearcherTest::_index_meta_ptr;

void HnswRabitqSearcherTest::SetUp(void) {
  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswRabitqSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(HnswRabitqSearcherTest, TestBasicSearch) {
  // Build index
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params build_params;
  build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);
  build_params.set("proxima.hnsw.rabitq.ex_bits", 2UL);
  build_params.set("proxima.hnsw.builder.ef_construction", 200U);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestBasicSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Load and search with searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw.searcher.ef", 100U);

  ASSERT_EQ(0, searcher->init(*_index_meta_ptr, search_params));

  auto storage = IndexFactory::CreateStorage("FileStorage");
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, searcher->open(storage));

  // Search
  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto context = searcher->create_context();
  context->set_topk(10);

  ASSERT_EQ(0, searcher->search(query.data(), query_meta, context));

  auto &result = context->result();
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  // Verify first result is closest
  ASSERT_EQ(0UL, result[0].key());

  ASSERT_EQ(0, searcher->close());
}

TEST_F(HnswRabitqSearcherTest, TestMultipleQueries) {
  // Build index
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params build_params;
  build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  string path = _dir + "/TestMultipleQueries";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Load searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw.searcher.ef", 100U);

  ASSERT_EQ(0, searcher->init(*_index_meta_ptr, search_params));

  auto storage = IndexFactory::CreateStorage("FileStorage");
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, searcher->open(storage));

  // Multiple searches
  size_t num_queries = 10;
  for (size_t q = 0; q < num_queries; q++) {
    NumericalVector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = static_cast<float>(q * dim + j) / 1000.0f;
    }

    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    auto context = searcher->create_context();
    context->set_topk(5);

    ASSERT_EQ(0, searcher->search(query.data(), query_meta, context));

    auto &result = context->result();
    ASSERT_GT(result.size(), 0UL);
    ASSERT_LE(result.size(), 5UL);
  }

  ASSERT_EQ(0, searcher->close());
}

TEST_F(HnswRabitqSearcherTest, TestDifferentTopK) {
  // Build index
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 500.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params build_params;
  build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  string path = _dir + "/TestDifferentTopK";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Load searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  ASSERT_EQ(0, searcher->init(*_index_meta_ptr, search_params));

  auto storage = IndexFactory::CreateStorage("FileStorage");
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, searcher->open(storage));

  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = static_cast<float>(j) / 500.0f;
  }

  // Test different topk values
  std::vector<uint32_t> topk_values = {1, 5, 10, 20, 50};

  for (auto topk : topk_values) {
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
    auto context = searcher->create_context();
    context->set_topk(topk);

    ASSERT_EQ(0, searcher->search(query.data(), query_meta, context));

    auto &result = context->result();
    ASSERT_GT(result.size(), 0UL);
    ASSERT_LE(result.size(), topk);
  }

  ASSERT_EQ(0, searcher->close());
}

TEST_F(HnswRabitqSearcherTest, TestBruteForceSearch) {
  // Build index
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 300UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 300.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params build_params;
  build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  string path = _dir + "/TestBruteForceSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Load searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  ASSERT_EQ(0, searcher->init(*_index_meta_ptr, search_params));

  auto storage = IndexFactory::CreateStorage("FileStorage");
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, searcher->open(storage));

  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = static_cast<float>(j) / 300.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto context = searcher->create_context();
  context->set_topk(10);

  ASSERT_EQ(0, searcher->search_bf(query.data(), query_meta, context));

  auto &result = context->result();
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  ASSERT_EQ(0, searcher->close());
}

TEST_F(HnswRabitqSearcherTest, TestReadOnlyMode) {
  // Build index
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 500UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 500.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params build_params;
  build_params.set("proxima.hnsw.rabitq.num_clusters", 16UL);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, build_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  string path = _dir + "/TestReadOnlyMode";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Open in read-only mode
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  ASSERT_EQ(0, searcher->init(*_index_meta_ptr, search_params));

  auto storage = IndexFactory::CreateStorage("FileStorage");
  ASSERT_EQ(0, storage->open(path, true));  // true = read-only
  ASSERT_EQ(0, searcher->open(storage));

  // Verify can search
  NumericalVector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = static_cast<float>(j) / 500.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto context = searcher->create_context();
  context->set_topk(10);

  ASSERT_EQ(0, searcher->search(query.data(), query_meta, context));

  auto &result = context->result();
  ASSERT_GT(result.size(), 0UL);

  ASSERT_EQ(0, searcher->close());
}

}  // namespace core
}  // namespace zvec
