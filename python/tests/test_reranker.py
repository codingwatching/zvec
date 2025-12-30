# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from unittest.mock import patch
import pytest
import math

from zvec import RrfReRanker, WeightedReRanker, Doc, MetricType


# ----------------------------
# RrfRanker Test Case
# ----------------------------
class TestRrfReRanker:
    def test_init(self):
        reranker = RrfReRanker(
            query="test", topn=5, rerank_field="content", rank_constant=100
        )
        assert reranker.query == "test"
        assert reranker.topn == 5
        assert reranker.rerank_field == "content"
        assert reranker.rank_constant == 100

    def test_rrf_score(self):
        reranker = RrfReRanker(query="test", rank_constant=60)
        # 根据公式 1.0 / (k + rank + 1)，其中k=60
        assert reranker._rrf_score(0) == 1.0 / (60 + 0 + 1)
        assert reranker._rrf_score(1) == 1.0 / (60 + 1 + 1)
        assert reranker._rrf_score(10) == 1.0 / (60 + 10 + 1)

    def test_rerank(self):
        reranker = RrfReRanker(query="test", topn=3)

        doc1 = Doc(id="1", score=0.8)
        doc2 = Doc(id="2", score=0.7)
        doc3 = Doc(id="3", score=0.9)
        doc4 = Doc(id="4", score=0.6)

        query_results = {"vector1": [doc1, doc2, doc3], "vector2": [doc3, doc1, doc4]}

        results = reranker.rerank(query_results)

        assert len(results) <= reranker.topn

        for doc in results:
            assert hasattr(doc, "score")

        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True)


# ----------------------------
# WeightedRanker Test Case
# ----------------------------
class TestWeightedReRanker:
    def test_init(self):
        weights = {"vector1": 0.7, "vector2": 0.3}
        reranker = WeightedReRanker(
            query="test",
            topn=5,
            rerank_field="content",
            metric=MetricType.L2,
            weights=weights,
        )
        assert reranker.query == "test"
        assert reranker.topn == 5
        assert reranker.rerank_field == "content"
        assert reranker.metric == MetricType.L2
        assert reranker.weights == weights

    def test_normalize_score(self):
        reranker = WeightedReRanker(query="test")

        score = reranker._normalize_score(1.0, MetricType.L2)
        expected = 1.0 - 2 * math.atan(1.0) / math.pi
        assert score == expected

        score = reranker._normalize_score(1.0, MetricType.IP)
        expected = 0.5 + math.atan(1.0) / math.pi
        assert score == expected

        score = reranker._normalize_score(1.0, MetricType.COSINE)
        expected = 1.0 - 1.0 / 2.0
        assert score == expected

        with pytest.raises(ValueError, match="Unsupported metric type"):
            reranker._normalize_score(1.0, "unsupported_metric")

    def test_rerank(self):
        weights = {"vector1": 0.7, "vector2": 0.3}
        reranker = WeightedReRanker(
            query="test", topn=3, weights=weights, metric=MetricType.L2
        )

        doc1 = Doc(id="1", score=0.8)
        doc2 = Doc(id="2", score=0.7)
        doc3 = Doc(id="3", score=0.9)

        query_results = {"vector1": [doc1, doc2], "vector2": [doc2, doc3]}

        results = reranker.rerank(query_results)

        assert len(results) <= reranker.topn

        for doc in results:
            assert hasattr(doc, "score")

        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True)


# # ----------------------------
# # QwenReRanker Test Case
# # ----------------------------
# class TestQwenReRanker:
#     def test_init_without_query(self):
#         with pytest.raises(ValueError):
#             QwenReRanker()
#
#     def test_init_without_api_key(self):
#         with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
#             with pytest.raises(ValueError, match="DashScope API key is required"):
#                 QwenReRanker(query="test")
#
#     @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
#     def test_init_with_env_api_key(self):
#         reranker = QwenReRanker(query="test")
#         assert reranker.query == "test"
#         assert reranker._api_key == "test_key"
#
#     def test_model_property(self):
#         reranker = QwenReRanker(query="test", api_key="test_key")
#         assert reranker.model == "gte-rerank-v2"
#
#         reranker = QwenReRanker(query="test", model="custom-model", api_key="test_key")
#         assert reranker.model == "custom-model"
#
#     def test_rerank_empty_results(self):
#         reranker = QwenReRanker(query="test", api_key="test_key")
#         results = reranker.rerank({})
#         assert results == []
#
#     def test_rerank_no_documents(self):
#         reranker = QwenReRanker(query="test", api_key="test_key")
#         query_results = {"vector1": [Doc(id="1")]}
#         with pytest.raises(ValueError, match="No documents to rerank"):
#             reranker.rerank(query_results)
#
#     @pytest.mark.skip(reason="Qwen ReRanker is not available in CI")
#     def test_rerank_success(self):
#         reranker = QwenReRanker(
#             topn=3,
#             query="test",
#             api_key="*",
#             rerank_field="content",
#         )
#         query_results = {
#             "vector1": [
#                 Doc(id="1", fields={"content": "This is a test document."}),
#                 Doc(id="2", fields={"content": "Another test document."}),
#                 Doc(id="3", fields={"content": "Yet another test document."}),
#                 Doc(id="4", fields={"content": "One more test document."}),
#             ],
#             "vector2": [
#                 Doc(id="5", fields={"content": "This is a test document2."}),
#                 Doc(id="6", fields={"content": "Another test document2."}),
#                 Doc(id="7", fields={"content": "Yet another test document2."}),
#                 Doc(id="8", fields={"content": "One more test document2."}),
#             ],
#         }
#         results = reranker.rerank(query_results)
#         assert len(results) == 3
