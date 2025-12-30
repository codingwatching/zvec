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

import heapq
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from http import HTTPStatus
from typing import Optional

from ..model.doc import Doc
from ..tool import require_module
from ..typing import MetricType


class ReRanker(ABC):
    """Abstract base class for re-ranking search results.

    Re-rankers refine the output of one or more vector queries by applying
    a secondary scoring strategy. They are used in the ``query()`` method of
    ``Collection`` via the ``reranker`` parameter.

    Args:
        query (Optional[str], optional): Query text used for re-ranking.
            Required for LLM-based re-rankers. Defaults to None.
        topn (int, optional): Number of top documents to return after re-ranking.
            Defaults to 10.
        rerank_field (Optional[str], optional): Field name used as input for
            re-ranking (e.g., document title or body). Defaults to None.

    Note:
        Subclasses must implement the ``rerank()`` method.
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
    ):
        self._query = query
        self._topn = topn
        self._rerank_field = rerank_field

    @property
    def topn(self) -> int:
        """int: Number of top documents to return after re-ranking."""
        return self._topn

    @property
    def query(self) -> str:
        """str: Query text used for re-ranking."""
        return self._query

    @property
    def rerank_field(self) -> Optional[str]:
        """Optional[str]: Field name used as re-ranking input."""
        return self._rerank_field

    @abstractmethod
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-rank documents from one or more vector queries.

        Args:
            query_results (dict[str, list[Doc]]): Mapping from vector field name
                to list of retrieved documents (sorted by relevance).

        Returns:
            list[Doc]: Re-ranked list of documents (length ≤ ``topn``),
                with updated ``score`` fields.
        """
        raise NotImplementedError


class RrfReRanker(ReRanker):
    """Re-ranker using Reciprocal Rank Fusion (RRF).

    RRF combines results from multiple queries without requiring relevance scores.
    It assigns higher weight to documents that appear early in multiple result lists.

    The RRF score for a document at rank ``r`` is: ``1 / (k + r + 1)``,
    where ``k`` is the rank constant.

    Args:
        query (Optional[str], optional): Ignored by RRF. Defaults to None.
        topn (int, optional): Number of top documents to return. Defaults to 10.
        rerank_field (Optional[str], optional): Ignored by RRF. Defaults to None.
        rank_constant (int, optional): Smoothing constant ``k`` in RRF formula.
            Larger values reduce the impact of early ranks. Defaults to 60.
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        rank_constant: int = 60,
    ):
        super().__init__(query, topn, rerank_field)
        self._rank_constant = rank_constant

    @property
    def rank_constant(self) -> int:
        return self._rank_constant

    def _rrf_score(self, rank: int):
        return 1.0 / (self._rank_constant + rank + 1)

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Apply Reciprocal Rank Fusion to combine multiple query results.

        Args:
            query_results (dict[str, list[Doc]]): Results from one or more vector queries.

        Returns:
            list[Doc]: Re-ranked documents with RRF scores in the ``score`` field.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for _, query_result in query_results.items():
            for rank, doc in enumerate(query_result):
                doc_id = doc.id
                rrf_score = self._rrf_score(rank)
                rrf_scores[doc_id] += rrf_score
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(self.topn, rrf_scores.items(), key=lambda x: x[1])
        results = []
        for doc_id, rrf_score in top_docs:
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=rrf_score)
            results.append(new_doc)
        return results


class WeightedReRanker(ReRanker):
    """Re-ranker that combines scores from multiple vector fields using weights.

    Each vector field's relevance score is normalized based on its metric type,
    then scaled by a user-provided weight. Final scores are summed across fields.

    Args:
        query (Optional[str], optional): Ignored. Defaults to None.
        topn (int, optional): Number of top documents to return. Defaults to 10.
        rerank_field (Optional[str], optional): Ignored. Defaults to None.
        metric (MetricType, optional): Distance metric used for score normalization.
            Defaults to ``MetricType.L2``.
        weights (Optional[dict[str, float]], optional): Weight per vector field.
            Fields not listed use weight 1.0. Defaults to None.

    Note:
        Supported metrics: L2, IP, COSINE. Scores are normalized to [0, 1].
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        metric: MetricType = MetricType.L2,
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(query, topn, rerank_field)
        self._weights = weights
        self._metric = metric

    @property
    def weights(self) -> dict[str, float]:
        """dict[str, float]: Weight mapping for vector fields."""
        return self._weights

    @property
    def metric(self) -> MetricType:
        """MetricType: Distance metric used for score normalization."""
        return self._metric

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Combine scores from multiple vector fields using weighted sum.

        Args:
            query_results (dict[str, list[Doc]]): Results per vector field.

        Returns:
            list[Doc]: Re-ranked documents with combined scores in ``score`` field.
        """
        weighted_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for vector_name, query_result in query_results.items():
            for _, doc in enumerate(query_result):
                doc_id = doc.id
                weighted_score = self._normalize_score(
                    doc.score, self.metric
                ) * self.weights.get(vector_name, 1.0)
                weighted_scores[doc_id] += weighted_score
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(
            self.topn, weighted_scores.items(), key=lambda x: x[1]
        )
        results = []
        for doc_id, weighted_score in top_docs:
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=weighted_score)
            results.append(new_doc)
        return results

    def _normalize_score(self, score: float, metric: MetricType) -> float:
        if metric == MetricType.L2:
            return 1.0 - 2 * math.atan(score) / math.pi
        if metric == MetricType.IP:
            return 0.5 + math.atan(score) / math.pi
        if metric == MetricType.COSINE:
            return 1.0 - score / 2.0
        raise ValueError("Unsupported metric type")


class QwenReRanker(ReRanker):
    """Re-ranker using Qwen (DashScope) LLM-based re-ranking API.

    This re-ranker sends documents to the DashScope TextReRank service for
    cross-encoder style re-ranking based on semantic relevance to the query.

    Args:
        query (str): Query text for semantic re-ranking. **Required**.
        topn (int, optional): Number of top documents to return. Defaults to 10.
        rerank_field (str): Field name containing document text for re-ranking.
            **Required**.
        model (str, optional): DashScope re-ranking model name.
            Defaults to ``"gte-rerank-v2"``.
        api_key (Optional[str], optional): DashScope API key. If not provided,
            reads from ``DASHSCOPE_API_KEY`` environment variable.

    Raises:
        ValueError: If ``query`` is missing, ``rerank_field`` is missing,
            or API key is not provided.

    Note:
        Requires the ``dashscope`` Python package.
        Documents without content in ``rerank_field`` are skipped.
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model: str = "gte-rerank-v2",
        api_key: Optional[str] = None,
    ):
        super().__init__(query, topn, rerank_field)
        if not query:
            raise ValueError("Query is required for reranking")
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DashScope API key is required")

    @property
    def model(self) -> str:
        """str: DashScope re-ranking model name."""
        return self._model

    def _connection(self):
        dashscope = require_module("dashscope")
        dashscope.api_key = self._api_key
        return dashscope

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-rank documents using Qwen's TextReRank API.

        Args:
            query_results (dict[str, list[Doc]]): Results from vector search.

        Returns:
            list[Doc]: Re-ranked documents with relevance scores from Qwen.

        Raises:
            ValueError: If API call fails or no valid documents are found.
        """
        if not query_results:
            return []

        id_to_doc: dict[str, Doc] = {}
        doc_ids = []
        contents = []

        for _, query_result in query_results.items():
            for doc in query_result:
                doc_id = doc.id
                if doc_id in id_to_doc:
                    continue

                field_value = doc.field(self.rerank_field)
                rank_content = str(field_value).strip() if field_value else ""
                if not rank_content:
                    continue

                id_to_doc[doc_id] = doc
                doc_ids.append(doc_id)
                contents.append(rank_content)

        if not contents:
            raise ValueError("No documents to rerank")

        resp = self._connection().TextReRank.call(
            model=self.model,
            query=self.query,
            documents=list(contents),
            top_n=self.topn,
            return_documents=False,
        )

        if resp.status_code != HTTPStatus.OK:
            raise ValueError(
                f"QwenReranker failed with status {resp.status_code}: {resp.message}"
            )

        results = []
        for item in resp.output.results:
            idx = item.index
            doc_id = doc_ids[idx]
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=item.relevance_score)
            results.append(new_doc)

        return results
