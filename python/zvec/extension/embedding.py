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

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from http import HTTPStatus
from typing import Optional, Union

from ..tool import require_module
from ..typing import DataType


class DenseEmbeddingFunction(ABC):
    """Abstract base class for dense vector embedding functions.

    Dense embedding functions map text to fixed-length real-valued vectors.
    Subclasses must implement the ``embed()`` method.

    Args:
        dimension (int): Dimensionality of the output embedding vector.
        data_type (DataType, optional): Numeric type of the embedding.
            Defaults to ``DataType.VECTOR_FP32``.

    Note:
        This class is callable: ``embedding_func("text")`` is equivalent to
        ``embedding_func.embed("text")``.
    """

    def __init__(self, dimension: int, data_type: DataType = DataType.VECTOR_FP32):
        self._dimension = dimension
        self._data_type = data_type

    @property
    def dimension(self) -> int:
        """int: The expected dimensionality of the embedding vector."""
        return self._dimension

    @property
    def data_type(self) -> DataType:
        """DataType: The numeric data type of the embedding (e.g., VECTOR_FP32)."""
        return self._data_type

    @abstractmethod
    def embed(self, text: str) -> list[Union[int, float]]:
        """Generate a dense embedding vector for the input text.

        Args:
            text (str): Input text to embed.

        Returns:
            list[Union[int, float]]: A list of numbers representing the embedding.
                Length must equal ``self.dimension``.
        """
        raise NotImplementedError

    def __call__(self, text: str) -> list[Union[int, float]]:
        return self.embed(text)


class SparseEmbeddingFunction(ABC):
    """Abstract base class for sparse vector embedding functions.

    Sparse embedding functions map text to a dictionary of {index: weight},
    where only non-zero dimensions are stored.

    Note:
        Subclasses must implement the ``embed()`` method.
    """

    @abstractmethod
    def embed(self, text: str) -> dict[int, float]:
        """Generate a sparse embedding for the input text.
        Args:
            text (str): Input text to embed.

        Returns:
            dict[int, float]: Mapping from dimension index to non-zero weight.
        """
        raise NotImplementedError


class QwenEmbeddingFunction(DenseEmbeddingFunction):
    """Dense embedding function using Qwen (DashScope) Text Embedding API.

    This implementation uses the DashScope service to generate embeddings
    via Qwen's text embedding models (e.g., ``text-embedding-v4``).

    Args:
        dimension (int): Desired embedding dimension (e.g., 1024).
        model (str, optional): DashScope embedding model name.
            Defaults to ``"text-embedding-v4"``.
        api_key (Optional[str], optional): DashScope API key. If not provided,
            reads from ``DASHSCOPE_API_KEY`` environment variable.

    Raises:
        ValueError: If API key is missing or input text is invalid.

    Note:
        Requires the ``dashscope`` Python package.
        Embedding results are cached using ``functools.lru_cache`` (maxsize=10).
    """

    def __init__(
        self,
        dimension: int,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
    ):
        super().__init__(dimension, DataType.VECTOR_FP32)
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DashScope API key is required")

    @property
    def model(self) -> str:
        """str: The DashScope embedding model name in use."""
        return self._model

    def _connection(self):
        dashscope = require_module("dashscope")
        dashscope.api_key = self._api_key
        return dashscope

    @lru_cache(maxsize=10)
    def embed(self, text: str) -> list[Union[int, float]]:
        """
        Generate embedding for a given text using Qwen (via DashScope).

        Args:
            text (str): Input text to embed. Must be non-empty and valid string.

        Returns:
            list[Union[int, float]]: The dense embedding vector.

        Raises:
            ValueError: If input is invalid or API response is malformed.
            RuntimeError: If network or internal error occurs during API call.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected 'text' to be str, got {type(text).__name__}")

        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty or whitespace only")

        resp = self._connection().TextEmbedding.call(
            model=self.model, input=text, dimension=self.dimension, output_type="dense"
        )

        if resp.status_code != HTTPStatus.OK:
            error_msg = getattr(resp, "message", "Unknown error")
            error_detail = f"Status={resp.status_code}, Message={error_msg}"
            raise ValueError(f"QwenEmbedding failed: {error_detail}")

        output = getattr(resp, "output", None)
        if not isinstance(output, dict):
            raise ValueError("Invalid response: missing or malformed 'output' field")

        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError(
                "Invalid response: 'embeddings' field is missing or not a list"
            )

        if len(embeddings) != 1:
            raise ValueError(
                f"Expected 1 embedding, got {len(embeddings)}. Response: {resp}"
            )

        first_emb = embeddings[0]
        if not isinstance(first_emb, dict):
            raise ValueError("Invalid response: embedding item is not a dictionary")

        return list(first_emb.get("embedding"))
