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
from http import HTTPStatus
from unittest.mock import MagicMock, patch


import pytest
from zvec.extension import QwenEmbeddingFunction


# ----------------------------
# QwenEmbeddingFunction Test Case
# ----------------------------
class TestQwenEmbeddingFunction:
    def test_init_with_api_key(self):
        # Test initialization with explicit API key
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        assert embedding_func.dimension == 128
        assert embedding_func.model == "text-embedding-v4"
        assert embedding_func._api_key == "test_key"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"})
    def test_init_with_env_api_key(self):
        # Test initialization with API key from environment
        embedding_func = QwenEmbeddingFunction(dimension=128)
        assert embedding_func._api_key == "env_key"

    def test_init_without_api_key(self):
        # Test initialization without API key raises ValueError
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenEmbeddingFunction(dimension=128)

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""})
    def test_init_with_empty_env_api_key(self):
        # Test initialization with empty API key from environment
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenEmbeddingFunction(dimension=128)

    def test_model_property(self):
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        assert embedding_func.model == "text-embedding-v4"

        embedding_func = QwenEmbeddingFunction(
            dimension=128, model="custom-model", api_key="test_key"
        )
        assert embedding_func.model == "custom-model"

    @patch("zvec.extension.embedding.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        # Test embed method with empty text raises ValueError
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(TypeError):
            embedding_func.embed(None)

    @patch("zvec.extension.embedding.require_module")
    def test_embed_success(self, mock_require_module):
        # Test successful embedding
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        result = embedding_func.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=128,
            output_type="dense",
        )

    @patch("zvec.extension.embedding.require_module")
    def test_embed_http_error(self, mock_require_module):
        # Test embedding with HTTP error
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @patch("zvec.extension.embedding.require_module")
    def test_embed_invalid_response(self, mock_require_module):
        # Test embedding with invalid response (wrong number of embeddings)
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output.embeddings = []
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @pytest.mark.skip(reason="Qwen Embedding is not available in CI")
    def test_embed(self):
        # Test embedding with invalid dimension
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="xxx")
        dense = embedding_func("test text")
        assert len(dense) == 128
