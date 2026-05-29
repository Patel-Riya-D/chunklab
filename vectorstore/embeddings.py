from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from utils.azure_config import get_azure_embedding_config


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers/{model_name}"
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


class HashingEmbedder:
    name = "local-hashing-fallback"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class AzureOpenAIEmbedder:
    name = "azure-openai-embeddings"

    def __init__(self) -> None:
        config = get_azure_embedding_config()
        if config is None:
            raise ValueError("Azure OpenAI embedding environment variables are not fully configured.")

        from openai import AzureOpenAI

        self.config = config
        self.name = f"azure-openai/{config.deployment}"
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.config.deployment,
            input=texts,
        )
        return [list(map(float, item.embedding)) for item in response.data]


def get_default_embedder(prefer_sentence_transformers: bool = False) -> Embedder:
    try:
        return AzureOpenAIEmbedder()
    except Exception:
        pass

    if prefer_sentence_transformers:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            return HashingEmbedder()
    return HashingEmbedder()
