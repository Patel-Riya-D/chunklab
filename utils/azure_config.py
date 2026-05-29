from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class AzureLLMConfig:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str


@dataclass(frozen=True)
class AzureEmbeddingConfig:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str


def get_azure_llm_config() -> AzureLLMConfig | None:
    values = {
        "api_key": os.getenv("AZURE_OPENAI_LLM_KEY"),
        "endpoint": os.getenv("AZURE_LLM_ENDPOINT"),
        "api_version": os.getenv("AZURE_LLM_API_VERSION"),
        "deployment": os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI"),
    }
    if all(values.values()):
        return AzureLLMConfig(**values)  # type: ignore[arg-type]
    return None


def get_azure_embedding_config() -> AzureEmbeddingConfig | None:
    values = {
        "api_key": os.getenv("AZURE_OPENAI_EMB_KEY"),
        "endpoint": os.getenv("AZURE_EMB_ENDPOINT"),
        "api_version": os.getenv("AZURE_EMB_API_VERSION"),
        "deployment": os.getenv("AZURE_EMB_DEPLOYMENT"),
    }
    if all(values.values()):
        return AzureEmbeddingConfig(**values)  # type: ignore[arg-type]
    return None

