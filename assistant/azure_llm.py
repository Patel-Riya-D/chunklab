from __future__ import annotations

from utils.azure_config import get_azure_llm_config
from utils.models import RetrievalResult


class AzureAnswerGenerator:
    name = "azure-openai-chat-completions"

    def __init__(self) -> None:
        config = get_azure_llm_config()
        if config is None:
            raise ValueError("Azure OpenAI LLM environment variables are not fully configured.")

        from openai import AzureOpenAI

        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        contexts = []
        for result in retrieved:
            title = result.chunk.metadata.get("title", "untitled")
            contexts.append(
                f"Chunk ID: {result.chunk.id}\n"
                f"Title: {title}\n"
                f"Score: {result.score:.3f}\n"
                f"Text:\n{result.chunk.text}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document retrieval assistant. Answer the user's question directly using only "
                    "the provided retrieved chunks. If the answer is not supported by the chunks, say that "
                    "the retrieved context does not contain enough information. Keep the answer concise, "
                    "then include a short 'Context used' line with chunk IDs."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Retrieved chunks:\n\n"
                    + "\n\n---\n\n".join(contexts)
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.config.deployment,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
