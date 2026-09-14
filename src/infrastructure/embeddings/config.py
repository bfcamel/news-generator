from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


DEFAULT_API_URL = (
    "https://llm.api.cloud.yandex.net/"
    "foundationModels/v1/textEmbedding"
)


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    api_key: str
    folder_id: str

    dimension: int = 512
    api_url: str = DEFAULT_API_URL

    timeout_seconds: float = 60.0
    max_retries: int = 5
    retry_base_seconds: float = 0.75
    concurrency: int = 3

    @property
    def doc_model_uri(self) -> str:
        return (
            f"emb://{self.folder_id}/"
            "text-embeddings-v2-doc/latest"
        )

    @property
    def query_model_uri(self) -> str:
        return (
            f"emb://{self.folder_id}/"
            "text-embeddings-v2-query/latest"
        )

    @classmethod
    def from_env(
        cls,
        *,
        load_dotenv_file: bool = True,
    ) -> "EmbeddingSettings":
        if load_dotenv_file:
            env_path = find_dotenv(
                usecwd=True,
            )

            if env_path:
                load_dotenv(
                    env_path,
                    override=False,
                )
            else:
                load_dotenv(
                    override=False,
                )

        api_key = os.getenv(
            "YANDEX_API_KEY",
        )

        folder_id = os.getenv(
            "YANDEX_FOLDER_ID",
        )

        if not api_key:
            raise RuntimeError(
                "YANDEX_API_KEY is missing in .env"
            )

        if not folder_id:
            raise RuntimeError(
                "YANDEX_FOLDER_ID is missing in .env"
            )

        dimension = int(
            os.getenv(
                "YANDEX_EMBEDDING_DIM",
                "512",
            )
        )

        if dimension not in {
            128,
            256,
            512,
            768,
        }:
            raise ValueError(
                "YANDEX_EMBEDDING_DIM must be one "
                "of 128, 256, 512, 768"
            )

        concurrency = int(
            os.getenv(
                "YANDEX_EMBEDDING_CONCURRENCY",
                "3",
            )
        )

        if concurrency < 1:
            raise ValueError(
                "YANDEX_EMBEDDING_CONCURRENCY "
                "must be >= 1"
            )

        timeout_seconds = float(
            os.getenv(
                "YANDEX_EMBEDDING_TIMEOUT",
                "60",
            )
        )

        max_retries = int(
            os.getenv(
                "YANDEX_EMBEDDING_MAX_RETRIES",
                "5",
            )
        )

        return cls(
            api_key=api_key,
            folder_id=folder_id,
            dimension=dimension,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
