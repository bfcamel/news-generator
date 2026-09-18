from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from .config import EmbeddingSettings
from .models import EmbeddingKind, EmbeddingResult


RETRYABLE_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}


class YandexEmbeddingError(RuntimeError):
    pass


class YandexEmbeddingClient:
    """
    Async client for Yandex Text Embeddings v2.

    Intended use:
        SemanticUnit.text -> embed_doc()
        Search/topic queries -> embed_query()
    """

    def __init__(
        self,
        settings: EmbeddingSettings,
    ) -> None:
        self.settings = settings

        self._semaphore = asyncio.Semaphore(
            settings.concurrency
        )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.timeout_seconds
            ),
            headers={
                "Authorization": (
                    f"Api-Key {settings.api_key}"
                ),
                "Content-Type": "application/json",
                "x-folder-id": settings.folder_id,
            },
        )

    async def __aenter__(
        self,
    ) -> "YandexEmbeddingClient":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_doc(
        self,
        text: str,
    ) -> EmbeddingResult:
        return await self._embed(
            text=text,
            kind=EmbeddingKind.DOC,
            model_uri=(
                self.settings.doc_model_uri
            ),
        )

    async def embed_query(
        self,
        text: str,
    ) -> EmbeddingResult:
        return await self._embed(
            text=text,
            kind=EmbeddingKind.QUERY,
            model_uri=(
                self.settings.query_model_uri
            ),
        )

    async def _embed(
        self,
        *,
        text: str,
        kind: EmbeddingKind,
        model_uri: str,
    ) -> EmbeddingResult:
        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                "Cannot generate an embedding "
                "for empty text"
            )

        async with self._semaphore:
            payload = await self._request_with_retry(
                text=normalized_text,
                model_uri=model_uri,
            )

        raw_embedding = payload.get(
            "embedding"
        )

        if not isinstance(
            raw_embedding,
            list,
        ):
            raise YandexEmbeddingError(
                "Yandex response does not contain "
                "an embedding array"
            )

        vector = [
            float(value)
            for value in raw_embedding
        ]

        expected_dim = (
            self.settings.dimension
        )

        if len(vector) != expected_dim:
            raise YandexEmbeddingError(
                "Unexpected embedding dimension: "
                f"{len(vector)}; "
                f"expected {expected_dim}"
            )

        raw_num_tokens = payload.get(
            "numTokens"
        )

        num_tokens = (
            int(raw_num_tokens)
            if raw_num_tokens is not None
            else None
        )

        raw_model_version = payload.get(
            "modelVersion"
        )

        model_version = (
            str(raw_model_version)
            if raw_model_version is not None
            else None
        )

        return EmbeddingResult(
            vector=vector,
            kind=kind,
            model_uri=model_uri,
            model_version=model_version,
            dimension=expected_dim,
            num_tokens=num_tokens,
        )

    async def _request_with_retry(
        self,
        *,
        text: str,
        model_uri: str,
    ) -> dict[str, Any]:
        last_exception: Exception | None = None

        attempts = (
            self.settings.max_retries + 1
        )

        for attempt in range(
            attempts
        ):
            try:
                response = await self._client.post(
                    self.settings.api_url,
                    json={
                        "modelUri": model_uri,
                        "text": text,
                        "dim": str(
                            self.settings.dimension
                        ),
                    },
                )

                if (
                    response.status_code
                    in RETRYABLE_STATUS_CODES
                ):
                    if attempt + 1 >= attempts:
                        raise YandexEmbeddingError(
                            self._format_http_error(
                                response
                            )
                        )

                    await self._sleep_before_retry(
                        attempt=attempt,
                        response=response,
                    )
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise YandexEmbeddingError(
                        self._format_http_error(
                            response
                        )
                    ) from exc

                data = response.json()

                if not isinstance(
                    data,
                    dict,
                ):
                    raise YandexEmbeddingError(
                        "Unexpected Yandex API "
                        "response type"
                    )

                return data

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                last_exception = exc

                if attempt + 1 >= attempts:
                    break

                await self._sleep_before_retry(
                    attempt=attempt,
                    response=None,
                )

        raise YandexEmbeddingError(
            "Yandex Embeddings request failed "
            "after retries"
        ) from last_exception

    async def _sleep_before_retry(
        self,
        *,
        attempt: int,
        response: httpx.Response | None,
    ) -> None:
        retry_after: float | None = None

        if response is not None:
            raw_retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            if raw_retry_after:
                try:
                    retry_after = float(
                        raw_retry_after
                    )
                except ValueError:
                    retry_after = None

        if retry_after is None:
            retry_after = (
                self.settings.retry_base_seconds
                * (2 ** attempt)
            )

            retry_after += random.uniform(
                0.0,
                self.settings.retry_base_seconds,
            )

        await asyncio.sleep(
            retry_after
        )

    @staticmethod
    def _format_http_error(
        response: httpx.Response,
    ) -> str:
        body = response.text

        if len(body) > 2000:
            body = (
                body[:2000]
                + "..."
            )

        return (
            "Yandex Embeddings API error: "
            f"status={response.status_code}; "
            f"body={body}"
        )
