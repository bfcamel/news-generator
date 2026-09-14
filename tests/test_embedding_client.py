from __future__ import annotations

import pytest

from src.infrastructure.embeddings.config import (
    EmbeddingSettings,
)
from src.infrastructure.embeddings.client import (
    YandexEmbeddingClient,
)


@pytest.mark.asyncio
async def test_empty_text_is_rejected() -> None:
    settings = EmbeddingSettings(
        api_key="test",
        folder_id="folder",
        dimension=512,
    )

    async with YandexEmbeddingClient(
        settings
    ) as client:
        with pytest.raises(
            ValueError
        ):
            await client.embed_doc(
                "   "
            )
