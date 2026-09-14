from __future__ import annotations

import asyncio
import math

from src.infrastructure.embeddings import (
    EmbeddingSettings,
    YandexEmbeddingClient,
)


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    dot = sum(
        a * b
        for a, b in zip(
            left,
            right,
        )
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left
        )
    )

    right_norm = math.sqrt(
        sum(
            value * value
            for value in right
        )
    )

    return (
        dot
        / (left_norm * right_norm)
    )


async def main() -> None:
    settings = (
        EmbeddingSettings.from_env()
    )

    english_document = (
        "Dromedaries and Bactrian camels "
        "can produce fertile first-generation "
        "hybrids."
    )

    russian_query = (
        "Дромадеры и бактрианы способны "
        "давать плодовитых гибридов "
        "первого поколения."
    )

    async with YandexEmbeddingClient(
        settings
    ) as client:
        doc = await client.embed_doc(
            english_document
        )

        query = await client.embed_query(
            russian_query
        )

    score = cosine_similarity(
        query.vector,
        doc.vector,
    )

    print(
        f"DOC model:   "
        f"{doc.model_uri}"
    )

    print(
        f"QUERY model: "
        f"{query.model_uri}"
    )

    print(
        f"DOC version: "
        f"{doc.model_version}"
    )

    print(
        f"QUERY version: "
        f"{query.model_version}"
    )

    print(
        f"Dimension:   "
        f"{len(doc.vector)}"
    )

    print(
        f"DOC tokens:  "
        f"{doc.num_tokens}"
    )

    print(
        f"QUERY tokens:"
        f" {query.num_tokens}"
    )

    print(
        f"Cosine:      "
        f"{score:.6f}"
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
