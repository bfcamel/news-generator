from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EmbeddingKind(StrEnum):
    DOC = "doc"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vector: list[float]

    kind: EmbeddingKind
    model_uri: str
    model_version: str | None

    dimension: int
    num_tokens: int | None

    def metadata(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "model_uri": self.model_uri,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "num_tokens": self.num_tokens,
        }
