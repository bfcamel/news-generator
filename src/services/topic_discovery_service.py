from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.domain.publication_trial import (
    PublicationDiscoveryMetrics,
    PublicationEvidence,
)
from src.repositories.publication_trial_store import (
    FilePublicationTrialStore,
)
from src.repositories.source_document_repository import (
    SourceDocumentRepository,
)


SEMANTIC_UNITS_INDEX = "semantic_units"


@dataclass(frozen=True, slots=True)
class TopicDiscoveryResult:
    evidence: list[PublicationEvidence]
    metrics: PublicationDiscoveryMetrics


class TopicDiscoveryService:
    """
    Finds a dense semantic neighborhood without calling an LLM.

    The best cluster is chosen by vector similarity, amount of evidence,
    source diversity and recent-use exclusion.
    """

    def __init__(
        self,
        *,
        es: AsyncElasticsearch,
        source_repository: SourceDocumentRepository,
        trial_store: FilePublicationTrialStore,
        seed_count: int = 24,
        neighbor_count: int = 12,
        max_evidence: int = 10,
        min_similarity: float = 0.72,
        recent_trial_limit: int = 20,
    ) -> None:
        self.es = es
        self.source_repository = (
            source_repository
        )
        self.trial_store = trial_store

        self.seed_count = seed_count
        self.neighbor_count = neighbor_count
        self.max_evidence = max_evidence
        self.min_similarity = min_similarity
        self.recent_trial_limit = (
            recent_trial_limit
        )

    async def discover(
        self,
    ) -> TopicDiscoveryResult:
        excluded_ids = (
            await self._recently_used_unit_ids()
        )

        seeds = await self._load_seed_candidates(
            excluded_ids=excluded_ids,
        )

        if not seeds and excluded_ids:
            # If the small test history temporarily covers too much
            # of the collection, allow reuse rather than failing.
            excluded_ids = set()

            seeds = await self._load_seed_candidates(
                excluded_ids=excluded_ids,
            )

        if not seeds:
            raise RuntimeError(
                "Не найдено Semantic Units с doc_embedding. "
                "Сначала сгенерируй embeddings."
            )

        source_map = {
            source.id: source
            for source
            in await self.source_repository.list_all()
        }

        best: TopicDiscoveryResult | None = None

        for seed in seeds:
            result = await self._build_cluster(
                seed=seed,
                excluded_ids=excluded_ids,
                source_map=source_map,
            )

            if result is None:
                continue

            if (
                best is None
                or (
                    result
                    .metrics
                    .selection_score
                    >
                    best
                    .metrics
                    .selection_score
                )
            ):
                best = result

        if best is None and excluded_ids:
            # Recent-history exclusion can make otherwise good
            # neighborhoods too sparse. Retry once without exclusions.
            seeds = await self._load_seed_candidates(
                excluded_ids=set(),
            )

            for seed in seeds:
                result = await self._build_cluster(
                    seed=seed,
                    excluded_ids=set(),
                    source_map=source_map,
                )

                if result is None:
                    continue

                if (
                    best is None
                    or (
                        result
                        .metrics
                        .selection_score
                        >
                        best
                        .metrics
                        .selection_score
                    )
                ):
                    best = result

        if best is None:
            raise RuntimeError(
                "Не удалось собрать достаточно связную "
                "группу Semantic Units."
            )

        return best

    async def _recently_used_unit_ids(
        self,
    ) -> set[str]:
        recent = await self.trial_store.list_recent(
            limit=self.recent_trial_limit
        )

        result: set[str] = set()

        for trial in recent:
            result.update(
                trial.selected_semantic_unit_ids
            )

        return result

    async def _load_seed_candidates(
        self,
        *,
        excluded_ids: set[str],
    ) -> list[dict[str, Any]]:
        must_not: list[dict[str, Any]] = []

        if excluded_ids:
            must_not.append(
                {
                    "terms": {
                        "id": sorted(
                            excluded_ids
                        )
                    }
                }
            )

        response = await self.es.search(
            index=SEMANTIC_UNITS_INDEX,
            size=self.seed_count,
            query={
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "exists": {
                                        "field": (
                                            "doc_embedding"
                                        )
                                    }
                                }
                            ],
                            "must_not": (
                                must_not
                            ),
                        }
                    },
                    "random_score": {},
                }
            },
            source_includes=[
                "id",
                "doc_embedding",
            ],
        )

        return response[
            "hits"
        ][
            "hits"
        ]

    async def _build_cluster(
        self,
        *,
        seed: dict[str, Any],
        excluded_ids: set[str],
        source_map: dict[str, Any],
    ) -> TopicDiscoveryResult | None:
        seed_source = seed.get(
            "_source"
        ) or {}

        vector = seed_source.get(
            "doc_embedding"
        )

        seed_id = str(
            seed_source.get(
                "id",
                seed.get(
                    "_id",
                    "",
                ),
            )
        )

        if not vector or not seed_id:
            return None

        filter_query: dict[str, Any] = {
            "exists": {
                "field": "doc_embedding"
            }
        }

        if excluded_ids:
            filter_query = {
                "bool": {
                    "filter": [
                        {
                            "exists": {
                                "field": (
                                    "doc_embedding"
                                )
                            }
                        }
                    ],
                    "must_not": [
                        {
                            "terms": {
                                "id": sorted(
                                    excluded_ids
                                )
                            }
                        }
                    ],
                }
            }

        response = await self.es.search(
            index=SEMANTIC_UNITS_INDEX,
            size=self.neighbor_count,
            knn={
                "field": "doc_embedding",
                "query_vector": vector,
                "k": self.neighbor_count,
                "num_candidates": max(
                    100,
                    self.neighbor_count * 10,
                ),
                "filter": filter_query,
            },
            source_excludes=[
                "doc_embedding",
                "embedding_model",
            ],
        )

        hits = response[
            "hits"
        ][
            "hits"
        ]

        filtered_hits = [
            hit
            for hit in hits
            if (
                str(
                    (
                        hit.get(
                            "_source"
                        )
                        or {}
                    ).get(
                        "id",
                        hit.get(
                            "_id",
                            "",
                        ),
                    )
                )
                == seed_id
                or float(
                    hit.get(
                        "_score",
                        0.0,
                    )
                )
                >= self.min_similarity
            )
        ]

        filtered_hits = filtered_hits[
            :self.max_evidence
        ]

        if len(filtered_hits) < 4:
            return None

        evidence: list[
            PublicationEvidence
        ] = []

        for hit in filtered_hits:
            source = hit.get(
                "_source"
            ) or {}

            source_document_id = str(
                source.get(
                    "source_document_id",
                    "",
                )
            )

            source_document = (
                source_map.get(
                    source_document_id
                )
            )

            if (
                not source_document_id
                or source_document is None
            ):
                continue

            evidence.append(
                PublicationEvidence(
                    semantic_unit_id=str(
                        source.get(
                            "id",
                            hit.get(
                                "_id",
                                "",
                            ),
                        )
                    ),
                    source_document_id=(
                        source_document_id
                    ),
                    similarity=float(
                        hit.get(
                            "_score",
                            0.0,
                        )
                    ),
                    text=str(
                        source.get(
                            "text",
                            "",
                        )
                    ),
                    taxa=list(
                        source.get(
                            "taxa"
                        )
                        or []
                    ),
                    section_title=(
                        source.get(
                            "section_title"
                        )
                    ),
                    page_start=(
                        source.get(
                            "page_start"
                        )
                    ),
                    page_end=(
                        source.get(
                            "page_end"
                        )
                    ),
                    source_title=(
                        source_document.title
                    ),
                    source_authors=(
                        source_document.authors
                    ),
                    source_year=(
                        source_document.year
                    ),
                    source_doi=(
                        source_document.doi
                    ),
                    source_url=(
                        source_document.url
                    ),
                )
            )

        if len(evidence) < 4:
            return None

        average_similarity = (
            sum(
                item.similarity
                for item in evidence
            )
            / len(evidence)
        )

        source_count = len(
            {
                item.source_document_id
                for item in evidence
            }
        )

        # Similarity dominates. More evidence and independent
        # sources are modest bonuses rather than hard requirements.
        selection_score = (
            average_similarity
            + 0.01 * min(
                len(evidence),
                10,
            )
            + 0.025 * min(
                source_count,
                4,
            )
        )

        return TopicDiscoveryResult(
            evidence=evidence,
            metrics=(
                PublicationDiscoveryMetrics(
                    seed_semantic_unit_id=(
                        seed_id
                    ),
                    average_similarity=(
                        average_similarity
                    ),
                    source_count=(
                        source_count
                    ),
                    evidence_count=(
                        len(evidence)
                    ),
                    selection_score=(
                        selection_score
                    ),
                )
            ),
        )
