from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.domain.atomic_thesis_candidate import (
    CandidateAtomicThesis,
    ExtractionStatus,
    SemanticUnitThesisExtraction,
)
from src.infrastructure.llm.yandex_gpt_thesis_extractor import (
    YandexGPTThesisExtractor,
)
from src.repositories.atomic_thesis_candidate_repository import (
    AtomicThesisCandidateRepository,
    make_extraction_id,
)
from src.repositories.source_document_repository import (
    SourceDocumentRepository,
)

from src.domain.atomic_thesis import (
    TaxonScope,
)


logger = logging.getLogger(
    "news_generator.atomic_thesis_extraction"
)

def normalize_taxon_scope(
    *,
    taxa: list[str],
    taxon_scope: TaxonScope,
) -> tuple[TaxonScope, bool]:
    """
    Защита от логически несовместимого сочетания,
    которое может быть формально валидным по JSON Schema.

    Никаких таксономических предположений здесь не делаем.

    Если scope противоречит количеству taxa,
    переводим его в unspecified и оставляем
    окончательное решение человеку при модерации.

    Возвращает:
        (нормализованный scope, был ли scope изменён)
    """

    count = len(taxa)

    invalid = False

    if (
        taxon_scope
        == TaxonScope.SPECIES
        and count != 1
    ):
        invalid = True

    elif (
        taxon_scope
        == TaxonScope.MULTI_SPECIES
        and count < 2
    ):
        invalid = True

    elif (
        taxon_scope
        in {
            TaxonScope.GENUS,
            TaxonScope.FAMILY,
        }
        and count < 1
    ):
        invalid = True

    if invalid:
        return (
            TaxonScope.UNSPECIFIED,
            True,
        )

    return (
        taxon_scope,
        False,
    )


@dataclass(
    frozen=True,
    slots=True,
)
class BatchExtractionResult:
    units_processed: int
    units_failed: int
    candidates_created: int
    zero_candidate_units: int


class AtomicThesisCandidateService:
    """
    SemanticUnit -> YandexGPT -> CandidateAtomicThesis.

    Здесь НЕТ embeddings.
    """

    def __init__(
        self,
        *,
        repository: (
            AtomicThesisCandidateRepository
        ),
        source_repository: (
            SourceDocumentRepository
        ),
        extractor: YandexGPTThesisExtractor,
    ) -> None:
        self.repository = repository

        self.source_repository = (
            source_repository
        )

        self.extractor = extractor

    async def generate_next(
        self,
        *,
        source_document_id: str,
        limit: int,
    ) -> BatchExtractionResult:
        source = (
            await self.source_repository.get(
                source_document_id
            )
        )

        if source is None:
            raise ValueError(
                "Источник не найден"
            )

        units = (
            await self.repository
            .units_for_generation(
                source_document_id=(
                    source_document_id
                ),
                prompt_version=(
                    self.extractor
                    .prompt_version
                ),
                limit=limit,
            )
        )

        processed = 0
        failed = 0
        candidate_count = 0
        zero_candidate_units = 0

        for number, unit in enumerate(
            units,
            start=1,
        ):
            extraction_id = (
                make_extraction_id(
                    semantic_unit_id=(
                        unit.id
                    ),
                    prompt_version=(
                        self.extractor
                        .prompt_version
                    ),
                )
            )

            try:
                response = (
                    await self.extractor.extract(
                        unit=unit,
                        source=source,
                    )
                )

                candidate_ids: list[
                    str
                ] = []

                for extracted in response.theses:
                    normalized_scope, scope_was_normalized = (
                        normalize_taxon_scope(
                            taxa=extracted.taxa,
                            taxon_scope=(
                                extracted.taxon_scope
                            ),
                        )
                    )

                    metadata = {
                        "semantic_unit_position": (
                            unit.position
                        ),
                        "section_title": (
                            unit.section_title
                        ),
                    }

                    # Сохраняем исходное решение модели,
                    # если оно оказалось логически несовместимым.
                    # Это позволит увидеть ошибку при ручной проверке.
                    if scope_was_normalized:
                        metadata[
                            "llm_taxon_scope"
                        ] = (
                            extracted
                            .taxon_scope
                            .value
                        )

                        metadata[
                            "taxon_scope_normalized"
                        ] = True

                        logger.warning(
                            "Invalid taxon scope from LLM: "
                            "unit=%s text=%r taxa=%r "
                            "scope=%s -> unspecified",
                            unit.id,
                            extracted.text,
                            extracted.taxa,
                            extracted.taxon_scope.value,
                        )

                    candidate = (
                        CandidateAtomicThesis(
                            id=(
                                "candidate_"
                                f"{uuid4().hex}"
                            ),
                            source_semantic_unit_id=(
                                unit.id
                            ),
                            source_document_id=(
                                unit.source_document_id
                            ),
                            text=(
                                extracted.text
                            ),
                            original_text=(
                                extracted.text
                            ),
                            taxa=(
                                extracted.taxa
                            ),
                            taxon_scope=(
                                normalized_scope
                            ),
                            extraction_model=(
                                self.extractor
                                .model_name
                            ),
                            prompt_version=(
                                self.extractor
                                .prompt_version
                            ),
                            metadata=metadata,
                        )
                    )

                    stored = (
                        await self.repository
                        .create_candidate(
                            candidate
                        )
                    )

                    candidate_ids.append(
                        stored.id
                    )

                now = datetime.now(
                    timezone.utc
                )

                extraction = (
                    SemanticUnitThesisExtraction(
                        id=extraction_id,
                        semantic_unit_id=(
                            unit.id
                        ),
                        source_document_id=(
                            unit
                            .source_document_id
                        ),
                        status=(
                            ExtractionStatus
                            .COMPLETED
                        ),
                        candidate_ids=(
                            candidate_ids
                        ),
                        model=(
                            self.extractor
                            .model_name
                        ),
                        prompt_version=(
                            self.extractor
                            .prompt_version
                        ),
                        updated_at=now,
                    )
                )

                await self.repository.save_extraction(
                    extraction
                )

                processed += 1

                candidate_count += len(
                    candidate_ids
                )

                if not candidate_ids:
                    zero_candidate_units += 1

                logger.info(
                    "[AtomicThesis extraction] "
                    "%d/%d | unit=%s | "
                    "candidates=%d",
                    number,
                    len(units),
                    unit.id,
                    len(candidate_ids),
                )

            except Exception as exc:
                failed += 1

                now = datetime.now(
                    timezone.utc
                )

                extraction = (
                    SemanticUnitThesisExtraction(
                        id=extraction_id,
                        semantic_unit_id=(
                            unit.id
                        ),
                        source_document_id=(
                            unit
                            .source_document_id
                        ),
                        status=(
                            ExtractionStatus
                            .FAILED
                        ),
                        candidate_ids=[],
                        model=(
                            self.extractor
                            .model_name
                        ),
                        prompt_version=(
                            self.extractor
                            .prompt_version
                        ),
                        error=str(exc),
                        updated_at=now,
                    )
                )

                await self.repository.save_extraction(
                    extraction
                )

                logger.exception(
                    "[AtomicThesis extraction] "
                    "%d/%d failed | unit=%s",
                    number,
                    len(units),
                    unit.id,
                )

        return BatchExtractionResult(
            units_processed=processed,
            units_failed=failed,
            candidates_created=(
                candidate_count
            ),
            zero_candidate_units=(
                zero_candidate_units
            ),
        )