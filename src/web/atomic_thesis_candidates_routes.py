from __future__ import annotations

import json

from typing import Any

from fastapi import (
    APIRouter,
    File,
    Request,
    UploadFile,
)
from fastapi.responses import (
    RedirectResponse,
    Response,
)
from fastapi.templating import (
    Jinja2Templates,
)
from pydantic import ValidationError

from src.domain.atomic_thesis import (
    TaxonScope,
)
from src.domain.atomic_thesis_candidate import (
    CandidateStatus,
)
from src.domain.atomic_thesis_review import (
    REVIEW_SCHEMA_VERSION,
    AtomicThesisReviewImport,
)
from src.repositories.atomic_thesis_candidate_repository import (
    AtomicThesisCandidateRepository,
)
from src.repositories.source_document_repository import (
    SourceDocumentRepository,
)
from src.services.atomic_thesis_candidate_service import (
    AtomicThesisCandidateService,
)
from src.services.atomic_thesis_moderation_service import (
    AtomicThesisModerationService,
)


MAX_REVIEW_FILE_SIZE = (
    1024 * 1024
)


def create_atomic_thesis_candidates_router(
    *,
    templates: Jinja2Templates,
    candidate_repository: (
        AtomicThesisCandidateRepository
    ),
    candidate_service: (
        AtomicThesisCandidateService
    ),
    moderation_service: (
        AtomicThesisModerationService
    ),
    source_repository: (
        SourceDocumentRepository
    ),
    taxon_suggestions: list[str],
) -> APIRouter:

    router = APIRouter()

    # ========================================================
    # DETAIL RENDER
    # ========================================================

    async def render_detail(
        *,
        request: Request,
        candidate_id: str,
        error: str | None = None,
        success: str | None = None,
        form_values: dict[str, Any] | None = None,
    ):
        candidate = (
            await candidate_repository
            .get_candidate(
                candidate_id
            )
        )

        if candidate is None:
            return RedirectResponse(
                "/atomic-thesis-candidates",
                status_code=303,
            )

        units = (
            await candidate_repository
            .get_semantic_units(
                [
                    candidate
                    .source_semantic_unit_id
                ]
            )
        )

        unit = units.get(
            candidate
            .source_semantic_unit_id
        )

        source = (
            await source_repository.get(
                candidate
                .source_document_id
            )
        )

        imported_taxa: list[str] = []

        if form_values is not None:
            raw_imported_taxa = (
                form_values.get(
                    "taxa",
                    [],
                )
            )

            if isinstance(
                raw_imported_taxa,
                list,
            ):
                imported_taxa = [
                    str(value)
                    for value
                    in raw_imported_taxa
                    if str(value).strip()
                ]

        taxon_options = list(
            dict.fromkeys(
                [
                    *candidate.taxa,
                    *imported_taxa,
                    *taxon_suggestions,
                ]
            )
        )

        return templates.TemplateResponse(
            request=request,
            name=(
                "atomic_thesis_candidate_detail.html"
            ),
            context={
                "candidate": candidate,
                "unit": unit,
                "source": source,
                "taxon_options": (
                    taxon_options
                ),
                "taxon_scopes": (
                    TaxonScope
                ),
                "error": error,
                "success": success,
                "form_values": (
                    form_values
                ),
            },
            status_code=(
                400
                if error
                else 200
            ),
        )

    # ========================================================
    # LIST
    # ========================================================

    @router.get(
        "/atomic-thesis-candidates"
    )
    async def candidates_page(
        request: Request,
    ):
        raw_status = (
            request.query_params.get(
                "status",
                "pending_review",
            )
        )

        try:
            status = CandidateStatus(
                raw_status
            )
        except ValueError:
            status = (
                CandidateStatus
                .PENDING_REVIEW
            )

        source_filter = (
            request.query_params.get(
                "source_document_id"
            )
            or None
        )

        candidates = (
            await candidate_repository
            .list_candidates(
                status=status,
                source_document_id=(
                    source_filter
                ),
                size=500,
            )
        )

        unit_map = (
            await candidate_repository
            .get_semantic_units(
                [
                    candidate
                    .source_semantic_unit_id
                    for candidate
                    in candidates
                ]
            )
        )

        documents = (
            await source_repository
            .list_all()
        )

        source_map = {
            document.id: document
            for document in documents
        }

        rows = [
            {
                "candidate": (
                    candidate
                ),
                "unit": unit_map.get(
                    candidate
                    .source_semantic_unit_id
                ),
                "source": source_map.get(
                    candidate
                    .source_document_id
                ),
            }
            for candidate in candidates
        ]

        pending_count = (
            await candidate_repository
            .count_candidates(
                CandidateStatus
                .PENDING_REVIEW
            )
        )

        return templates.TemplateResponse(
            request=request,
            name=(
                "atomic_thesis_candidates.html"
            ),
            context={
                "rows": rows,
                "documents": documents,
                "status": status,
                "statuses": (
                    CandidateStatus
                ),
                "source_filter": (
                    source_filter
                ),
                "pending_count": (
                    pending_count
                ),
                "generated": (
                    request
                    .query_params
                    .get(
                        "generated"
                    )
                ),
                "failed": (
                    request
                    .query_params
                    .get(
                        "failed"
                    )
                ),
                "units_processed": (
                    request
                    .query_params
                    .get(
                        "units_processed"
                    )
                ),
            },
        )

    # ========================================================
    # GENERATION
    # ========================================================

    @router.post(
        "/atomic-thesis-candidates/generate"
    )
    async def generate_candidates(
        request: Request,
    ):
        form = await request.form()

        source_document_id = str(
            form[
                "source_document_id"
            ]
        )

        limit = int(
            form.get(
                "limit",
                "10",
            )
        )

        limit = max(
            1,
            min(
                limit,
                25,
            ),
        )

        result = (
            await candidate_service
            .generate_next(
                source_document_id=(
                    source_document_id
                ),
                limit=limit,
            )
        )

        return RedirectResponse(
            (
                "/atomic-thesis-candidates"
                "?status=pending_review"
                "&source_document_id="
                f"{source_document_id}"
                "&generated="
                f"{result.candidates_created}"
                "&failed="
                f"{result.units_failed}"
                "&units_processed="
                f"{result.units_processed}"
            ),
            status_code=303,
        )

    # ========================================================
    # EXPORT FOR EXTERNAL REVIEW
    # ========================================================

    @router.get(
        "/atomic-thesis-candidates/"
        "{candidate_id}/export"
    )
    async def export_candidate_review(
        candidate_id: str,
    ):
        candidate = (
            await candidate_repository
            .require_candidate(
                candidate_id
            )
        )

        units = (
            await candidate_repository
            .get_semantic_units(
                [
                    candidate
                    .source_semantic_unit_id
                ]
            )
        )

        unit = units.get(
            candidate
            .source_semantic_unit_id
        )

        if unit is None:
            return Response(
                content=(
                    "SemanticUnit not found"
                ),
                status_code=404,
            )

        source = (
            await source_repository.get(
                candidate
                .source_document_id
            )
        )

        source_data: dict[
            str,
            Any,
        ] | None = None

        if source is not None:
            source_data = {
                "id": source.id,
                "title": source.title,
                "authors": (
                    source.authors
                ),
                "year": source.year,
                "doi": source.doi,
                "url": source.url,
                "primary_taxon": (
                    source.primary_taxon
                ),
            }

        export_data = {
            "schema_version": (
                REVIEW_SCHEMA_VERSION
            ),

            "candidate_id": (
                candidate.id
            ),

            "source_semantic_unit_id": (
                candidate
                .source_semantic_unit_id
            ),

            "source": (
                source_data
            ),

            "semantic_unit": {
                "id": unit.id,
                "source_document_id": (
                    unit
                    .source_document_id
                ),
                "position": (
                    unit.position
                ),
                "section_title": (
                    unit.section_title
                ),
                "page_start": (
                    unit.page_start
                ),
                "page_end": (
                    unit.page_end
                ),
                "taxa": (
                    unit.taxa
                ),
                "text": (
                    unit.text
                ),
            },

            "candidate": {
                "id": (
                    candidate.id
                ),
                "original_text": (
                    candidate
                    .original_text
                ),
                "current_text": (
                    candidate.text
                ),
                "taxa": (
                    candidate.taxa
                ),
                "taxon_scope": (
                    candidate
                    .taxon_scope
                    .value
                ),
                "extraction_model": (
                    candidate
                    .extraction_model
                ),
                "prompt_version": (
                    candidate
                    .prompt_version
                ),
                "metadata": (
                    candidate.metadata
                ),
            },

            "review_instructions": [
                (
                    "Проверяй тезис только "
                    "по semantic_unit.text."
                ),
                (
                    "Не используй внешние "
                    "научные знания для "
                    "добавления новых фактов."
                ),
                (
                    "Проверь субъект, "
                    "предикат, числа, диапазоны, "
                    "единицы измерения и условия."
                ),
                (
                    "Не усиливай причинность "
                    "или степень уверенности."
                ),
                (
                    "Проверь taxa и "
                    "taxon_scope."
                ),
                (
                    "Если тезис содержит "
                    "несколько независимых "
                    "утверждений, в review "
                    "оставь только одно "
                    "атомарное утверждение."
                ),
                (
                    "Изменяй только объект "
                    "review."
                ),
                (
                    "Не добавляй статус "
                    "одобрения. Окончательное "
                    "одобрение выполняется "
                    "человеком через Web UI."
                ),
            ],

            # ------------------------------------------------
            # ЭТОТ БЛОК Я МОГУ ИСПРАВИТЬ,
            # А ПОЛЬЗОВАТЕЛЬ ЗАТЕМ ИМПОРТИРОВАТЬ.
            # ------------------------------------------------

            "review": {
                "text": (
                    candidate.text
                ),
                "taxa": (
                    candidate.taxa
                ),
                "taxon_scope": (
                    candidate
                    .taxon_scope
                    .value
                ),
                "review_note": (
                    candidate
                    .review_note
                ),
            },
        }

        content = json.dumps(
            export_data,
            ensure_ascii=False,
            indent=2,
        )

        filename = (
            "atomic-thesis-review-"
            f"{candidate.id}.json"
        )

        return Response(
            content=content,
            media_type=(
                "application/json; "
                "charset=utf-8"
            ),
            headers={
                "Content-Disposition": (
                    "attachment; "
                    f'filename="{filename}"'
                )
            },
        )

    # ========================================================
    # IMPORT EXTERNAL REVIEW
    # ========================================================

    @router.post(
        "/atomic-thesis-candidates/"
        "{candidate_id}/import"
    )
    async def import_candidate_review(
        candidate_id: str,
        request: Request,
        file: UploadFile = File(...),
    ):
        """
        Загружает исправленный review JSON только в HTML-форму.

        В Elasticsearch здесь ничего не сохраняется.
        Кандидат не одобряется, AtomicThesis не создаётся,
        embeddings не запускаются.
        """

        candidate = (
            await candidate_repository
            .get_candidate(
                candidate_id
            )
        )

        if candidate is None:
            return RedirectResponse(
                "/atomic-thesis-candidates",
                status_code=303,
            )

        try:
            raw = await file.read(
                MAX_REVIEW_FILE_SIZE
                + 1
            )

            if (
                len(raw)
                > MAX_REVIEW_FILE_SIZE
            ):
                raise ValueError(
                    "Файл слишком большой. "
                    "Максимум 1 МБ."
                )

            try:
                text = raw.decode(
                    "utf-8-sig"
                )
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "Файл должен быть "
                    "UTF-8 JSON."
                ) from exc

            try:
                data = json.loads(
                    text
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Файл содержит "
                    "некорректный JSON."
                ) from exc

            try:
                review_file = (
                    AtomicThesisReviewImport
                    .model_validate(
                        data
                    )
                )
            except ValidationError as exc:
                raise ValueError(
                    "Файл не соответствует "
                    "формату "
                    "atomic_thesis_review_v1:\\n"
                    f"{exc}"
                ) from exc

            if (
                review_file.candidate_id
                != candidate.id
            ):
                raise ValueError(
                    "candidate_id в JSON "
                    "не соответствует "
                    "открытому кандидату."
                )

            if (
                review_file
                .source_semantic_unit_id
                !=
                candidate
                .source_semantic_unit_id
            ):
                raise ValueError(
                    "source_semantic_unit_id "
                    "в JSON не соответствует "
                    "этому кандидату."
                )

            review = (
                review_file.review
            )

            # ------------------------------------------------
            # ВАЖНО:
            # не вызываем moderation_service.edit().
            #
            # Исправления существуют только в отрисованной
            # HTML-форме, пока пользователь вручную не нажмёт
            # "Сохранить правки" или
            # "Одобрить и создать embeddings".
            # ------------------------------------------------

            return await render_detail(
                request=request,
                candidate_id=(
                    candidate_id
                ),
                success=(
                    "Исправления из JSON "
                    "загружены в форму. "
                    "Они ещё НЕ сохранены, "
                    "тезис НЕ одобрен, "
                    "embeddings НЕ созданы."
                ),
                form_values={
                    "text": (
                        review.text
                    ),
                    "taxa": (
                        review.taxa
                    ),
                    "taxon_scope": (
                        review.taxon_scope
                    ),
                    "review_note": (
                        review.review_note
                        or ""
                    ),
                },
            )

        except Exception as exc:
            return await render_detail(
                request=request,
                candidate_id=(
                    candidate_id
                ),
                error=str(exc),
            )

        finally:
            await file.close()

    # ========================================================
    # DETAIL
    # ========================================================

    @router.get(
        "/atomic-thesis-candidates/"
        "{candidate_id}"
    )
    async def candidate_detail(
        candidate_id: str,
        request: Request,
    ):
        return await render_detail(
            request=request,
            candidate_id=(
                candidate_id
            ),
        )

    # ========================================================
    # SAVE
    # ========================================================

    @router.post(
        "/atomic-thesis-candidates/"
        "{candidate_id}/save"
    )
    async def save_candidate(
        candidate_id: str,
        request: Request,
    ):
        form = await request.form()

        try:
            await moderation_service.edit(
                candidate_id=(
                    candidate_id
                ),
                text=str(
                    form[
                        "text"
                    ]
                ).strip(),
                taxa=[
                    str(
                        value
                    ).strip()
                    for value
                    in form.getlist(
                        "taxa"
                    )
                    if str(
                        value
                    ).strip()
                ],
                taxon_scope=(
                    TaxonScope(
                        str(
                            form[
                                "taxon_scope"
                            ]
                        )
                    )
                ),
                review_note=(
                    str(
                        form.get(
                            "review_note",
                            "",
                        )
                    ).strip()
                    or None
                ),
            )

        except Exception as exc:
            return await render_detail(
                request=request,
                candidate_id=(
                    candidate_id
                ),
                error=str(exc),
            )

        return RedirectResponse(
            (
                "/atomic-thesis-candidates/"
                f"{candidate_id}"
            ),
            status_code=303,
        )

    # ========================================================
    # APPROVE
    # ========================================================

    @router.post(
        "/atomic-thesis-candidates/"
        "{candidate_id}/approve"
    )
    async def approve_candidate(
        candidate_id: str,
        request: Request,
    ):
        form = await request.form()

        try:
            result = (
                await moderation_service
                .approve(
                    candidate_id=(
                        candidate_id
                    ),
                    text=str(
                        form[
                            "text"
                        ]
                    ).strip(),
                    taxa=[
                        str(
                            value
                        ).strip()
                        for value
                        in form.getlist(
                            "taxa"
                        )
                        if str(
                            value
                        ).strip()
                    ],
                    taxon_scope=(
                        TaxonScope(
                            str(
                                form[
                                    "taxon_scope"
                                ]
                            )
                        )
                    ),
                    review_note=(
                        str(
                            form.get(
                                "review_note",
                                "",
                            )
                        ).strip()
                        or None
                    ),
                )
            )

        except Exception as exc:
            return await render_detail(
                request=request,
                candidate_id=(
                    candidate_id
                ),
                error=str(exc),
            )

        return RedirectResponse(
            (
                "/atomic-theses/"
                f"{result.atomic_thesis_id}"
            ),
            status_code=303,
        )

    # ========================================================
    # REJECT
    # ========================================================

    @router.post(
        "/atomic-thesis-candidates/"
        "{candidate_id}/reject"
    )
    async def reject_candidate(
        candidate_id: str,
        request: Request,
    ):
        form = await request.form()

        await moderation_service.reject(
            candidate_id=(
                candidate_id
            ),
            review_note=(
                str(
                    form.get(
                        "review_note",
                        "",
                    )
                ).strip()
                or None
            ),
        )

        return RedirectResponse(
            "/atomic-thesis-candidates",
            status_code=303,
        )

    return router
