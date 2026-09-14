import json
import logging

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from src.domain.semantic_unit import SemanticUnit
from src.domain.source_document import (
    ContainerType,
    DocumentType,
    SourceDocument,
)
from src.infrastructure.elasticsearch.client import es
from src.infrastructure.elasticsearch.indices import ensure_indices
from src.repositories.semantic_unit_repository import SemanticUnitRepository
from src.repositories.source_document_repository import SourceDocumentRepository


BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(
        "news_generator.web"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            formatter
        )

        file_handler = RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )
        logger.addHandler(
            file_handler
        )

    logging.getLogger(
        "elastic_transport"
    ).setLevel(
        logging.WARNING
    )

    logging.getLogger(
        "elasticsearch"
    ).setLevel(
        logging.WARNING
    )

    return logger


logger = configure_logging()


templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)

source_repository = (
    SourceDocumentRepository()
)

semantic_unit_repository = (
    SemanticUnitRepository()
)


LANGUAGES = {
    "ru": "Русский",
    "en": "English",
}


# Это только подсказки для UI.
# Поле taxa не ограничено этим списком:
# при необходимости можно вводить новый валидный таксон.
TAXON_SUGGESTIONS = [
    "Camelidae",
    "Camelini",
    "Camelus",
    "Camelus dromedarius",
    "Camelus bactrianus",
    "Camelus ferus",
    "Camelus thomasi",
    "Camelus knoblochi",
    "Paracamelus",
    "Lama",
    "Vicugna",
]


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    logger.info(
        "Starting News Generator Admin"
    )

    try:
        await ensure_indices()

        logger.info(
            "Elasticsearch indices are ready"
        )

        yield

    except Exception:
        logger.exception(
            "Application lifespan error"
        )
        raise

    finally:
        logger.info(
            "Stopping News Generator Admin"
        )

        await es.close()

        logger.info(
            "Elasticsearch connection closed"
        )


app = FastAPI(
    title="News Generator Admin",
    lifespan=lifespan,
)


def empty_to_none(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    value = value.strip()

    return value or None


def optional_int(
    value: str | None,
) -> int | None:
    value = empty_to_none(
        value
    )

    if value is None:
        return None

    return int(
        value
    )


def split_people(
    value: str | None,
) -> list[str]:
    value = empty_to_none(
        value
    )

    if value is None:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def parse_taxa(
    value: str | None,
) -> list[str]:
    """
    UI хранит несколько таксонов одной строкой:

    Camelus ferus, Camelus bactrianus
    """

    if value is None:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for raw_item in value.split(","):
        item = raw_item.strip()

        if not item:
            continue

        if item in seen:
            continue

        seen.add(
            item
        )
        result.append(
            item
        )

    return result


def normalize_json_taxa(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return parse_taxa(
            value
        )

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            "taxa должен быть строкой "
            "или массивом строк"
        )

    result: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(
            item,
            str,
        ):
            raise ValueError(
                "Каждый элемент taxa "
                "должен быть строкой"
            )

        item = item.strip()

        if not item:
            continue

        if item in seen:
            continue

        seen.add(
            item
        )
        result.append(
            item
        )

    return result


async def count_units_by_source_document() -> dict[str, int]:
    response = await es.search(
        index="semantic_units",
        size=0,
        aggs={
            "by_source": {
                "terms": {
                    "field": "source_document_id",
                    "size": 10_000,
                }
            }
        },
    )

    buckets = response[
        "aggregations"
    ][
        "by_source"
    ][
        "buckets"
    ]

    return {
        bucket["key"]: bucket["doc_count"]
        for bucket in buckets
    }


async def get_semantic_units_page_data() -> dict[str, Any]:
    documents = (
        await source_repository.list_all()
    )

    unit_counts = (
        await count_units_by_source_document()
    )

    total_units = sum(
        unit_counts.values()
    )

    return {
        "documents": documents,
        "unit_counts": unit_counts,
        "total_units": total_units,
        "taxon_suggestions": TAXON_SUGGESTIONS,
    }


async def get_units_for_source(
    source_document_id: str,
) -> list[SemanticUnit]:
    response = await es.search(
        index="semantic_units",
        size=1000,
        query={
            "term": {
                "source_document_id": (
                    source_document_id
                )
            }
        },
        sort=[
            {
                "position": {
                    "order": "asc",
                }
            }
        ],
    )

    return [
        SemanticUnit.model_validate(
            hit["_source"]
        )
        for hit in response[
            "hits"
        ][
            "hits"
        ]
    ]


def parse_import_json(
    data: Any,
) -> list[dict[str, Any]]:
    """
    Поддерживает:

    [
        {
            "position": 0,
            "section_title": "...",
            "taxa": ["Camelus ferus"],
            "text": "..."
        }
    ]

    или:

    {
        "units": [...]
    }

    Если taxa отсутствует, на preview будет
    подставлен primary_taxon источника.

    Если taxa явно задан как [], пустой список
    сохраняется и primary_taxon не подставляется.
    """

    if isinstance(
        data,
        dict,
    ):
        data = data.get(
            "units"
        )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "JSON должен содержать список "
            "SemanticUnit или объект "
            "с полем 'units'"
        )

    units: list[
        dict[str, Any]
    ] = []

    for index, item in enumerate(
        data
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Элемент #{index} "
                "должен быть JSON-объектом"
            )

        text = item.get(
            "text"
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):
            raise ValueError(
                f"У элемента #{index} "
                "отсутствует text"
            )

        position = item.get(
            "position",
            index,
        )

        try:
            position = int(
                position
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Некорректный position "
                f"у элемента #{index}"
            ) from exc

        taxa_was_provided = (
            "taxa" in item
        )

        taxa = (
            normalize_json_taxa(
                item.get(
                    "taxa"
                )
            )
            if taxa_was_provided
            else None
        )

        units.append(
            {
                "position": position,
                "section_title": (
                    item.get(
                        "section_title"
                    )
                    or ""
                ),
                "page_start": (
                    item.get(
                        "page_start"
                    )
                    or ""
                ),
                "page_end": (
                    item.get(
                        "page_end"
                    )
                    or ""
                ),
                "taxa": taxa,
                "text": text.strip(),
            }
        )

    return units


@app.get("/")
async def root():
    return RedirectResponse(
        "/sources",
        status_code=303,
    )


# ============================================================
# SOURCES
# ============================================================


@app.get(
    "/sources"
)
async def sources_page(
    request: Request,
):
    documents = (
        await source_repository.list_all()
    )

    return templates.TemplateResponse(
        request=request,
        name="sources.html",
        context={
            "documents": documents,
            "document_types": DocumentType,
            "container_types": ContainerType,
            "languages": LANGUAGES,
            "taxon_suggestions": TAXON_SUGGESTIONS,
            "error": None,
            "success": None,
        },
    )


@app.post(
    "/sources"
)
async def create_source(
    request: Request,
):
    form = await request.form()

    try:
        primary_taxon = empty_to_none(
            form.get(
                "primary_taxon"
            )
        )

        if primary_taxon is None:
            raise ValueError(
                "Для источника необходимо "
                "указать primary_taxon"
            )

        document = SourceDocument(
            id=(
                f"doc_{uuid4().hex}"
            ),
            document_type=form[
                "document_type"
            ],
            title=form[
                "title"
            ],
            authors=split_people(
                form.get(
                    "authors"
                )
            ),
            editors=split_people(
                form.get(
                    "editors"
                )
            ),
            year=optional_int(
                form.get(
                    "year"
                )
            ),
            container_type=empty_to_none(
                form.get(
                    "container_type"
                )
            ),
            container_title=empty_to_none(
                form.get(
                    "container_title"
                )
            ),
            publisher=empty_to_none(
                form.get(
                    "publisher"
                )
            ),
            chapter_number=empty_to_none(
                form.get(
                    "chapter_number"
                )
            ),
            page_start=optional_int(
                form.get(
                    "page_start"
                )
            ),
            page_end=optional_int(
                form.get(
                    "page_end"
                )
            ),
            doi=empty_to_none(
                form.get(
                    "doi"
                )
            ),
            isbn=empty_to_none(
                form.get(
                    "isbn"
                )
            ),
            url=empty_to_none(
                form.get(
                    "url"
                )
            ),
            language=(
                form.get(
                    "language"
                )
                or "ru"
            ),
            primary_taxon=primary_taxon,
            full_text=form[
                "full_text"
            ],
        )

        await source_repository.create(
            document
        )

        logger.info(
            "Source document created: "
            "id=%s type=%s "
            "primary_taxon=%r title=%r",
            document.id,
            document.document_type.value,
            document.primary_taxon,
            document.title,
        )

    except (
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning(
            "Failed to create source document: %s",
            exc,
        )

        documents = (
            await source_repository.list_all()
        )

        return templates.TemplateResponse(
            request=request,
            name="sources.html",
            context={
                "documents": documents,
                "document_types": DocumentType,
                "container_types": ContainerType,
                "languages": LANGUAGES,
                "taxon_suggestions": TAXON_SUGGESTIONS,
                "error": str(
                    exc
                ),
                "success": None,
            },
            status_code=400,
        )

    return RedirectResponse(
        "/sources",
        status_code=303,
    )


# ============================================================
# SEMANTIC UNITS
# ============================================================


@app.get(
    "/semantic-units"
)
async def semantic_units_page(
    request: Request,
):
    page_data = (
        await get_semantic_units_page_data()
    )

    return templates.TemplateResponse(
        request=request,
        name="semantic_units.html",
        context={
            **page_data,
            "error": None,
            "success": None,
        },
    )


# ------------------------------------------------------------
# Ручное добавление
# ------------------------------------------------------------


@app.post(
    "/semantic-units"
)
async def create_semantic_unit(
    request: Request,
):
    form = await request.form()

    try:
        source_document_id = str(
            form[
                "source_document_id"
            ]
        )

        source = (
            await source_repository.get(
                source_document_id
            )
        )

        if source is None:
            raise ValueError(
                "Источник не найден"
            )

        taxa = parse_taxa(
            form.get(
                "taxa"
            )
        )

        unit = SemanticUnit(
            id=(
                f"unit_{uuid4().hex}"
            ),
            source_document_id=(
                source_document_id
            ),
            taxa=taxa,
            text=form[
                "text"
            ],
            position=int(
                form[
                    "position"
                ]
            ),
            section_title=empty_to_none(
                form.get(
                    "section_title"
                )
            ),
            page_start=optional_int(
                form.get(
                    "page_start"
                )
            ),
            page_end=optional_int(
                form.get(
                    "page_end"
                )
            ),
        )

        await semantic_unit_repository.create(
            unit
        )

        logger.info(
            "SemanticUnit created manually: "
            "id=%s source_document_id=%s "
            "position=%s taxa=%r section=%r",
            unit.id,
            unit.source_document_id,
            unit.position,
            unit.taxa,
            unit.section_title,
        )

    except (
        ValidationError,
        ValueError,
        KeyError,
    ) as exc:
        logger.warning(
            "Failed to create SemanticUnit manually: "
            "source_document_id=%s error=%s",
            form.get(
                "source_document_id"
            ),
            exc,
        )

        page_data = (
            await get_semantic_units_page_data()
        )

        return templates.TemplateResponse(
            request=request,
            name="semantic_units.html",
            context={
                **page_data,
                "error": str(
                    exc
                ),
                "success": None,
            },
            status_code=400,
        )

    return RedirectResponse(
        "/semantic-units",
        status_code=303,
    )


# ------------------------------------------------------------
# JSON -> PREVIEW
# ------------------------------------------------------------


@app.post(
    "/semantic-units/import/preview"
)
async def semantic_units_import_preview(
    request: Request,
):
    form = await request.form()

    source_document_id = str(
        form[
            "source_document_id"
        ]
    )

    upload = form.get(
        "json_file"
    )

    if upload is None:
        page_data = (
            await get_semantic_units_page_data()
        )

        return templates.TemplateResponse(
            request=request,
            name="semantic_units.html",
            context={
                **page_data,
                "error": (
                    "JSON-файл не выбран"
                ),
                "success": None,
            },
            status_code=400,
        )

    try:
        raw = await upload.read()

        data = json.loads(
            raw.decode(
                "utf-8"
            )
        )

        units = parse_import_json(
            data
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        logger.warning(
            "Failed to parse SemanticUnit JSON: "
            "source_document_id=%s error=%s",
            source_document_id,
            exc,
        )

        page_data = (
            await get_semantic_units_page_data()
        )

        return templates.TemplateResponse(
            request=request,
            name="semantic_units.html",
            context={
                **page_data,
                "error": (
                    f"Ошибка JSON: {exc}"
                ),
                "success": None,
            },
            status_code=400,
        )

    source = (
        await source_repository.get(
            source_document_id
        )
    )

    if source is None:
        return RedirectResponse(
            "/semantic-units",
            status_code=303,
        )

    for unit in units:
        # taxa=None значит, что поле в JSON отсутствовало.
        # taxa=[] значит, что пустой список был задан намеренно.
        if (
            unit[
                "taxa"
            ]
            is None
            and source.primary_taxon
        ):
            unit[
                "taxa"
            ] = [
                source.primary_taxon
            ]

    logger.info(
        "SemanticUnit JSON loaded: "
        "source_document_id=%s units=%d",
        source_document_id,
        len(
            units
        ),
    )

    return templates.TemplateResponse(
        request=request,
        name="semantic_units_preview.html",
        context={
            "source": source,
            "units": units,
            "taxon_suggestions": TAXON_SUGGESTIONS,
            "error": None,
        },
    )


# ------------------------------------------------------------
# PREVIEW -> ELASTICSEARCH
# ------------------------------------------------------------


@app.post(
    "/semantic-units/import/commit"
)
async def semantic_units_import_commit(
    request: Request,
):
    form = await request.form(
        max_fields=5000,
    )

    source_document_id = str(
        form[
            "source_document_id"
        ]
    )

    count = int(
        form[
            "unit_count"
        ]
    )

    logger.info(
        "Starting SemanticUnit import: "
        "source_document_id=%s received=%d",
        source_document_id,
        count,
    )

    added = 0
    skipped = 0
    errors: list[str] = []

    for index in range(
        count
    ):
        include = form.get(
            f"unit_{index}_include"
        )

        if include is None:
            skipped += 1
            continue

        try:
            unit = SemanticUnit(
                id=(
                    f"unit_{uuid4().hex}"
                ),
                source_document_id=(
                    source_document_id
                ),
                position=int(
                    form[
                        f"unit_{index}_position"
                    ]
                ),
                section_title=empty_to_none(
                    form.get(
                        f"unit_{index}_section_title"
                    )
                ),
                page_start=optional_int(
                    form.get(
                        f"unit_{index}_page_start"
                    )
                ),
                page_end=optional_int(
                    form.get(
                        f"unit_{index}_page_end"
                    )
                ),
                taxa=parse_taxa(
                    form.get(
                        f"unit_{index}_taxa"
                    )
                ),
                text=str(
                    form[
                        f"unit_{index}_text"
                    ]
                ),
            )

            await semantic_unit_repository.create(
                unit
            )

            added += 1

        except (
            ValidationError,
            ValueError,
            KeyError,
        ) as exc:
            errors.append(
                f"Unit #{index}: {exc}"
            )

            logger.warning(
                "SemanticUnit import error: "
                "source_document_id=%s "
                "form_index=%d error=%s",
                source_document_id,
                index,
                exc,
            )

    logger.info(
        "SemanticUnit import finished: "
        "source_document_id=%s "
        "total=%d added=%d skipped=%d errors=%d",
        source_document_id,
        count,
        added,
        skipped,
        len(
            errors
        ),
    )

    page_data = (
        await get_semantic_units_page_data()
    )

    return templates.TemplateResponse(
        request=request,
        name="semantic_units.html",
        context={
            **page_data,
            "error": (
                "\n".join(
                    errors
                )
                if errors
                else None
            ),
            "success": (
                f"Добавлено: {added}. "
                f"Исключено: {skipped}. "
                f"Ошибок: {len(errors)}."
            ),
        },
    )


# ============================================================
# TAXA REVIEW FOR EXISTING UNITS
# ============================================================


@app.get(
    "/semantic-units/source/{source_document_id}"
)
async def semantic_units_source_page(
    source_document_id: str,
    request: Request,
):
    source = (
        await source_repository.get(
            source_document_id
        )
    )

    if source is None:
        return RedirectResponse(
            "/semantic-units",
            status_code=303,
        )

    units = await get_units_for_source(
        source_document_id
    )

    return templates.TemplateResponse(
        request=request,
        name="semantic_units_source.html",
        context={
            "source": source,
            "units": units,
            "taxon_suggestions": TAXON_SUGGESTIONS,
            "error": None,
            "success": None,
        },
    )


@app.post(
    "/semantic-units/source/{source_document_id}/taxa"
)
async def update_semantic_units_taxa(
    source_document_id: str,
    request: Request,
):
    source = (
        await source_repository.get(
            source_document_id
        )
    )

    if source is None:
        return RedirectResponse(
            "/semantic-units",
            status_code=303,
        )

    form = await request.form(
        max_fields=5000,
    )

    current_units = (
        await get_units_for_source(
            source_document_id
        )
    )

    units_by_id = {
        unit.id: unit
        for unit in current_units
    }

    count = int(
        form.get(
            "unit_count",
            "0",
        )
    )

    updated = 0
    errors: list[str] = []

    for index in range(
        count
    ):
        unit_id = str(
            form.get(
                f"unit_{index}_id",
                "",
            )
        )

        current = units_by_id.get(
            unit_id
        )

        if current is None:
            errors.append(
                f"Unit #{index}: "
                "не найден или относится "
                "к другому источнику"
            )
            continue

        taxa = parse_taxa(
            form.get(
                f"unit_{index}_taxa"
            )
        )

        if taxa == current.taxa:
            continue

        try:
            await es.update(
                index="semantic_units",
                id=unit_id,
                doc={
                    "taxa": taxa,
                    "updated_at": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                },
            )

            updated += 1

            logger.info(
                "SemanticUnit taxa updated: "
                "id=%s source_document_id=%s "
                "old=%r new=%r",
                unit_id,
                source_document_id,
                current.taxa,
                taxa,
            )

        except Exception as exc:
            logger.exception(
                "Failed to update taxa for "
                "SemanticUnit id=%s",
                unit_id,
            )

            errors.append(
                f"Unit {unit_id}: {exc}"
            )

    if updated:
        await es.indices.refresh(
            index="semantic_units"
        )

    units = await get_units_for_source(
        source_document_id
    )

    return templates.TemplateResponse(
        request=request,
        name="semantic_units_source.html",
        context={
            "source": source,
            "units": units,
            "taxon_suggestions": TAXON_SUGGESTIONS,
            "error": (
                "\n".join(
                    errors
                )
                if errors
                else None
            ),
            "success": (
                f"Обновлено taxa: {updated}."
            ),
        },
        status_code=(
            400
            if errors
            else 200
        ),
    )
