import asyncio
import json
import logging

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
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
from src.infrastructure.embeddings import (
    EmbeddingSettings,
    YandexEmbeddingClient,
)
from src.repositories.embedding_repository import EmbeddingRepository
from src.repositories.semantic_unit_repository import SemanticUnitRepository
from src.repositories.source_document_repository import SourceDocumentRepository
from src.services.semantic_unit_service import SemanticUnitService

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


def format_seconds(
    seconds: float,
) -> str:
    total_seconds = max(
        0,
        int(
            round(
                seconds
            )
        ),
    )

    minutes, seconds = divmod(
        total_seconds,
        60,
    )

    hours, minutes = divmod(
        minutes,
        60,
    )

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def log_progress(
    *,
    stage: str,
    current: int,
    total: int,
    started_at: float,
    extra: str = "",
) -> None:
    """
    Унифицированный progress-log для долгих операций.

    Пример:
    [SemanticUnit import] 37/186 (19.9%) |
    12.4 items/s | elapsed=00:03 | eta=00:12
    """

    elapsed = max(
        perf_counter()
        - started_at,
        0.000001,
    )

    rate = (
        current / elapsed
        if current > 0
        else 0.0
    )

    percent = (
        (current / total) * 100
        if total > 0
        else 100.0
    )

    remaining = max(
        total - current,
        0,
    )

    eta = (
        remaining / rate
        if rate > 0
        else 0.0
    )

    message = (
        "[%s] %d/%d (%.1f%%) | "
        "%.2f items/s | "
        "elapsed=%s | eta=%s"
    )

    args: list[Any] = [
        stage,
        current,
        total,
        percent,
        rate,
        format_seconds(
            elapsed
        ),
        format_seconds(
            eta
        ),
    ]

    if extra:
        message += " | %s"
        args.append(
            extra
        )

    logger.info(
        message,
        *args,
    )


templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)

source_repository = (
    SourceDocumentRepository()
)

semantic_unit_repository = (
    SemanticUnitRepository()
)

embedding_settings = (
    EmbeddingSettings.from_env()
)

embedding_client = (
    YandexEmbeddingClient(
        embedding_settings
    )
)

embedding_repository = (
    EmbeddingRepository(
        es
    )
)

semantic_unit_service = (
    SemanticUnitService(
        repository=semantic_unit_repository,
        embedding_repository=embedding_repository,
        embedding_client=embedding_client,
    )
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

        try:
            await embedding_client.close()

            logger.info(
                "Yandex embedding client closed"
            )

        finally:
            await es.close()

            logger.info(
                "Elasticsearch connection closed"
            )


app = FastAPI(
    title="News Generator Admin",
    lifespan=lifespan,
)


def empty_to_none(
    value: Any,
) -> str | None:
    if value is None:
        return None

    value = str(
        value
    ).strip()

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



def normalize_string_list(
    value: Any,
    field_name: str,
) -> list[str]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        value = [
            item.strip()
            for item in value.splitlines()
            if item.strip()
        ]

    if not isinstance(
        value,
        list,
    ):
        raise ValueError(
            f"{field_name} должен быть массивом строк"
        )

    result: list[str] = []

    for item in value:
        if not isinstance(
            item,
            str,
        ):
            raise ValueError(
                f"Все элементы {field_name} "
                "должны быть строками"
            )

        item = item.strip()

        if item:
            result.append(
                item
            )

    return result


def parse_source_import_json(
    data: Any,
) -> dict[str, Any]:
    """
    Поддерживает:

    {
        "document_type": "article",
        ...
    }

    или:

    {
        "source": {
            ...
        }
    }
    """

    if (
        isinstance(data, dict)
        and "source" in data
    ):
        data = data[
            "source"
        ]

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "JSON источника должен быть объектом"
        )

    required = [
        "document_type",
        "title",
        "language",
        "primary_taxon",
        "full_text",
    ]

    for field in required:
        value = data.get(
            field
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise ValueError(
                f"Поле {field} обязательно"
            )

    metadata = data.get(
        "metadata",
        {},
    )

    if metadata is None:
        metadata = {}

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "metadata должен быть JSON-объектом"
        )

    def integer_or_none(
        name: str,
    ) -> int | None:
        value = data.get(
            name
        )

        if value in (
            None,
            "",
        ):
            return None

        try:
            return int(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} должен быть числом"
            ) from exc

    return {
        "document_type": str(
            data["document_type"]
        ).strip(),

        "title": str(
            data["title"]
        ).strip(),

        "authors": normalize_string_list(
            data.get(
                "authors"
            ),
            "authors",
        ),

        "editors": normalize_string_list(
            data.get(
                "editors"
            ),
            "editors",
        ),

        "year": integer_or_none(
            "year"
        ),

        "container_type": empty_to_none(
            data.get(
                "container_type"
            )
        ),

        "container_title": empty_to_none(
            data.get(
                "container_title"
            )
        ),

        "publisher": empty_to_none(
            data.get(
                "publisher"
            )
        ),

        "chapter_number": empty_to_none(
            data.get(
                "chapter_number"
            )
        ),

        "page_start": integer_or_none(
            "page_start"
        ),

        "page_end": integer_or_none(
            "page_end"
        ),

        "doi": empty_to_none(
            data.get(
                "doi"
            )
        ),

        "isbn": empty_to_none(
            data.get(
                "isbn"
            )
        ),

        "url": empty_to_none(
            data.get(
                "url"
            )
        ),

        "language": str(
            data["language"]
        ).strip(),

        "primary_taxon": str(
            data["primary_taxon"]
        ).strip(),

        "full_text": str(
            data["full_text"]
        ).strip(),

        "metadata": metadata,
    }

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

        create_result = (
            await semantic_unit_service.create(
                unit
            )
        )

        logger.info(
            "SemanticUnit created manually: "
            "id=%s source_document_id=%s "
            "position=%s taxa=%r section=%r "
            "embedding_created=%s",
            unit.id,
            unit.source_document_id,
            unit.position,
            unit.taxa,
            unit.section_title,
            create_result.embedding_created,
        )

        if not create_result.embedding_created:
            logger.warning(
                "SemanticUnit %s was created without "
                "DOC embedding: %s",
                unit.id,
                create_result.embedding_error,
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

    source = (
        await source_repository.get(
            source_document_id
        )
    )

    if source is None:
        page_data = (
            await get_semantic_units_page_data()
        )

        return templates.TemplateResponse(
            request=request,
            name="semantic_units.html",
            context={
                **page_data,
                "error": "Источник не найден",
                "success": None,
            },
            status_code=400,
        )

    count = int(
        form[
            "unit_count"
        ]
    )

    selected_indices = [
        index
        for index in range(
            count
        )
        if form.get(
            f"unit_{index}_include"
        )
        is not None
    ]

    selected_count = len(
        selected_indices
    )

    skipped = (
        count
        - selected_count
    )

    import_started_at = perf_counter()

    logger.info(
        "Starting SemanticUnit import: "
        "source_document_id=%s "
        "source_title=%r "
        "received=%d selected=%d skipped=%d",
        source_document_id,
        source.title,
        count,
        selected_count,
        skipped,
    )

    added = 0
    errors: list[str] = []
    created_units: list[SemanticUnit] = []

    # --------------------------------------------------------
    # ЭТАП 1: сохранение Semantic Units в Elasticsearch
    # --------------------------------------------------------

    save_started_at = perf_counter()

    logger.info(
        "[SemanticUnit save] Starting: "
        "%d selected units",
        selected_count,
    )

    for selected_number, index in enumerate(
        selected_indices,
        start=1,
    ):
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

            await semantic_unit_service.create(
                unit,
                generate_embedding=False,
            )

            created_units.append(
                unit
            )

            added += 1

            log_progress(
                stage="SemanticUnit save",
                current=selected_number,
                total=selected_count,
                started_at=save_started_at,
                extra=(
                    f"saved={added} "
                    f"errors={len(errors)} "
                    f"position={unit.position} "
                    f"id={unit.id}"
                ),
            )

        except (
            ValidationError,
            ValueError,
            KeyError,
        ) as exc:
            errors.append(
                f"Unit #{index}: {exc}"
            )

            logger.warning(
                "[SemanticUnit save] "
                "%d/%d failed | "
                "source_document_id=%s "
                "form_index=%d error=%s",
                selected_number,
                selected_count,
                source_document_id,
                index,
                exc,
            )

            log_progress(
                stage="SemanticUnit save",
                current=selected_number,
                total=selected_count,
                started_at=save_started_at,
                extra=(
                    f"saved={added} "
                    f"errors={len(errors)} "
                    f"form_index={index}"
                ),
            )

        except Exception as exc:
            errors.append(
                f"Unit #{index}: {exc}"
            )

            logger.exception(
                "[SemanticUnit save] "
                "%d/%d unexpected error | "
                "source_document_id=%s "
                "form_index=%d",
                selected_number,
                selected_count,
                source_document_id,
                index,
            )

            log_progress(
                stage="SemanticUnit save",
                current=selected_number,
                total=selected_count,
                started_at=save_started_at,
                extra=(
                    f"saved={added} "
                    f"errors={len(errors)} "
                    f"form_index={index}"
                ),
            )

    logger.info(
        "[SemanticUnit save] Finished: "
        "saved=%d/%d errors=%d duration=%s",
        added,
        selected_count,
        len(
            errors
        ),
        format_seconds(
            perf_counter()
            - save_started_at
        ),
    )

    # --------------------------------------------------------
    # ЭТАП 2: генерация DOC embeddings
    # --------------------------------------------------------

    embedding_results = []

    if created_units:
        embedding_started_at = perf_counter()
        embedding_total = len(
            created_units
        )

        logger.info(
            "[SemanticUnit embeddings] Starting: "
            "%d/%d saved units require embeddings | "
            "concurrency=%d",
            embedding_total,
            selected_count,
            embedding_settings.concurrency,
        )

        tasks = [
            asyncio.create_task(
                semantic_unit_service.generate_embedding(
                    unit
                )
            )
            for unit in created_units
        ]

        embeddings_created = 0
        embeddings_failed = 0

        for completed_number, task in enumerate(
            asyncio.as_completed(
                tasks
            ),
            start=1,
        ):
            result = await task

            embedding_results.append(
                result
            )

            if result.embedding_created:
                embeddings_created += 1
                status = "ok"
            else:
                embeddings_failed += 1
                status = (
                    "failed: "
                    f"{result.embedding_error}"
                )

            log_progress(
                stage="SemanticUnit embeddings",
                current=completed_number,
                total=embedding_total,
                started_at=embedding_started_at,
                extra=(
                    f"ok={embeddings_created} "
                    f"failed={embeddings_failed} "
                    f"id={result.unit.id} "
                    f"status={status}"
                ),
            )

        logger.info(
            "[SemanticUnit embeddings] Finished: "
            "created=%d/%d failed=%d duration=%s",
            embeddings_created,
            embedding_total,
            embeddings_failed,
            format_seconds(
                perf_counter()
                - embedding_started_at
            ),
        )

    else:
        embeddings_created = 0
        embeddings_failed = 0

        logger.info(
            "[SemanticUnit embeddings] Skipped: "
            "no SemanticUnits were created"
        )

    total_duration = (
        perf_counter()
        - import_started_at
    )

    logger.info(
        "SemanticUnit import finished: "
        "source_document_id=%s "
        "received=%d selected=%d "
        "added=%d skipped=%d "
        "errors=%d "
        "embeddings_created=%d "
        "embeddings_failed=%d "
        "duration=%s",
        source_document_id,
        count,
        selected_count,
        added,
        skipped,
        len(
            errors
        ),
        embeddings_created,
        embeddings_failed,
        format_seconds(
            total_duration
        ),
    )

    page_data = (
        await get_semantic_units_page_data()
    )

    success_parts = [
        f"Выбрано: {selected_count}.",
        f"Добавлено: {added}.",
        f"Исключено: {skipped}.",
        f"Ошибок импорта: {len(errors)}.",
        (
            "Embeddings создано: "
            f"{embeddings_created}."
        ),
    ]

    if embeddings_failed:
        success_parts.append(
            "Без embedding осталось: "
            f"{embeddings_failed}. "
            "Их можно восстановить командой "
            "`uv run python -m "
            "scripts.generate_embeddings "
            "semantic-units`."
        )

    success_parts.append(
        "Время: "
        f"{format_seconds(total_duration)}."
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
            "success": " ".join(
                success_parts
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

@app.post(
    "/sources/import/preview"
)
async def source_import_preview(
    request: Request,
):
    form = await request.form()

    upload = form.get(
        "json_file"
    )

    if upload is None:
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
                "error": "JSON-файл не выбран",
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

        source_data = (
            parse_source_import_json(
                data
            )
        )

        # Сразу прогоняем через Pydantic-модель.
        # Так ошибки увидим ещё до сохранения.
        preview_document = SourceDocument(
            id=(
                f"preview_{uuid4().hex}"
            ),
            **source_data,
        )

        source_data = (
            preview_document.model_dump(
                mode="json"
            )
        )

        # Эти поля создаются самой системой.
        source_data.pop(
            "id",
            None,
        )

        source_data.pop(
            "fingerprint",
            None,
        )

        source_data.pop(
            "created_at",
            None,
        )

        source_data.pop(
            "updated_at",
            None,
        )

        logger.info(
            "Source JSON loaded for preview: "
            "title=%r primary_taxon=%r",
            source_data[
                "title"
            ],
            source_data[
                "primary_taxon"
            ],
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        logger.warning(
            "Failed to parse source JSON: %s",
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
                "error": (
                    f"Ошибка JSON: {exc}"
                ),
                "success": None,
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        request=request,
        name="source_import_preview.html",
        context={
            "source": source_data,
            "document_types": DocumentType,
            "container_types": ContainerType,
            "languages": LANGUAGES,
            "taxon_suggestions": TAXON_SUGGESTIONS,
            "error": None,
        },
    )

@app.post(
    "/sources/import/commit"
)
async def source_import_commit(
    request: Request,
):
    form = await request.form()

    try:
        metadata_raw = str(
            form.get(
                "metadata",
                "{}",
            )
        ).strip()

        metadata = (
            json.loads(
                metadata_raw
            )
            if metadata_raw
            else {}
        )

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                "metadata должен быть JSON-объектом"
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

            authors=[
                item.strip()
                for item in str(
                    form.get(
                        "authors",
                        "",
                    )
                ).splitlines()
                if item.strip()
            ],

            editors=[
                item.strip()
                for item in str(
                    form.get(
                        "editors",
                        "",
                    )
                ).splitlines()
                if item.strip()
            ],

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

            language=form[
                "language"
            ],

            primary_taxon=form[
                "primary_taxon"
            ],

            full_text=form[
                "full_text"
            ],

            metadata=metadata,
        )

        await source_repository.create(
            document
        )

        logger.info(
            "Source document imported from JSON: "
            "id=%s title=%r primary_taxon=%r",
            document.id,
            document.title,
            document.primary_taxon,
        )

    except (
        ValidationError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning(
            "Failed to import source document: %s",
            exc,
        )

        return templates.TemplateResponse(
            request=request,
            name="source_import_preview.html",
            context={
                "source": dict(
                    form
                ),
                "document_types": DocumentType,
                "container_types": ContainerType,
                "languages": LANGUAGES,
                "taxon_suggestions": TAXON_SUGGESTIONS,
                "error": str(
                    exc
                ),
            },
            status_code=400,
        )

    return RedirectResponse(
        "/sources",
        status_code=303,
    )
