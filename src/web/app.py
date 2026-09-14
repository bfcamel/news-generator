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

from src.domain.atomic_thesis import (
    AtomicThesis,
    TaxonScope,
    ThesisStatus,
    calculate_text_hash,
)
from src.domain.semantic_unit import SemanticUnit
from src.domain.source_document import (
    ContainerType,
    DocumentType,
    SourceDocument,
)
from src.infrastructure.elasticsearch.atomic_theses_index import (
    ensure_atomic_theses_index,
)
from src.infrastructure.elasticsearch.client import es
from src.infrastructure.elasticsearch.indices import ensure_indices
from src.infrastructure.embeddings import (
    EmbeddingSettings,
    YandexEmbeddingClient,
)
from src.repositories.atomic_thesis_repository import (
    AtomicThesisAlreadyExistsError,
    AtomicThesisRepository,
    MissingSemanticUnitsError,
)
from src.repositories.embedding_repository import EmbeddingRepository
from src.repositories.semantic_unit_repository import SemanticUnitRepository
from src.repositories.source_document_repository import SourceDocumentRepository
from src.services.atomic_thesis_service import AtomicThesisService
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

atomic_thesis_repository = (
    AtomicThesisRepository(
        es
    )
)

atomic_thesis_service = (
    AtomicThesisService(
        repository=atomic_thesis_repository,
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


THESIS_STATUS_LABELS = {
    ThesisStatus.PENDING_REVIEW: "На проверке",
    ThesisStatus.ACTIVE: "Активен",
    ThesisStatus.DISABLED: "Отключён",
}


TAXON_SCOPE_LABELS = {
    TaxonScope.SPECIES: "Один вид",
    TaxonScope.MULTI_SPECIES: "Несколько видов",
    TaxonScope.GENUS: "Род",
    TaxonScope.FAMILY: "Семейство",
    TaxonScope.UNSPECIFIED: "Не указан",
}


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    logger.info(
        "Starting News Generator Admin"
    )

    try:
        await ensure_indices()
        await ensure_atomic_theses_index()

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



async def get_semantic_units_by_ids(
    unit_ids: list[str],
) -> list[SemanticUnit]:
    """
    Возвращает SemanticUnit в том же порядке,
    в котором пришли unit_ids.
    """

    unique_ids = list(
        dict.fromkeys(
            unit_ids
        )
    )

    if not unique_ids:
        return []

    response = await es.mget(
        index="semantic_units",
        ids=unique_ids,
    )

    by_id: dict[
        str,
        SemanticUnit,
    ] = {}

    for doc in response[
        "docs"
    ]:
        if not doc.get(
            "found",
            False,
        ):
            continue

        unit = SemanticUnit.model_validate(
            doc[
                "_source"
            ]
        )

        by_id[
            unit.id
        ] = unit

    return [
        by_id[
            unit_id
        ]
        for unit_id in unique_ids
        if unit_id in by_id
    ]


async def get_source_map() -> dict[
    str,
    SourceDocument,
]:
    documents = (
        await source_repository.list_all()
    )

    return {
        document.id: document
        for document in documents
    }


async def build_thesis_support_rows(
    thesis: AtomicThesis,
) -> list[dict[str, Any]]:
    units = await get_semantic_units_by_ids(
        thesis.semantic_unit_ids
    )

    source_map = await get_source_map()

    return [
        {
            "unit": unit,
            "source": source_map.get(
                unit.source_document_id
            ),
        }
        for unit in units
    ]


async def build_atomic_thesis_list_rows(
    theses: list[AtomicThesis],
) -> list[dict[str, Any]]:
    all_unit_ids = list(
        dict.fromkeys(
            unit_id
            for thesis in theses
            for unit_id in thesis.semantic_unit_ids
        )
    )

    unit_to_source: dict[
        str,
        str,
    ] = {}

    if all_unit_ids:
        response = await es.mget(
            index="semantic_units",
            ids=all_unit_ids,
            source_includes=[
                "source_document_id"
            ],
        )

        for doc in response[
            "docs"
        ]:
            if not doc.get(
                "found",
                False,
            ):
                continue

            source = doc.get(
                "_source"
            ) or {}

            source_document_id = (
                source.get(
                    "source_document_id"
                )
            )

            if source_document_id:
                unit_to_source[
                    str(
                        doc[
                            "_id"
                        ]
                    )
                ] = str(
                    source_document_id
                )

    rows: list[
        dict[str, Any]
    ] = []

    for thesis in theses:
        source_document_ids = {
            unit_to_source[
                unit_id
            ]
            for unit_id in thesis.semantic_unit_ids
            if unit_id in unit_to_source
        }

        rows.append(
            {
                "thesis": thesis,
                "semantic_unit_count": len(
                    thesis.semantic_unit_ids
                ),
                "source_document_count": len(
                    source_document_ids
                ),
            }
        )

    return rows


def validate_taxon_scope_input(
    *,
    taxa: list[str],
    taxon_scope: TaxonScope,
) -> None:
    if (
        taxon_scope
        == TaxonScope.SPECIES
        and len(
            taxa
        )
        != 1
    ):
        raise ValueError(
            "Для scope=species нужно указать "
            "ровно один таксон"
        )

    if (
        taxon_scope
        == TaxonScope.MULTI_SPECIES
        and len(
            taxa
        )
        < 2
    ):
        raise ValueError(
            "Для scope=multi_species нужно указать "
            "минимум два таксона"
        )

    if (
        taxon_scope
        in {
            TaxonScope.GENUS,
            TaxonScope.FAMILY,
        }
        and not taxa
    ):
        raise ValueError(
            "Для genus/family нужно указать таксон"
        )


async def search_semantic_units_for_thesis(
    *,
    text: str,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Candidate retrieval для ручного создания тезиса.

    Русский тезис -> query embedding ->
    ближайшие SemanticUnit.doc_embedding.

    Score используется только для ранжирования кандидатов,
    а не как доказательство поддержки тезиса.
    """

    query_embedding = (
        await embedding_client.embed_query(
            text
        )
    )

    response = await es.search(
        index="semantic_units",
        size=limit,
        knn={
            "field": "doc_embedding",
            "query_vector": (
                query_embedding.vector
            ),
            "k": limit,
            "num_candidates": max(
                100,
                limit * 5,
            ),
        },
        source_excludes=[
            "doc_embedding",
            "embedding_model",
        ],
    )

    source_map = await get_source_map()

    candidates: list[
        dict[str, Any]
    ] = []

    for hit in response[
        "hits"
    ][
        "hits"
    ]:
        unit = SemanticUnit.model_validate(
            hit[
                "_source"
            ]
        )

        candidates.append(
            {
                "unit": unit,
                "score": float(
                    hit[
                        "_score"
                    ]
                ),
                "source": source_map.get(
                    unit.source_document_id
                ),
            }
        )

    return candidates


def build_thesis_review_payload(
    *,
    thesis: AtomicThesis,
    support_rows: list[
        dict[str, Any]
    ],
) -> str:
    """
    Текст, который можно одним кликом скопировать
    и прислать в ChatGPT перед публикацией.
    """

    independent_source_ids = {
        row[
            "unit"
        ].source_document_id
        for row in support_rows
    }

    lines = [
        "ПРОВЕРКА АТОМАРНОГО ТЕЗИСА "
        "ПЕРЕД ПУБЛИКАЦИЕЙ",
        "",
        f"ID: {thesis.id}",
        f"Статус: {thesis.status.value}",
        (
            "Таксономический scope: "
            f"{thesis.taxon_scope.value}"
        ),
        (
            "Таксоны: "
            + (
                ", ".join(
                    thesis.taxa
                )
                if thesis.taxa
                else "не указаны"
            )
        ),
        "",
        "ТЕЗИС:",
        thesis.text,
        "",
        (
            "Подтверждающих Semantic Units: "
            f"{len(support_rows)}"
        ),
        (
            "Независимых источников: "
            f"{len(independent_source_ids)}"
        ),
        "",
        "ПОДТВЕРЖДЕНИЯ:",
    ]

    for index, row in enumerate(
        support_rows,
        start=1,
    ):
        unit: SemanticUnit = row[
            "unit"
        ]
        source: SourceDocument | None = row[
            "source"
        ]

        lines.extend(
            [
                "",
                f"[{index}] SemanticUnit {unit.id}",
            ]
        )

        if source is not None:
            lines.append(
                f"Источник: {source.title}"
            )

            if source.authors:
                lines.append(
                    "Авторы: "
                    + ", ".join(
                        source.authors
                    )
                )

            if source.year is not None:
                lines.append(
                    f"Год: {source.year}"
                )

            if source.doi:
                lines.append(
                    f"DOI: {source.doi}"
                )

            if source.url:
                lines.append(
                    f"URL: {source.url}"
                )

        if unit.section_title:
            lines.append(
                "Раздел: "
                f"{unit.section_title}"
            )

        if unit.page_start is not None:
            page_value = str(
                unit.page_start
            )

            if (
                unit.page_end is not None
                and unit.page_end
                != unit.page_start
            ):
                page_value += (
                    f"–{unit.page_end}"
                )

            lines.append(
                f"Страница: {page_value}"
            )

        if unit.taxa:
            lines.append(
                "Taxa SemanticUnit: "
                + ", ".join(
                    unit.taxa
                )
            )

        lines.extend(
            [
                "Текст SemanticUnit:",
                unit.text,
            ]
        )

    lines.extend(
        [
            "",
            "ЗАДАЧА ПРОВЕРКИ:",
            (
                "Проверь, действительно ли каждый "
                "Semantic Unit подтверждает тезис. "
                "Отдельно проверь субъект, предикат, "
                "числа и диапазоны, направление "
                "сравнения, таксон, ограничения, "
                "неопределённость и отсутствие "
                "необоснованного обобщения. "
                "Если формулировка тезиса требует "
                "исправления, предложи точную "
                "русскую формулировку."
            ),
        ]
    )

    return "\\n".join(
        lines
    )


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
# ATOMIC THESES
# ============================================================


@app.get(
    "/atomic-theses"
)
async def atomic_theses_page(
    request: Request,
):
    theses = (
        await atomic_thesis_repository.list_all(
            size=2000
        )
    )

    rows = (
        await build_atomic_thesis_list_rows(
            theses
        )
    )

    return templates.TemplateResponse(
        request=request,
        name="atomic_theses.html",
        context={
            "rows": rows,
            "total_theses": len(
                theses
            ),
            "taxon_scopes": TaxonScope,
            "taxon_scope_labels": (
                TAXON_SCOPE_LABELS
            ),
            "thesis_status_labels": (
                THESIS_STATUS_LABELS
            ),
            "taxon_suggestions": (
                TAXON_SUGGESTIONS
            ),
            "error": None,
            "success": None,
        },
    )


@app.post(
    "/atomic-theses/preview"
)
async def atomic_thesis_preview(
    request: Request,
):
    form = await request.form()

    try:
        text = str(
            form[
                "text"
            ]
        ).strip()

        if not text:
            raise ValueError(
                "Текст тезиса не может быть пустым"
            )

        taxa: list[str] = []
        seen_taxa: set[str] = set()

        for raw_taxon in form.getlist(
            "taxa"
        ):
            for taxon in parse_taxa(
                str(
                    raw_taxon
                )
            ):
                if taxon in seen_taxa:
                    continue

                seen_taxa.add(
                    taxon
                )
                taxa.append(
                    taxon
                )

        taxon_scope = TaxonScope(
            str(
                form.get(
                    "taxon_scope",
                    TaxonScope.UNSPECIFIED.value,
                )
            )
        )

        validate_taxon_scope_input(
            taxa=taxa,
            taxon_scope=taxon_scope,
        )

        candidate_count = int(
            form.get(
                "candidate_count",
                "20",
            )
        )

        candidate_count = min(
            max(
                candidate_count,
                5,
            ),
            50,
        )

        candidates = (
            await search_semantic_units_for_thesis(
                text=text,
                limit=candidate_count,
            )
        )

        exact_existing = (
            await atomic_thesis_repository
            .find_by_text_hash(
                calculate_text_hash(
                    text
                )
            )
        )

        logger.info(
            "AtomicThesis preview prepared: "
            "text_hash=%s candidates=%d "
            "exact_existing=%s",
            calculate_text_hash(
                text
            ),
            len(
                candidates
            ),
            (
                exact_existing.id
                if exact_existing
                is not None
                else None
            ),
        )

        return templates.TemplateResponse(
            request=request,
            name="atomic_thesis_preview.html",
            context={
                "text": text,
                "taxa": taxa,
                "taxa_string": (
                    ", ".join(
                        taxa
                    )
                ),
                "taxon_scope": (
                    taxon_scope
                ),
                "taxon_scope_labels": (
                    TAXON_SCOPE_LABELS
                ),
                "candidates": candidates,
                "candidate_count": (
                    candidate_count
                ),
                "exact_existing": (
                    exact_existing
                ),
                "error": None,
            },
        )

    except (
        ValueError,
        KeyError,
    ) as exc:
        logger.warning(
            "Failed to prepare AtomicThesis "
            "preview: %s",
            exc,
        )

        theses = (
            await atomic_thesis_repository
            .list_all(
                size=2000
            )
        )

        rows = (
            await build_atomic_thesis_list_rows(
                theses
            )
        )

        return templates.TemplateResponse(
            request=request,
            name="atomic_theses.html",
            context={
                "rows": rows,
                "total_theses": len(
                    theses
                ),
                "taxon_scopes": TaxonScope,
                "taxon_scope_labels": (
                    TAXON_SCOPE_LABELS
                ),
                "thesis_status_labels": (
                    THESIS_STATUS_LABELS
                ),
                "taxon_suggestions": (
                    TAXON_SUGGESTIONS
                ),
                "error": str(
                    exc
                ),
                "success": None,
            },
            status_code=400,
        )

    except Exception as exc:
        logger.exception(
            "Unexpected AtomicThesis preview "
            "error"
        )

        theses = (
            await atomic_thesis_repository
            .list_all(
                size=2000
            )
        )

        rows = (
            await build_atomic_thesis_list_rows(
                theses
            )
        )

        return templates.TemplateResponse(
            request=request,
            name="atomic_theses.html",
            context={
                "rows": rows,
                "total_theses": len(
                    theses
                ),
                "taxon_scopes": TaxonScope,
                "taxon_scope_labels": (
                    TAXON_SCOPE_LABELS
                ),
                "thesis_status_labels": (
                    THESIS_STATUS_LABELS
                ),
                "taxon_suggestions": (
                    TAXON_SUGGESTIONS
                ),
                "error": (
                    "Не удалось подобрать "
                    "Semantic Units: "
                    f"{exc}"
                ),
                "success": None,
            },
            status_code=500,
        )


@app.post(
    "/atomic-theses/commit"
)
async def atomic_thesis_commit(
    request: Request,
):
    form = await request.form(
        max_fields=1000,
    )

    try:
        text = str(
            form[
                "text"
            ]
        ).strip()

        if not text:
            raise ValueError(
                "Текст тезиса не может быть пустым"
            )

        taxa = parse_taxa(
            form.get(
                "taxa"
            )
        )

        taxon_scope = TaxonScope(
            str(
                form.get(
                    "taxon_scope",
                    TaxonScope.UNSPECIFIED.value,
                )
            )
        )

        validate_taxon_scope_input(
            taxa=taxa,
            taxon_scope=taxon_scope,
        )

        semantic_unit_ids = [
            str(
                value
            )
            for value in form.getlist(
                "semantic_unit_ids"
            )
            if str(
                value
            ).strip()
        ]

        semantic_unit_ids = list(
            dict.fromkeys(
                semantic_unit_ids
            )
        )

        if not semantic_unit_ids:
            raise ValueError(
                "Нужно оставить хотя бы один "
                "подтверждающий Semantic Unit"
            )

        # Всегда пересчитываем exact duplicate на сервере.
        # Hidden-поля из preview не считаются источником истины.
        exact_existing = (
            await atomic_thesis_repository
            .find_by_text_hash(
                calculate_text_hash(
                    text
                )
            )
        )

        if exact_existing is not None:
            added_support = 0

            for semantic_unit_id in (
                semantic_unit_ids
            ):
                before = set(
                    exact_existing
                    .semantic_unit_ids
                )

                exact_existing = (
                    await atomic_thesis_service
                    .add_support(
                        thesis_id=(
                            exact_existing.id
                        ),
                        semantic_unit_id=(
                            semantic_unit_id
                        ),
                    )
                )

                if (
                    semantic_unit_id
                    not in before
                ):
                    added_support += 1

            logger.info(
                "AtomicThesis exact duplicate: "
                "id=%s added_support=%d",
                exact_existing.id,
                added_support,
            )

            return RedirectResponse(
                (
                    "/atomic-theses/"
                    f"{exact_existing.id}"
                    "?support_added="
                    f"{added_support}"
                ),
                status_code=303,
            )

        result = (
            await atomic_thesis_service
            .create_from_data(
                text=text,
                semantic_unit_ids=(
                    semantic_unit_ids
                ),
                taxa=taxa,
                taxon_scope=(
                    taxon_scope
                ),
                status=(
                    ThesisStatus
                    .PENDING_REVIEW
                ),
                metadata={
                    "created_via": "web",
                    "support_selection": (
                        "manual_from_vector_candidates"
                    ),
                },
                generate_embeddings=True,
            )
        )

        logger.info(
            "AtomicThesis created from web: "
            "id=%s support_units=%d "
            "query_embedding=%s "
            "doc_embedding=%s",
            result.thesis.id,
            len(
                result.thesis
                .semantic_unit_ids
            ),
            (
                result
                .query_embedding_created
            ),
            (
                result
                .doc_embedding_created
            ),
        )

        return RedirectResponse(
            (
                "/atomic-theses/"
                f"{result.thesis.id}"
                "?created=1"
            ),
            status_code=303,
        )

    except (
        ValidationError,
        ValueError,
        KeyError,
        MissingSemanticUnitsError,
        AtomicThesisAlreadyExistsError,
    ) as exc:
        logger.warning(
            "Failed to create AtomicThesis "
            "from web: %s",
            exc,
        )

        theses = (
            await atomic_thesis_repository
            .list_all(
                size=2000
            )
        )

        rows = (
            await build_atomic_thesis_list_rows(
                theses
            )
        )

        return templates.TemplateResponse(
            request=request,
            name="atomic_theses.html",
            context={
                "rows": rows,
                "total_theses": len(
                    theses
                ),
                "taxon_scopes": TaxonScope,
                "taxon_scope_labels": (
                    TAXON_SCOPE_LABELS
                ),
                "thesis_status_labels": (
                    THESIS_STATUS_LABELS
                ),
                "taxon_suggestions": (
                    TAXON_SUGGESTIONS
                ),
                "error": str(
                    exc
                ),
                "success": None,
            },
            status_code=400,
        )


@app.get(
    "/atomic-theses/{thesis_id}"
)
async def atomic_thesis_detail(
    thesis_id: str,
    request: Request,
):
    thesis = (
        await atomic_thesis_repository.get(
            thesis_id
        )
    )

    if thesis is None:
        return RedirectResponse(
            "/atomic-theses",
            status_code=303,
        )

    support_rows = (
        await build_thesis_support_rows(
            thesis
        )
    )

    source_ids = {
        row[
            "unit"
        ].source_document_id
        for row in support_rows
    }

    review_payload = (
        build_thesis_review_payload(
            thesis=thesis,
            support_rows=support_rows,
        )
    )

    created = (
        request.query_params.get(
            "created"
        )
        == "1"
    )

    support_added = int(
        request.query_params.get(
            "support_added",
            "0",
        )
    )

    success: str | None = None

    if created:
        success = (
            "Тезис создан и связан с "
            f"{len(support_rows)} Semantic Units."
        )

    elif support_added:
        success = (
            "К существующему тезису добавлено "
            f"новых подтверждений: {support_added}."
        )

    return templates.TemplateResponse(
        request=request,
        name="atomic_thesis_detail.html",
        context={
            "thesis": thesis,
            "support_rows": support_rows,
            "semantic_unit_count": len(
                support_rows
            ),
            "source_document_count": len(
                source_ids
            ),
            "review_payload": (
                review_payload
            ),
            "thesis_status_labels": (
                THESIS_STATUS_LABELS
            ),
            "taxon_scope_labels": (
                TAXON_SCOPE_LABELS
            ),
            "statuses": ThesisStatus,
            "success": success,
            "error": None,
        },
    )


@app.post(
    "/atomic-theses/{thesis_id}/status"
)
async def atomic_thesis_update_status(
    thesis_id: str,
    request: Request,
):
    form = await request.form()

    try:
        status = ThesisStatus(
            str(
                form[
                    "status"
                ]
            )
        )

        await atomic_thesis_repository.set_status(
            thesis_id=thesis_id,
            status=status,
        )

        logger.info(
            "AtomicThesis status updated: "
            "id=%s status=%s",
            thesis_id,
            status.value,
        )

    except (
        ValueError,
        KeyError,
    ) as exc:
        logger.warning(
            "Failed to update AtomicThesis "
            "status: id=%s error=%s",
            thesis_id,
            exc,
        )

    return RedirectResponse(
        f"/atomic-theses/{thesis_id}",
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
