from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_fingerprint_value(value: str) -> str:
    """Нормализация строк перед вычислением fingerprint."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def make_fingerprint(data: dict[str, Any]) -> str:
    """
    Создаёт стабильный fingerprint документа.

    Приоритет:
    1. DOI
    2. ISBN для целой книги
    3. Тип документа + название + авторы + год +
       родительское издание + номер главы
    """

    doi = data.get("doi")

    if doi:
        identity = f"doi:{normalize_fingerprint_value(str(doi))}"

    else:
        document_type = data.get("document_type")

        if isinstance(document_type, StrEnum):
            document_type = document_type.value

        isbn = data.get("isbn")

        if document_type == DocumentType.BOOK.value and isbn:
            normalized_isbn = re.sub(r"[^0-9Xx]", "", str(isbn)).casefold()
            identity = f"isbn:{normalized_isbn}"

        else:
            authors = data.get("authors") or []

            normalized_authors = "|".join(
                normalize_fingerprint_value(str(author))
                for author in authors
            )

            identity = "|".join(
                [
                    normalize_fingerprint_value(
                        str(document_type or "")
                    ),
                    normalize_fingerprint_value(
                        str(data.get("title") or "")
                    ),
                    normalized_authors,
                    str(data.get("year") or ""),
                    normalize_fingerprint_value(
                        str(data.get("container_title") or "")
                    ),
                    normalize_fingerprint_value(
                        str(data.get("chapter_number") or "")
                    ),
                ]
            )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


class DocumentType(StrEnum):
    """Тип документа, который непосредственно загружен в базу знаний."""

    ARTICLE = "article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    REPORT = "report"


class ContainerType(StrEnum):
    """Тип издания или ресурса, частью которого является документ."""

    JOURNAL = "journal"
    BOOK = "book"
    PROCEEDINGS = "proceedings"
    WEBSITE = "website"
    OTHER = "other"


class SourceDocument(BaseModel):
    """Исходный документ базы знаний."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Внутренний уникальный идентификатор документа.
    id: NonEmptyStr

    primary_taxon: NonEmptyStr | None = None

    # Что именно загружено:
    # статья, книга, глава книги или отчёт.
    document_type: DocumentType

    # Название статьи, книги, главы или отчёта.
    title: NonEmptyStr

    # Авторы непосредственно этого документа.
    authors: list[NonEmptyStr] = Field(default_factory=list)

    # Редакторы книги, сборника и т.п.
    editors: list[NonEmptyStr] = Field(default_factory=list)

    # Год публикации.
    year: int | None = Field(
        default=None,
        ge=1800,
        le=2200,
    )

    # Тип публикации, внутри которой находится документ.
    #
    # Примеры:
    # статья -> journal
    # глава -> book
    # доклад -> proceedings
    #
    # Для целой книги обычно None.
    container_type: ContainerType | None = None

    # Название родительской публикации.
    #
    # Например:
    # "Journal of Camelid Science"
    # "The Camel: Biology and Management"
    # "Proceedings of the International Camel Conference"
    container_title: NonEmptyStr | None = None

    # Издательство книги, сборника, отчёта и т.п.
    publisher: NonEmptyStr | None = None

    # Номер главы.
    #
    # Оставляем строкой, поскольку встречаются не только "1", "2", "3",
    # но и "10A", "Appendix A" и другие варианты.
    chapter_number: NonEmptyStr | None = None

    # Страницы документа в исходном издании.
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    # Стандартные идентификаторы источника.
    doi: NonEmptyStr | None = None
    isbn: NonEmptyStr | None = None

    # URL исходного документа или страницы источника.
    url: NonEmptyStr | None = None

    # Язык документа.
    # Например: "ru", "en", "de".
    language: str = Field(
        default="ru",
        min_length=2,
        max_length=10,
    )

    # Полный очищенный текст статьи, книги, главы или отчёта.
    full_text: NonEmptyStr

    # Отпечаток документа для обнаружения повторной загрузки.
    # Вычисляется автоматически.
    fingerprint: NonEmptyStr

    # Дополнительные данные, для которых нет отдельного поля.
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def generate_fingerprint(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)
        data["fingerprint"] = make_fingerprint(data)

        return data

    @model_validator(mode="after")
    def validate_structure(self) -> "SourceDocument":
        # page_end без page_start неоднозначен.
        if self.page_end is not None and self.page_start is None:
            raise ValueError(
                "page_start is required when page_end is specified"
            )

        # Конец диапазона страниц не может находиться раньше начала.
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must be greater than or equal to page_start"
            )

        # Если задан контейнер, должно быть указано его название.
        if self.container_type is not None and self.container_title is None:
            raise ValueError(
                "container_title is required when container_type is specified"
            )

        # Название контейнера без его типа неоднозначно.
        if self.container_title is not None and self.container_type is None:
            raise ValueError(
                "container_type is required when container_title is specified"
            )

        # Целая книга сама является источником
        # и не находится внутри другой книги.
        if self.document_type == DocumentType.BOOK:
            if self.container_type is not None:
                raise ValueError(
                    "A whole book must not have container_type"
                )

            if self.container_title is not None:
                raise ValueError(
                    "A whole book must not have container_title"
                )

            if self.chapter_number is not None:
                raise ValueError(
                    "A whole book must not have chapter_number"
                )

        # Глава обязательно должна ссылаться на книгу.
        if self.document_type == DocumentType.BOOK_CHAPTER:
            if self.container_type != ContainerType.BOOK:
                raise ValueError(
                    "A book chapter must have container_type='book'"
                )

            if self.container_title is None:
                raise ValueError(
                    "A book chapter must have container_title"
                )

        # Если документ обозначен статьёй, BOOK должен оформляться
        # как BOOK_CHAPTER, а не ARTICLE.
        if (
            self.document_type == DocumentType.ARTICLE
            and self.container_type == ContainerType.BOOK
        ):
            raise ValueError(
                "An article cannot have container_type='book'; "
                "use document_type='book_chapter'"
            )

        # Номер главы допустим только для главы книги.
        if (
            self.chapter_number is not None
            and self.document_type != DocumentType.BOOK_CHAPTER
        ):
            raise ValueError(
                "chapter_number is only allowed for book chapters"
            )

        return self