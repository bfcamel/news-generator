from datetime import datetime, timezone
import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_unit_text(text: str) -> str:
    """
    Нормализует текст перед вычислением hash.

    Убирает различия, вызванные:
    - регистром;
    - лишними пробелами;
    - переносами строк;
    - табуляцией.
    """
    return re.sub(r"\s+", " ", text).strip().casefold()


def make_text_hash(text: str) -> str:
    """Создаёт стабильный SHA-256 hash текста SemanticUnit."""

    normalized = normalize_unit_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


class SemanticUnit(BaseModel):
    """
    Законченный по смыслу фрагмент исходного документа.

    SemanticUnit принадлежит одному SourceDocument и должна
    содержать достаточно контекста, чтобы её можно было понять
    без чтения соседних фрагментов.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Внутренний уникальный идентификатор смысловой единицы.
    id: NonEmptyStr

    taxa: list[NonEmptyStr] = Field(
        default_factory=list
    )

    # Ссылка на SourceDocument.id.
    source_document_id: NonEmptyStr

    # Текст смысловой единицы.
    text: NonEmptyStr

    # Hash нормализованного текста.
    #
    # Используется вместе с source_document_id
    # для обнаружения дубликатов внутри одного источника.
    #
    # Вычисляется автоматически.
    text_hash: NonEmptyStr

    # Порядковый номер смысловой единицы внутри документа.
    # Начинаем с 0.
    position: int = Field(..., ge=0)

    # Раздел документа, из которого извлечена единица.
    #
    # Например:
    # "Introduction"
    # "2.3 Water metabolism"
    section_title: NonEmptyStr | None = None

    # Позиция текста относительно full_text исходного документа.
    #
    # char_start включительно.
    # char_end исключительно.
    #
    # То есть:
    # full_text[char_start:char_end]
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    # Страницы оригинального источника, если они известны.
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    # Вектор смысловой единицы для семантического поиска.
    #
    # Пока может быть None.
    # Позже будет заполнен отдельным процессом генерации embeddings.
    doc_embedding: list[float] | None = None

    # Модель, которой был построен embedding.
    #
    # Например:
    # "text-search-doc/latest"
    embedding_model: NonEmptyStr | None = None

    # Дополнительные служебные данные.
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def generate_text_hash(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        text = data.get("text")

        if isinstance(text, str) and text.strip():
            data["text_hash"] = make_text_hash(text)

        return data

    @model_validator(mode="after")
    def validate_structure(self) -> "SemanticUnit":
        # char_start и char_end должны либо присутствовать вместе,
        # либо отсутствовать вместе.
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError(
                "char_start and char_end must either both be specified "
                "or both be None"
            )

        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError(
                "char_end must be greater than char_start"
            )

        # page_end без page_start неоднозначен.
        if self.page_end is not None and self.page_start is None:
            raise ValueError(
                "page_start is required when page_end is specified"
            )

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end must be greater than or equal to page_start"
            )

        # Если embedding уже есть, должна быть указана модель.
        if self.doc_embedding is not None and self.embedding_model is None:
            raise ValueError(
                "embedding_model is required when doc_embedding is specified"
            )

        # И наоборот: модель без embedding не имеет смысла.
        if self.embedding_model is not None and self.doc_embedding is None:
            raise ValueError(
                "embedding_model cannot be specified without doc_embedding"
            )

        return self