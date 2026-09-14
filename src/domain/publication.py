from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PublicationStatus(StrEnum):
    """Текущее состояние публикации."""

    # Материал сгенерирован, но ещё не прошёл проверку.
    DRAFT = "draft"

    # Материал проверен и готов к публикации.
    READY = "ready"

    # Материал успешно опубликован.
    PUBLISHED = "published"

    # Попытка публикации завершилась ошибкой.
    FAILED = "failed"


class PublicationPlatform(StrEnum):
    """Площадка, на которой публикуется материал."""

    VK = "vk"
    TELEGRAM = "telegram"
    WEBSITE = "website"
    OTHER = "other"


class Publication(BaseModel):
    """
    Сгенерированный научно-популярный материал
    и запись об использовании знаний из базы.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Внутренний уникальный идентификатор публикации.
    id: NonEmptyStr

    # Атомарные тезисы, использованные при создании материала.
    thesis_ids: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
    )

    # Конкретные смысловые единицы,
    # которые были переданы модели как фактический контекст.
    semantic_unit_ids: list[NonEmptyStr] = Field(
        default_factory=list,
    )

    # Заголовок материала, если он используется на площадке.
    title: NonEmptyStr | None = None

    # Финальный текст материала.
    text: NonEmptyStr

    # Текущее состояние материала.
    status: PublicationStatus = PublicationStatus.DRAFT

    # Площадка публикации.
    #
    # Может быть None, пока материал находится в draft.
    platform: PublicationPlatform | None = None

    # Идентификатор публикации во внешней системе.
    #
    # Например, ID записи VK.
    external_id: NonEmptyStr | None = None

    # Прямая ссылка на опубликованный материал, если она известна.
    external_url: NonEmptyStr | None = None

    # Время успешной публикации.
    published_at: datetime | None = None

    # Текст последней ошибки публикации.
    #
    # Используется при status=FAILED.
    error_message: NonEmptyStr | None = None

    # Дополнительные служебные данные.
    #
    # Например:
    # {
    #     "generation_model": "...",
    #     "prompt_version": "v3",
    #     "temperature": 0.4,
    # }
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_structure(self) -> "Publication":
        # Одни и те же тезисы не должны повторяться.
        if len(self.thesis_ids) != len(set(self.thesis_ids)):
            raise ValueError(
                "thesis_ids must contain unique values"
            )

        # Аналогично для SemanticUnit.
        if len(self.semantic_unit_ids) != len(set(self.semantic_unit_ids)):
            raise ValueError(
                "semantic_unit_ids must contain unique values"
            )

        # Опубликованный материал обязательно должен знать,
        # где и когда он был опубликован.
        if self.status == PublicationStatus.PUBLISHED:
            if self.platform is None:
                raise ValueError(
                    "platform is required for a published publication"
                )

            if self.external_id is None:
                raise ValueError(
                    "external_id is required for a published publication"
                )

            if self.published_at is None:
                raise ValueError(
                    "published_at is required for a published publication"
                )

        # Дата публикации допустима только после успешной публикации.
        if (
            self.published_at is not None
            and self.status != PublicationStatus.PUBLISHED
        ):
            raise ValueError(
                "published_at is only allowed when status='published'"
            )

        # Внешний ID имеет смысл только вместе с платформой.
        if self.external_id is not None and self.platform is None:
            raise ValueError(
                "platform is required when external_id is specified"
            )

        # Ссылка на внешнюю публикацию также требует площадку.
        if self.external_url is not None and self.platform is None:
            raise ValueError(
                "platform is required when external_url is specified"
            )

        # Ошибка публикации должна храниться только
        # у публикации со статусом FAILED.
        if (
            self.error_message is not None
            and self.status != PublicationStatus.FAILED
        ):
            raise ValueError(
                "error_message is only allowed when status='failed'"
            )

        return self