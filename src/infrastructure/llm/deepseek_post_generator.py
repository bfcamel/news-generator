from __future__ import annotations

import hashlib
import json
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.domain.publication_trial import (
    PublicationEvidence,
)
from src.domain.types import NonEmptyStr


PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "post_generation.txt"
)

OUTPUT_SCHEMA_PLACEHOLDER = (
    "{{OUTPUT_JSON_SCHEMA}}"
)

POST_GENERATION_SCHEMA: dict[str, Any] = {
    "$schema": (
        "https://json-schema.org/"
        "draft/2020-12/schema"
    ),
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "minLength": 1,
        },
        "title": {
            "type": "string",
            "minLength": 1,
        },
        "paragraphs": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {
                "type": "string",
                "minLength": 1,
            },
        },
        "used_semantic_unit_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "minLength": 1,
            },
        },
    },
    "required": [
        "topic",
        "title",
        "paragraphs",
        "used_semantic_unit_ids",
    ],
    "additionalProperties": False,
}


class GeneratedPostPayload(BaseModel):
    """
    Exact JSON object requested from DeepSeek.

    The model returns paragraphs separately so the application can
    guarantee a real multi-paragraph publication.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    topic: NonEmptyStr
    title: NonEmptyStr

    paragraphs: list[
        NonEmptyStr
    ] = Field(
        ...,
        min_length=3,
        max_length=5,
    )

    used_semantic_unit_ids: list[
        NonEmptyStr
    ] = Field(
        ...,
        min_length=1,
    )

    @field_validator(
        "paragraphs"
    )
    @classmethod
    def normalize_paragraphs(
        cls,
        paragraphs: list[str],
    ) -> list[str]:
        normalized = [
            " ".join(
                paragraph.split()
            )
            for paragraph in paragraphs
        ]

        if any(
            not paragraph
            for paragraph in normalized
        ):
            raise ValueError(
                "paragraphs cannot contain empty items"
            )

        return normalized

    @model_validator(
        mode="after"
    )
    def unique_units(
        self,
    ) -> "GeneratedPostPayload":
        if len(
            self.used_semantic_unit_ids
        ) != len(
            set(
                self.used_semantic_unit_ids
            )
        ):
            raise ValueError(
                "used_semantic_unit_ids must be unique"
            )

        return self


class GeneratedPost(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    topic: NonEmptyStr
    title: NonEmptyStr
    post_text: NonEmptyStr

    used_semantic_unit_ids: list[
        NonEmptyStr
    ] = Field(
        ...,
        min_length=1,
    )


@dataclass(
    frozen=True,
    slots=True,
)
class DeepSeekPostSettings:
    api_key: str
    folder_id: str

    model: str = "deepseek-v4-flash"

    base_url: str = (
        "https://ai.api.cloud.yandex.net/v1"
    )

    max_output_tokens: int = 3500

    @property
    def model_uri(
        self,
    ) -> str:
        if self.model.startswith(
            "gpt://"
        ):
            return self.model

        return (
            f"gpt://{self.folder_id}/"
            f"{self.model}"
        )

    @classmethod
    def from_env(
        cls,
    ) -> "DeepSeekPostSettings":
        env_path = find_dotenv(
            usecwd=True
        )

        if env_path:
            load_dotenv(
                env_path,
                override=False,
            )

        api_key = os.getenv(
            "YANDEX_API_KEY"
        )

        folder_id = os.getenv(
            "YANDEX_FOLDER_ID"
        )

        if not api_key:
            raise RuntimeError(
                "YANDEX_API_KEY is missing"
            )

        if not folder_id:
            raise RuntimeError(
                "YANDEX_FOLDER_ID is missing"
            )

        return cls(
            api_key=api_key,
            folder_id=folder_id,
            model=os.getenv(
                "YANDEX_DEEPSEEK_MODEL",
                "deepseek-v4-flash",
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class GeneratedPostResult:
    post: GeneratedPost
    model: str
    prompt_file: str
    prompt_sha256: str


def render_generation_prompt() -> str:
    template = PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not template:
        raise RuntimeError(
            "Post generation prompt is empty"
        )

    placeholder_count = (
        template.count(
            OUTPUT_SCHEMA_PLACEHOLDER
        )
    )

    if placeholder_count != 1:
        raise RuntimeError(
            "Post generation prompt must contain "
            "exactly one "
            "{{OUTPUT_JSON_SCHEMA}} marker"
        )

    schema = json.dumps(
        POST_GENERATION_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    return template.replace(
        OUTPUT_SCHEMA_PLACEHOLDER,
        schema,
    )


def _load_json_object(
    raw: str,
) -> dict[str, Any]:
    text = raw.strip()

    if not text:
        raise RuntimeError(
            "DeepSeek returned empty output"
        )

    candidates: list[str] = [
        text
    ]

    object_start = text.find(
        "{"
    )

    object_end = text.rfind(
        "}"
    )

    if (
        object_start >= 0
        and object_end
        > object_start
    ):
        candidates.append(
            text[
                object_start:
                object_end + 1
            ]
        )

    seen: set[str] = set()

    for candidate in candidates:
        if candidate in seen:
            continue

        seen.add(
            candidate
        )

        try:
            data = json.loads(
                candidate
            )
        except json.JSONDecodeError:
            continue

        if isinstance(
            data,
            dict,
        ):
            return data

    raise RuntimeError(
        "DeepSeek returned invalid JSON"
    )


def parse_generated_post(
    raw: str,
) -> GeneratedPost:
    normalized_raw = raw.strip()
    fence = chr(96) * 3

    if normalized_raw.startswith(
        fence
    ):
        lines = (
            normalized_raw
            .splitlines()
        )

        if lines:
            first = (
                lines[0]
                .strip()
                .lower()
            )

            if first in {
                fence,
                fence + "json",
            }:
                lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == fence
        ):
            lines = lines[:-1]

        normalized_raw = "\n".join(
            lines
        ).strip()

    data = _load_json_object(
        normalized_raw
    )

    payload = (
        GeneratedPostPayload
        .model_validate(
            data
        )
    )

    return GeneratedPost(
        topic=payload.topic,
        title=payload.title,
        post_text="\n\n".join(
            payload.paragraphs
        ),
        used_semantic_unit_ids=(
            payload
            .used_semantic_unit_ids
        ),
    )


class DeepSeekPostGenerator:
    def __init__(
        self,
        settings: DeepSeekPostSettings,
    ) -> None:
        self.settings = settings

        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            project=settings.folder_id,
        )

    async def generate(
        self,
        *,
        evidence: list[
            PublicationEvidence
        ],
    ) -> GeneratedPostResult:
        if not evidence:
            raise ValueError(
                "Evidence cannot be empty"
            )

        prompt = render_generation_prompt()

        prompt_sha256 = hashlib.sha256(
            prompt.encode(
                "utf-8"
            )
        ).hexdigest()

        payload = {
            "semantic_units": [
                {
                    "id": (
                        item.semantic_unit_id
                    ),
                    "text": item.text,
                    "taxa": item.taxa,
                    "section_title": (
                        item.section_title
                    ),
                    "page_start": (
                        item.page_start
                    ),
                    "page_end": (
                        item.page_end
                    ),
                    "source": {
                        "title": (
                            item.source_title
                        ),
                        "authors": (
                            item.source_authors
                        ),
                        "year": (
                            item.source_year
                        ),
                        "doi": (
                            item.source_doi
                        ),
                    },
                }
                for item in evidence
            ]
        }

        response = (
            await self.client.responses.create(
                model=(
                    self.settings.model_uri
                ),
                instructions=prompt,
                input=(
                    "Ниже входные Semantic Units. "
                    "Используй только их как источник "
                    "научных фактов.\n\n"
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                ),
                max_output_tokens=(
                    self.settings
                    .max_output_tokens
                ),
            )
        )

        raw = response.output_text

        if not raw:
            raise RuntimeError(
                "DeepSeek returned empty output"
            )

        post = parse_generated_post(
            raw
        )

        allowed_ids = {
            item.semantic_unit_id
            for item in evidence
        }

        unknown_ids = (
            set(
                post
                .used_semantic_unit_ids
            )
            - allowed_ids
        )

        if unknown_ids:
            raise RuntimeError(
                "DeepSeek returned unknown "
                "SemanticUnit IDs: "
                + ", ".join(
                    sorted(
                        unknown_ids
                    )
                )
            )

        return GeneratedPostResult(
            post=post,
            model=self.settings.model,
            prompt_file=str(
                PROMPT_PATH.relative_to(
                    Path(__file__)
                    .resolve()
                    .parents[3]
                )
            ),
            prompt_sha256=(
                prompt_sha256
            ),
        )

    async def close(
        self,
    ) -> None:
        await self.client.close()
