from __future__ import annotations

import hashlib
import json
import os

from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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

POST_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
        },
        "title": {
            "type": "string",
        },
        "post_text": {
            "type": "string",
        },
        "used_semantic_unit_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "topic",
        "title",
        "post_text",
        "used_semantic_unit_ids",
    ],
    "additionalProperties": False,
}


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

    @model_validator(mode="after")
    def unique_units(
        self,
    ) -> "GeneratedPost":
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


@dataclass(
    frozen=True,
    slots=True,
)
class YandexGPTPostSettings:
    api_key: str
    folder_id: str

    model: str = "yandexgpt-5.1/latest"
    base_url: str = (
        "https://ai.api.cloud.yandex.net/v1"
    )
    temperature: float = 0.35
    max_output_tokens: int = 3500

    @property
    def model_uri(
        self,
    ) -> str:
        return (
            f"gpt://{self.folder_id}/"
            f"{self.model}"
        )

    @classmethod
    def from_env(
        cls,
    ) -> "YandexGPTPostSettings":
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
                "YANDEX_GPT_MODEL",
                "yandexgpt-5.1/latest",
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


class YandexGPTPostGenerator:
    def __init__(
        self,
        settings: YandexGPTPostSettings,
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

        prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()

        if not prompt:
            raise RuntimeError(
                "Post generation prompt is empty"
            )

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
                temperature=(
                    self.settings.temperature
                ),
                instructions=prompt,
                input=json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                max_output_tokens=(
                    self.settings
                    .max_output_tokens
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": (
                            "generated_camel_post"
                        ),
                        "description": (
                            "A Russian popular-science "
                            "post grounded only in the "
                            "provided Semantic Units"
                        ),
                        "strict": True,
                        "schema": (
                            POST_GENERATION_SCHEMA
                        ),
                    },
                    "verbosity": "low",
                },
            )
        )

        raw = response.output_text

        if not raw:
            raise RuntimeError(
                "YandexGPT returned empty output"
            )

        data = json.loads(
            raw
        )

        post = GeneratedPost.model_validate(
            data
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
                "YandexGPT returned unknown "
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
