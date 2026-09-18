from __future__ import annotations

import json
import os

from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

from src.domain.atomic_thesis_candidate import (
    AtomicThesisExtractionResponse,
)
from src.domain.semantic_unit import SemanticUnit
from src.domain.source_document import (
    SourceDocument,
)


PROMPT_VERSION = (
    "atomic_thesis_extractor_v1"
)


ATOMIC_THESIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "theses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    },
                    "taxa": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "taxon_scope": {
                        "type": "string",
                        "enum": [
                            "species",
                            "multi_species",
                            "genus",
                            "family",
                            "unspecified",
                        ],
                    },
                },
                "required": [
                    "text",
                    "taxa",
                    "taxon_scope",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "theses",
    ],
    "additionalProperties": False,
}


INSTRUCTIONS = """
Ты выполняешь строгое извлечение научных утверждений
из одного SemanticUnit.

Единственный источник фактов — поле semantic_unit_text
текущего запроса.

Твоя задача:
вернуть от 0 до N атомарных научных тезисов
на русском языке.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:

1. Не используй внешние знания.
2. Не используй информацию из других источников.
3. Не добавляй сведения, которых нет в semantic_unit_text.
4. Один тезис должен содержать одно самостоятельное
   проверяемое утверждение.
5. Если в SemanticUnit несколько независимых утверждений,
   разделяй их.
6. Допустимо вернуть пустой массив theses.
7. Каждый тезис должен быть понятен без исходного абзаца.
8. Не используй неопределённые местоимения, если без
   исходного текста непонятен субъект.
9. Сохраняй все важные числа, диапазоны и единицы измерения.
10. Сохраняй ограничения по популяции, месту, времени,
    экспериментальной группе и другим условиям.
11. Не округляй и не пересчитывай числовые значения
    без необходимости.
12. Не делай новых арифметических или логических выводов.
13. Не превращай ассоциацию или корреляцию в причинность.
14. Не превращай may, could, suggest, possibly,
    hypothesis и аналогичные формулировки
    в установленный факт.
15. Не создавай красивых общих выводов вместо конкретных
    научных утверждений.
16. semantic_unit_taxa, source_title, section_title и
    source_primary_taxon используются только как контекст.
    Они не являются дополнительными источниками фактов.
17. taxa определяй отдельно для каждого конкретного тезиса.
18. Не копируй taxa SemanticUnit автоматически.

19. taxon_scope и taxa должны быть согласованы:

    species:
    - относится ровно к одному виду;
    - taxa ОБЯЗАТЕЛЬНО содержит ровно один элемент.

    multi_species:
    - утверждение непосредственно относится минимум
      к двум видам или сравнивает их;
    - taxa ОБЯЗАТЕЛЬНО содержит минимум два элемента.

    genus:
    - утверждение относится к роду;
    - taxa ОБЯЗАТЕЛЬНО содержит минимум один таксон рода.

    family:
    - утверждение относится к семейству;
    - taxa ОБЯЗАТЕЛЬНО содержит минимум один таксон семейства.

    unspecified:
    - используй, если taxon_scope нельзя определить
      надёжно или если данных недостаточно.

20. Никогда не указывай multi_species,
    если в taxa меньше двух таксонов.

21. Никогда не указывай species,
    если в taxa не ровно один таксон.

22. Если сомневаешься между scope,
    выбирай unspecified, а не делай предположение.

23. При конфликте полноты и точности выбирай точность.
    Лучше пропустить сомнительный тезис, чем придумать факт.

Тезис — элемент научной базы знаний.
Не пиши его как пост для социальной сети.

Ответ должен строго соответствовать переданной JSON Schema.
""".strip()


@dataclass(
    frozen=True,
    slots=True,
)
class YandexGPTSettings:
    api_key: str
    folder_id: str

    model: str = "yandexgpt-5.1/latest"

    base_url: str = (
        "https://ai.api.cloud.yandex.net/v1"
    )

    temperature: float = 0.1
    max_output_tokens: int = 2500

    @property
    def model_uri(self) -> str:
        return (
            f"gpt://{self.folder_id}/"
            f"{self.model}"
        )

    @classmethod
    def from_env(
        cls,
    ) -> "YandexGPTSettings":
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


class YandexGPTThesisExtractor:
    def __init__(
        self,
        settings: YandexGPTSettings,
    ) -> None:
        self.settings = settings

        self.client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            project=settings.folder_id,
        )

    @property
    def model_name(
        self,
    ) -> str:
        return self.settings.model

    @property
    def prompt_version(
        self,
    ) -> str:
        return PROMPT_VERSION

    async def extract(
        self,
        *,
        unit: SemanticUnit,
        source: SourceDocument | None,
    ) -> AtomicThesisExtractionResponse:
        payload = {
            "semantic_unit_id": unit.id,
            "semantic_unit_text": unit.text,
            "semantic_unit_taxa": unit.taxa,
            "section_title": (
                unit.section_title
            ),
            "page_start": unit.page_start,
            "page_end": unit.page_end,
            "source_title": (
                source.title
                if source
                else None
            ),
            "source_primary_taxon": (
                source.primary_taxon
                if source
                else None
            ),
        }

        response = (
            await self.client.responses.create(
                model=(
                    self.settings.model_uri
                ),
                temperature=(
                    self.settings.temperature
                ),
                instructions=INSTRUCTIONS,
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
                            "atomic_thesis_extraction"
                        ),
                        "description": (
                            "0..N atomic scientific "
                            "theses extracted from "
                            "one SemanticUnit"
                        ),
                        "strict": True,
                        "schema": (
                            ATOMIC_THESIS_JSON_SCHEMA
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

        data = json.loads(raw)

        return (
            AtomicThesisExtractionResponse
            .model_validate(data)
        )

    async def close(
        self,
    ) -> None:
        await self.client.close()