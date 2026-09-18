from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.infrastructure.llm.deepseek_post_generator import (
    DeepSeekPostSettings,
    POST_GENERATION_SCHEMA,
    parse_generated_post,
    render_generation_prompt,
)


def test_deepseek_model_uri() -> None:
    settings = DeepSeekPostSettings(
        api_key="test",
        folder_id="folder123",
    )

    assert settings.model_uri == (
        "gpt://folder123/"
        "deepseek-v4-flash"
    )


def test_json_schema_is_rendered_inside_prompt() -> None:
    prompt = render_generation_prompt()

    assert "{{OUTPUT_JSON_SCHEMA}}" not in prompt

    schema = json.dumps(
        POST_GENERATION_SCHEMA,
        ensure_ascii=False,
        indent=2,
    )

    assert schema in prompt
    assert '"paragraphs"' in prompt
    assert '"minItems": 3' in prompt
    assert '"maxItems": 5' in prompt


def test_generated_paragraphs_are_joined() -> None:
    raw = """{
  "topic": "Тема",
  "title": "Заголовок",
  "paragraphs": [
    "Первый абзац.",
    "Второй абзац.",
    "Третий абзац."
  ],
  "used_semantic_unit_ids": [
    "unit_1"
  ]
}"""

    post = parse_generated_post(
        raw
    )

    assert post.post_text == (
        "Первый абзац.\n\n"
        "Второй абзац.\n\n"
        "Третий абзац."
    )


def test_less_than_three_paragraphs_is_rejected() -> None:
    raw = """{
  "topic": "Тема",
  "title": "Заголовок",
  "paragraphs": [
    "Первый абзац.",
    "Второй абзац."
  ],
  "used_semantic_unit_ids": [
    "unit_1"
  ]
}"""

    with pytest.raises(
        ValidationError
    ):
        parse_generated_post(
            raw
        )
