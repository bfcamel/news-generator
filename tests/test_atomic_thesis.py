from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.atomic_thesis import (
    AtomicThesis,
    TaxonScope,
    ThesisStatus,
    calculate_text_hash,
)


def test_text_hash_is_generated() -> None:
    thesis = AtomicThesis(
        id="thesis_1",
        text=(
            "Беременность у дромадера "
            "длится около 13 месяцев."
        ),
        semantic_unit_ids=[
            "unit_1"
        ],
        taxa=[
            "Camelus dromedarius"
        ],
        taxon_scope=TaxonScope.SPECIES,
    )

    assert thesis.text_hash == (
        calculate_text_hash(
            thesis.text
        )
    )


def test_text_hash_ignores_whitespace_and_case() -> None:
    left = calculate_text_hash(
        "Верблюд   живёт в пустыне."
    )

    right = calculate_text_hash(
        "  ВЕРБЛЮД живёт в пустыне. "
    )

    assert left == right


def test_semantic_unit_ids_are_deduplicated() -> None:
    thesis = AtomicThesis(
        id="thesis_1",
        text="Тестовый тезис.",
        semantic_unit_ids=[
            "unit_1",
            "unit_1",
            "unit_2",
        ],
    )

    assert thesis.semantic_unit_ids == [
        "unit_1",
        "unit_2",
    ]


def test_species_scope_requires_one_taxon() -> None:
    with pytest.raises(
        ValidationError
    ):
        AtomicThesis(
            id="thesis_1",
            text="Тестовый тезис.",
            semantic_unit_ids=[
                "unit_1"
            ],
            taxa=[],
            taxon_scope=(
                TaxonScope.SPECIES
            ),
        )


def test_multi_species_requires_two_taxa() -> None:
    with pytest.raises(
        ValidationError
    ):
        AtomicThesis(
            id="thesis_1",
            text="Тестовый тезис.",
            semantic_unit_ids=[
                "unit_1"
            ],
            taxa=[
                "Camelus dromedarius"
            ],
            taxon_scope=(
                TaxonScope.MULTI_SPECIES
            ),
        )


def test_default_status_is_pending_review() -> None:
    thesis = AtomicThesis(
        id="thesis_1",
        text="Тестовый тезис.",
        semantic_unit_ids=[
            "unit_1"
        ],
    )

    assert thesis.status == (
        ThesisStatus.PENDING_REVIEW
    )
