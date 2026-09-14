from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import find_dotenv, load_dotenv


EMBEDDING_URL = (
    "https://llm.api.cloud.yandex.net/"
    "foundationModels/v1/textEmbedding"
)

EMBEDDING_DIM = 512
DEFAULT_TOP_K = 10

CACHE_DIR = (
    Path(__file__).resolve().parent
    / ".embedding_benchmark_cache"
)


# =============================================================================
# TEST DATA
# =============================================================================
#
# Цель теста:
#   Русский атомарный тезис -> query embedding
#   Английский SemanticUnit -> doc embedding
#
# Две намеренно "плотные" темы:
#   1. hybridization
#   2. gestation
#
# Внутри каждой темы есть много очень похожих по лексике фактов.
# Это важнее, чем простой тест на различение двух совершенно разных тем.
#
# semantic_unit_id здесь тестовый, а не id из Elasticsearch.
# =============================================================================


SEMANTIC_UNITS: list[dict[str, str]] = [
    # -------------------------------------------------------------------------
    # TOPIC 1: HYBRIDIZATION
    # -------------------------------------------------------------------------
    {
        "id": "hyb_fertile_f1",
        "topic": "hybridization",
        "text": (
            "Although dromedaries and Bactrian camels are taxonomically "
            "distinct species, they can interbreed and produce fertile "
            "first-generation hybrids."
        ),
    },
    {
        "id": "hyb_f1_larger_5_12",
        "topic": "hybridization",
        "text": (
            "F1 hybrids are reported to be about 5–12% larger in body size "
            "than either pure-bred parent."
        ),
    },
    {
        "id": "hyb_f1_better_working",
        "topic": "hybridization",
        "text": (
            "F1 dromedary-Bactrian hybrids are described as having better "
            "working ability than their pure-bred parents."
        ),
    },
    {
        "id": "hyb_f1_greater_endurance",
        "topic": "hybridization",
        "text": (
            "F1 dromedary-Bactrian hybrids are described as having greater "
            "endurance than their pure-bred parents."
        ),
    },
    {
        "id": "hyb_f1_cold_wet_terrain",
        "topic": "hybridization",
        "text": (
            "F1 dromedary-Bactrian hybrids are described as better able "
            "than the pure-bred parents to withstand colder, wetter climates "
            "and rough terrain."
        ),
    },
    {
        "id": "hyb_heterosis_from_one_year",
        "topic": "hybridization",
        "text": (
            "The positive effects of heterosis in camel hybrids are described "
            "as becoming stronger from about one year of age onward."
        ),
    },
    {
        "id": "hyb_f1_cross_heterosis_disappears",
        "topic": "hybridization",
        "text": (
            "When F1 camel hybrids are bred with one another, the positive "
            "effects of heterosis disappear."
        ),
    },
    {
        "id": "hyb_jarbay_unfit",
        "topic": "hybridization",
        "text": (
            "The F2 jarbay produced by F1 × F1 matings is described as an "
            "unfit animal."
        ),
    },
    {
        "id": "hyb_higher_birth_weight",
        "topic": "hybridization",
        "text": (
            "Hybrid camel calves are reported to have higher birth weights "
            "than pure-bred dromedary and Bactrian camel calves."
        ),
    },
    {
        "id": "hyb_f1_birth_weight_45_4",
        "topic": "hybridization",
        "text": (
            "The review gives an F1 hybrid mean birth weight of about "
            "45.4 kg, compared with cited means of about 34.5 kg for "
            "dromedary calves and 34.55 kg for Bactrian camel calves."
        ),
    },
    {
        "id": "hyb_higher_growth_rate",
        "topic": "hybridization",
        "text": (
            "Hybrid camel calves are described as having a higher growth rate "
            "than pure-bred dromedary and Bactrian camel calves."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_more_milk",
        "topic": "hybridization",
        "text": (
            "Increasing the proportion of dromedary ancestry in a hybrid "
            "is associated with higher overall milk production."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_longer_lactation",
        "topic": "hybridization",
        "text": (
            "Higher dromedary ancestry is associated with longer and more "
            "abundant lactation in camel hybrids."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_better_udder",
        "topic": "hybridization",
        "text": (
            "Higher dromedary ancestry is associated with improved udder "
            "size and conformation, better quarter symmetry, and improved "
            "teat spacing and orientation."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_less_fat",
        "topic": "hybridization",
        "text": (
            "Higher dromedary ancestry is associated with a lower percentage "
            "of fat in hybrid camel milk."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_less_wool",
        "topic": "hybridization",
        "text": (
            "Higher dromedary ancestry is associated with lower wool "
            "production in camel hybrids."
        ),
    },
    {
        "id": "hyb_dromedary_ancestry_less_cold_adaptation",
        "topic": "hybridization",
        "text": (
            "Higher dromedary ancestry is associated with reduced adaptation "
            "to colder climates in camel hybrids."
        ),
    },
    {
        "id": "hyb_bactrian_ancestry_more_wool",
        "topic": "hybridization",
        "text": (
            "Increasing the proportion of Bactrian ancestry in a hybrid "
            "is associated with higher wool production."
        ),
    },
    {
        "id": "hyb_bactrian_ancestry_more_weight",
        "topic": "hybridization",
        "text": (
            "Increasing the proportion of Bactrian ancestry in a hybrid "
            "is associated with greater live weight."
        ),
    },
    {
        "id": "hyb_bactrian_ancestry_more_milk_fat",
        "topic": "hybridization",
        "text": (
            "Increasing the proportion of Bactrian ancestry in a hybrid "
            "is associated with a higher percentage of fat in milk."
        ),
    },
    {
        "id": "hyb_bactrian_ancestry_shorter_lactation",
        "topic": "hybridization",
        "text": (
            "Higher Bactrian ancestry is associated with shorter lactation "
            "in camel hybrids."
        ),
    },
    {
        "id": "hyb_bactrian_ancestry_lower_milk_yield",
        "topic": "hybridization",
        "text": (
            "Higher Bactrian ancestry is associated with lower absolute "
            "milk yield in camel hybrids."
        ),
    },
    {
        "id": "hyb_all_less_wool_than_bactrian",
        "topic": "hybridization",
        "text": (
            "All hybrids discussed in the review are described as producing "
            "less wool than pure Bactrian camels."
        ),
    },
    {
        "id": "hyb_milk_fat_between_parents",
        "topic": "hybridization",
        "text": (
            "All hybrids discussed in the review are described as having "
            "lower milk-fat percentage than pure Bactrian camels but higher "
            "milk-fat percentage than pure dromedaries."
        ),
    },

    # -------------------------------------------------------------------------
    # TOPIC 2: GESTATION
    # -------------------------------------------------------------------------
    {
        "id": "gest_bactrian_mean_442_5",
        "topic": "gestation",
        "text": (
            "The review gives a mean gestation length of about "
            "442.5 ± 5.1 days for Bactrian camels."
        ),
    },
    {
        "id": "gest_turkmen_mean_425",
        "topic": "gestation",
        "text": (
            "The review gives a mean gestation length of about "
            "425 ± 3.9 days for Turkmen dromedaries."
        ),
    },
    {
        "id": "gest_kazakh_mean_417_2",
        "topic": "gestation",
        "text": (
            "The review gives a mean gestation length of about "
            "417.2 ± 3.1 days for Kazakh dromedaries."
        ),
    },
    {
        "id": "gest_hybrids_between_parents",
        "topic": "gestation",
        "text": (
            "All camel hybrids discussed in the review are described as "
            "having gestation lengths shorter than pure Bactrian camels "
            "but longer than Turkmen or Kazakh dromedaries."
        ),
    },
    {
        "id": "gest_dromedary_ancestry_shortens",
        "topic": "gestation",
        "text": (
            "Dromedary ancestry is described as shortening gestation "
            "duration in camel hybrids."
        ),
    },
    {
        "id": "gest_bactrian_ancestry_lengthens",
        "topic": "gestation",
        "text": (
            "The review states that gestation length increases as the "
            "proportion of Bactrian ancestry increases."
        ),
    },
    {
        "id": "gest_f4_closest_dromedary",
        "topic": "gestation",
        "text": (
            "F4 hybrids are reported to have gestation lengths closest to "
            "those of Turkmen and Kazakh dromedaries."
        ),
    },
    {
        "id": "gest_bactrian_longer_than_turkmen",
        "topic": "gestation",
        "text": (
            "The mean gestation length reported for Bactrian camels is "
            "longer than the mean reported for Turkmen dromedaries."
        ),
    },
    {
        "id": "gest_bactrian_longer_than_kazakh",
        "topic": "gestation",
        "text": (
            "The mean gestation length reported for Bactrian camels is "
            "longer than the mean reported for Kazakh dromedaries."
        ),
    },
    {
        "id": "gest_turkmen_longer_than_kazakh",
        "topic": "gestation",
        "text": (
            "The mean gestation length reported for Turkmen dromedaries "
            "is longer than the mean reported for Kazakh dromedaries."
        ),
    },
    {
        "id": "gest_bactrian_minus_kazakh_25_3",
        "topic": "gestation",
        "text": (
            "Using the reported means, Bactrian camel gestation is about "
            "25.3 days longer than Kazakh dromedary gestation."
        ),
    },
    {
        "id": "gest_bactrian_minus_turkmen_17_5",
        "topic": "gestation",
        "text": (
            "Using the reported means, Bactrian camel gestation is about "
            "17.5 days longer than Turkmen dromedary gestation."
        ),
    },
    {
        "id": "gest_turkmen_minus_kazakh_7_8",
        "topic": "gestation",
        "text": (
            "Using the reported means, Turkmen dromedary gestation is about "
            "7.8 days longer than Kazakh dromedary gestation."
        ),
    },
    {
        "id": "gest_more_bactrian_more_days",
        "topic": "gestation",
        "text": (
            "Within dromedary-Bactrian hybrids, a greater Bactrian genetic "
            "contribution is associated with a longer pregnancy."
        ),
    },
    {
        "id": "gest_more_dromedary_fewer_days",
        "topic": "gestation",
        "text": (
            "Within dromedary-Bactrian hybrids, a greater dromedary genetic "
            "contribution is associated with a shorter pregnancy."
        ),
    },
    {
        "id": "gest_parental_order",
        "topic": "gestation",
        "text": (
            "Among the parental populations cited in the review, the mean "
            "gestation length is highest in Bactrian camels, intermediate "
            "in Turkmen dromedaries, and lowest in Kazakh dromedaries."
        ),
    },
]


# =============================================================================
# RUSSIAN ATOMIC THESIS QUERIES
# =============================================================================
#
# Тезисы специально написаны по-русски и не всегда являются буквальным
# переводом English SemanticUnit.
#
# relevant_ids может содержать несколько unit, если тезис реально
# поддерживается несколькими тестовыми формулировками.
# =============================================================================


THESES: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # HYBRIDIZATION
    # -------------------------------------------------------------------------
    {
        "id": "thesis_h01",
        "topic": "hybridization",
        "text": (
            "Дромадеры и бактрианы способны давать плодовитых "
            "гибридов первого поколения."
        ),
        "relevant_ids": ["hyb_fertile_f1"],
    },
    {
        "id": "thesis_h02",
        "topic": "hybridization",
        "text": (
            "Гибриды F1 дромадера и бактриана примерно на 5–12% "
            "крупнее чистопородных родителей."
        ),
        "relevant_ids": ["hyb_f1_larger_5_12"],
    },
    {
        "id": "thesis_h03",
        "topic": "hybridization",
        "text": (
            "Гибриды F1 превосходят чистопородных дромадеров и "
            "бактрианов по выносливости."
        ),
        "relevant_ids": ["hyb_f1_greater_endurance"],
    },
    {
        "id": "thesis_h04",
        "topic": "hybridization",
        "text": (
            "Гибриды F1 лучше чистопородных родителей переносят "
            "холодный и влажный климат и пересечённую местность."
        ),
        "relevant_ids": ["hyb_f1_cold_wet_terrain"],
    },
    {
        "id": "thesis_h05",
        "topic": "hybridization",
        "text": (
            "При скрещивании гибридов F1 между собой положительный "
            "эффект гетерозиса исчезает."
        ),
        "relevant_ids": ["hyb_f1_cross_heterosis_disappears"],
    },
    {
        "id": "thesis_h06",
        "topic": "hybridization",
        "text": (
            "Гибридные верблюжата рождаются тяжелее чистопородных "
            "верблюжат обоих родительских видов."
        ),
        "relevant_ids": [
            "hyb_higher_birth_weight",
            "hyb_f1_birth_weight_45_4",
        ],
    },
    {
        "id": "thesis_h07",
        "topic": "hybridization",
        "text": (
            "Средняя масса новорождённого гибрида F1 составляет "
            "примерно 45,4 кг."
        ),
        "relevant_ids": ["hyb_f1_birth_weight_45_4"],
    },
    {
        "id": "thesis_h08",
        "topic": "hybridization",
        "text": (
            "Чем выше доля генов дромадера у гибрида, тем выше "
            "его общая молочная продуктивность."
        ),
        "relevant_ids": ["hyb_dromedary_ancestry_more_milk"],
    },
    {
        "id": "thesis_h09",
        "topic": "hybridization",
        "text": (
            "Увеличение доли генов бактриана у гибрида повышает "
            "шерстную продуктивность."
        ),
        "relevant_ids": ["hyb_bactrian_ancestry_more_wool"],
    },
    {
        "id": "thesis_h10",
        "topic": "hybridization",
        "text": (
            "Чем больше у гибрида генетическая доля бактриана, "
            "тем выше жирность его молока."
        ),
        "relevant_ids": ["hyb_bactrian_ancestry_more_milk_fat"],
    },

    # -------------------------------------------------------------------------
    # GESTATION
    # -------------------------------------------------------------------------
    {
        "id": "thesis_g01",
        "topic": "gestation",
        "text": (
            "У бактрианов беременность длится в среднем около "
            "442,5 дня."
        ),
        "relevant_ids": ["gest_bactrian_mean_442_5"],
    },
    {
        "id": "thesis_g02",
        "topic": "gestation",
        "text": (
            "Средняя продолжительность беременности у туркменских "
            "дромадеров составляет около 425 дней."
        ),
        "relevant_ids": ["gest_turkmen_mean_425"],
    },
    {
        "id": "thesis_g03",
        "topic": "gestation",
        "text": (
            "У казахских дромадеров беременность в среднем длится "
            "примерно 417,2 дня."
        ),
        "relevant_ids": ["gest_kazakh_mean_417_2"],
    },
    {
        "id": "thesis_g04",
        "topic": "gestation",
        "text": (
            "Беременность у гибридов короче, чем у чистопородных "
            "бактрианов, но длиннее, чем у дромадеров."
        ),
        "relevant_ids": ["gest_hybrids_between_parents"],
    },
    {
        "id": "thesis_g05",
        "topic": "gestation",
        "text": (
            "Увеличение доли генов дромадера сокращает "
            "продолжительность беременности гибридов."
        ),
        "relevant_ids": [
            "gest_dromedary_ancestry_shortens",
            "gest_more_dromedary_fewer_days",
        ],
    },
    {
        "id": "thesis_g06",
        "topic": "gestation",
        "text": (
            "Чем выше доля генов бактриана у гибрида, тем дольше "
            "продолжается беременность."
        ),
        "relevant_ids": [
            "gest_bactrian_ancestry_lengthens",
            "gest_more_bactrian_more_days",
        ],
    },
    {
        "id": "thesis_g07",
        "topic": "gestation",
        "text": (
            "По продолжительности беременности гибриды поколения F4 "
            "ближе всего к дромадерам."
        ),
        "relevant_ids": ["gest_f4_closest_dromedary"],
    },
    {
        "id": "thesis_g08",
        "topic": "gestation",
        "text": (
            "Беременность у бактриана примерно на 25,3 дня длиннее, "
            "чем у казахского дромадера."
        ),
        "relevant_ids": ["gest_bactrian_minus_kazakh_25_3"],
    },
    {
        "id": "thesis_g09",
        "topic": "gestation",
        "text": (
            "Беременность у бактрианов длится дольше, чем у "
            "туркменских дромадеров."
        ),
        "relevant_ids": [
            "gest_bactrian_longer_than_turkmen",
            "gest_bactrian_minus_turkmen_17_5",
        ],
    },
    {
        "id": "thesis_g10",
        "topic": "gestation",
        "text": (
            "Среди бактрианов, туркменских дромадеров и казахских "
            "дромадеров самая длинная беременность у бактрианов, "
            "а самая короткая — у казахских дромадеров."
        ),
        "relevant_ids": ["gest_parental_order"],
    },
]


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass(frozen=True)
class EmbeddedText:
    id: str
    topic: str
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class RankingItem:
    rank: int
    score: float
    doc: EmbeddedText


# =============================================================================
# MATH
# =============================================================================


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Vector sizes differ: {len(left)} != {len(right)}"
        )

    dot = sum(
        a * b
        for a, b in zip(left, right)
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left
        )
    )

    right_norm = math.sqrt(
        sum(
            value * value
            for value in right
        )
    )

    if left_norm == 0 or right_norm == 0:
        raise ValueError(
            "Zero-length embedding vector"
        )

    return (
        dot
        / (left_norm * right_norm)
    )


# =============================================================================
# CACHE
# =============================================================================


def cache_key(
    *,
    model_uri: str,
    text: str,
    dim: int,
) -> str:
    raw = (
        f"{model_uri}\n"
        f"{dim}\n"
        f"{text}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def load_cached_embedding(
    *,
    model_uri: str,
    text: str,
    dim: int,
) -> list[float] | None:
    key = cache_key(
        model_uri=model_uri,
        text=text,
        dim=dim,
    )

    path = (
        CACHE_DIR
        / f"{key}.json"
    )

    if not path.exists():
        return None

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    embedding = [
        float(value)
        for value in data[
            "embedding"
        ]
    ]

    if len(embedding) != dim:
        return None

    return embedding


def save_cached_embedding(
    *,
    model_uri: str,
    text: str,
    dim: int,
    embedding: list[float],
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = cache_key(
        model_uri=model_uri,
        text=text,
        dim=dim,
    )

    path = (
        CACHE_DIR
        / f"{key}.json"
    )

    path.write_text(
        json.dumps(
            {
                "model_uri": model_uri,
                "dim": dim,
                "text": text,
                "embedding": embedding,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# YANDEX CLIENT
# =============================================================================


class YandexEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        use_cache: bool = True,
    ) -> None:
        self._folder_id = folder_id
        self._use_cache = use_cache

        self.doc_model_uri = (
            f"emb://{folder_id}/"
            "text-embeddings-v2-doc/latest"
        )

        self.query_model_uri = (
            f"emb://{folder_id}/"
            "text-embeddings-v2-query/latest"
        )

        self._client = httpx.Client(
            timeout=60.0,
            headers={
                "Authorization": (
                    f"Api-Key {api_key}"
                ),
                "Content-Type": (
                    "application/json"
                ),
                "x-folder-id": folder_id,
            },
        )

    def close(self) -> None:
        self._client.close()

    def _embed(
        self,
        *,
        text: str,
        model_uri: str,
    ) -> list[float]:
        if self._use_cache:
            cached = (
                load_cached_embedding(
                    model_uri=model_uri,
                    text=text,
                    dim=EMBEDDING_DIM,
                )
            )

            if cached is not None:
                return cached

        response = self._client.post(
            EMBEDDING_URL,
            json={
                "modelUri": model_uri,
                "text": text,
                "dim": str(
                    EMBEDDING_DIM
                ),
            },
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "Yandex Embeddings API error:\n"
                f"status="
                f"{response.status_code}\n"
                f"body="
                f"{response.text}"
            ) from exc

        payload = response.json()

        embedding = [
            float(value)
            for value in payload[
                "embedding"
            ]
        ]

        if (
            len(embedding)
            != EMBEDDING_DIM
        ):
            raise RuntimeError(
                "Unexpected embedding "
                f"dimension: "
                f"{len(embedding)}; "
                f"expected "
                f"{EMBEDDING_DIM}"
            )

        if self._use_cache:
            save_cached_embedding(
                model_uri=model_uri,
                text=text,
                dim=EMBEDDING_DIM,
                embedding=embedding,
            )

        return embedding

    def embed_doc(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(
            text=text,
            model_uri=self.doc_model_uri,
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(
            text=text,
            model_uri=self.query_model_uri,
        )


# =============================================================================
# CONFIG
# =============================================================================


def load_config() -> tuple[str, str]:
    env_path = find_dotenv(
        usecwd=True
    )

    if env_path:
        load_dotenv(
            env_path
        )
    else:
        load_dotenv()

    api_key = os.getenv(
        "YANDEX_API_KEY"
    )

    folder_id = os.getenv(
        "YANDEX_FOLDER_ID"
    )

    if not api_key:
        raise RuntimeError(
            "YANDEX_API_KEY is missing "
            "in .env"
        )

    if not folder_id:
        raise RuntimeError(
            "YANDEX_FOLDER_ID is missing "
            "in .env"
        )

    # YANDEX_KEY_ID для Embeddings API
    # в этом тесте не требуется.
    return api_key, folder_id


# =============================================================================
# EMBEDDING
# =============================================================================


def embed_semantic_units(
    client: YandexEmbeddingClient,
) -> list[EmbeddedText]:
    result: list[EmbeddedText] = []

    print(
        "Embedding Semantic Units "
        "with DOC model..."
    )

    for index, unit in enumerate(
        SEMANTIC_UNITS,
        start=1,
    ):
        print(
            f"  [{index:02d}/"
            f"{len(SEMANTIC_UNITS):02d}] "
            f"{unit['id']}"
        )

        result.append(
            EmbeddedText(
                id=unit["id"],
                topic=unit["topic"],
                text=unit["text"],
                embedding=(
                    client.embed_doc(
                        unit["text"]
                    )
                ),
            )
        )

    return result


# =============================================================================
# RANKING
# =============================================================================


def rank_documents(
    *,
    query_embedding: list[float],
    docs: list[EmbeddedText],
) -> list[RankingItem]:
    scored = sorted(
        (
            (
                cosine_similarity(
                    query_embedding,
                    doc.embedding,
                ),
                doc,
            )
            for doc in docs
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        RankingItem(
            rank=index,
            score=score,
            doc=doc,
        )
        for index, (
            score,
            doc,
        ) in enumerate(
            scored,
            start=1,
        )
    ]


def first_relevant_rank(
    *,
    ranking: list[RankingItem],
    relevant_ids: set[str],
) -> int | None:
    for item in ranking:
        if item.doc.id in relevant_ids:
            return item.rank

    return None


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(
    *,
    client: YandexEmbeddingClient,
    docs: list[EmbeddedText],
    display_top_k: int,
) -> None:
    total = len(
        THESES
    )

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    topic_hits = 0

    reciprocal_rank_sum = 0.0

    per_topic: dict[
        str,
        dict[str, float],
    ] = {}

    print()
    print(
        "=" * 100
    )
    print(
        "RUSSIAN ATOMIC THESIS "
        "-> ENGLISH SEMANTIC UNITS"
    )
    print(
        "=" * 100
    )

    for thesis_index, thesis in enumerate(
        THESES,
        start=1,
    ):
        query_embedding = (
            client.embed_query(
                thesis["text"]
            )
        )

        ranking = rank_documents(
            query_embedding=query_embedding,
            docs=docs,
        )

        relevant_ids = set(
            thesis[
                "relevant_ids"
            ]
        )

        rank = first_relevant_rank(
            ranking=ranking,
            relevant_ids=relevant_ids,
        )

        top1_ok = (
            rank is not None
            and rank <= 1
        )

        top3_ok = (
            rank is not None
            and rank <= 3
        )

        top5_ok = (
            rank is not None
            and rank <= 5
        )

        top10_ok = (
            rank is not None
            and rank <= 10
        )

        top1_topic_ok = (
            ranking[0].doc.topic
            == thesis["topic"]
        )

        top1_hits += int(
            top1_ok
        )

        top3_hits += int(
            top3_ok
        )

        top5_hits += int(
            top5_ok
        )

        top10_hits += int(
            top10_ok
        )

        topic_hits += int(
            top1_topic_ok
        )

        reciprocal_rank = (
            1.0 / rank
            if rank is not None
            else 0.0
        )

        reciprocal_rank_sum += (
            reciprocal_rank
        )

        topic_stats = (
            per_topic.setdefault(
                thesis["topic"],
                {
                    "count": 0.0,
                    "top1": 0.0,
                    "top3": 0.0,
                    "top5": 0.0,
                    "top10": 0.0,
                    "mrr_sum": 0.0,
                },
            )
        )

        topic_stats[
            "count"
        ] += 1

        topic_stats[
            "top1"
        ] += int(
            top1_ok
        )

        topic_stats[
            "top3"
        ] += int(
            top3_ok
        )

        topic_stats[
            "top5"
        ] += int(
            top5_ok
        )

        topic_stats[
            "top10"
        ] += int(
            top10_ok
        )

        topic_stats[
            "mrr_sum"
        ] += reciprocal_rank

        print()
        print(
            f"[{thesis_index:02d}/"
            f"{total:02d}] "
            f"{thesis['id']}"
        )

        print(
            thesis["text"]
        )

        print(
            "Expected: "
            + ", ".join(
                thesis[
                    "relevant_ids"
                ]
            )
        )

        if rank is None:
            print(
                "First relevant rank: "
                "NOT FOUND"
            )
        else:
            print(
                "First relevant rank: "
                f"{rank}"
            )

        top1 = ranking[0]

        print(
            "Top-1: "
            f"{top1.doc.id} | "
            f"score="
            f"{top1.score:.6f} | "
            f"topic="
            f"{top1.doc.topic} | "
            f"{'PASS' if top1_ok else 'MISS'}"
        )

        print(
            f"Top-{display_top_k}:"
        )

        for item in ranking[
            :display_top_k
        ]:
            marker = (
                "*"
                if (
                    item.doc.id
                    in relevant_ids
                )
                else " "
            )

            print(
                f"  {marker} "
                f"{item.rank:2d}. "
                f"{item.score:.6f} | "
                f"{item.doc.id:<42} | "
                f"{item.doc.topic}"
            )

            print(
                "       "
                f"{item.doc.text}"
            )

    mrr = (
        reciprocal_rank_sum
        / total
    )

    print()
    print(
        "=" * 100
    )
    print(
        "GLOBAL SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        f"Semantic Units: "
        f"{len(docs)}"
    )

    print(
        f"Russian theses: "
        f"{total}"
    )

    print(
        f"Top-1 exact:   "
        f"{top1_hits}/{total} "
        f"({top1_hits / total:.1%})"
    )

    print(
        f"Top-3 recall:  "
        f"{top3_hits}/{total} "
        f"({top3_hits / total:.1%})"
    )

    print(
        f"Top-5 recall:  "
        f"{top5_hits}/{total} "
        f"({top5_hits / total:.1%})"
    )

    print(
        f"Top-10 recall: "
        f"{top10_hits}/{total} "
        f"({top10_hits / total:.1%})"
    )

    print(
        f"Top-1 topic:   "
        f"{topic_hits}/{total} "
        f"({topic_hits / total:.1%})"
    )

    print(
        f"MRR:           "
        f"{mrr:.4f}"
    )

    print()
    print(
        "=" * 100
    )
    print(
        "PER-TOPIC SUMMARY"
    )
    print(
        "=" * 100
    )

    for topic, stats in sorted(
        per_topic.items()
    ):
        count = int(
            stats["count"]
        )

        topic_mrr = (
            stats["mrr_sum"]
            / count
        )

        print()
        print(
            topic
        )

        print(
            "  Top-1:   "
            f"{int(stats['top1'])}/"
            f"{count} "
            f"({stats['top1'] / count:.1%})"
        )

        print(
            "  Top-3:   "
            f"{int(stats['top3'])}/"
            f"{count} "
            f"({stats['top3'] / count:.1%})"
        )

        print(
            "  Top-5:   "
            f"{int(stats['top5'])}/"
            f"{count} "
            f"({stats['top5'] / count:.1%})"
        )

        print(
            "  Top-10:  "
            f"{int(stats['top10'])}/"
            f"{count} "
            f"({stats['top10'] / count:.1%})"
        )

        print(
            "  MRR:     "
            f"{topic_mrr:.4f}"
        )

    print()
    print(
        "=" * 100
    )
    print(
        "INTERPRETATION"
    )
    print(
        "=" * 100
    )

    if (
        top5_hits / total
        >= 0.95
    ):
        print(
            "PASS: Top-5 recall is at least "
            "95%. Embeddings are suitable "
            "for candidate retrieval."
        )
    else:
        print(
            "WARNING: Top-5 recall is below "
            "95%. Candidate retrieval may "
            "need a larger TOP_K, query "
            "translation, or another model."
        )

    if (
        top10_hits == total
    ):
        print(
            "PASS: every intended SemanticUnit "
            "was found inside Top-10."
        )
    else:
        print(
            "WARNING: some intended SemanticUnits "
            "were outside Top-10."
        )

    print()
    print(
        "Important: cosine similarity is used "
        "only for candidate ranking."
    )

    print(
        "Do not treat a high cosine score as "
        "proof that a SemanticUnit supports "
        "an AtomicThesis."
    )


# =============================================================================
# OPTIONAL JSON REPORT
# =============================================================================


def build_json_report(
    *,
    client: YandexEmbeddingClient,
    docs: list[EmbeddedText],
    top_k: int,
) -> dict[str, Any]:
    results: list[
        dict[str, Any]
    ] = []

    for thesis in THESES:
        query_embedding = (
            client.embed_query(
                thesis["text"]
            )
        )

        ranking = rank_documents(
            query_embedding=query_embedding,
            docs=docs,
        )

        relevant_ids = set(
            thesis[
                "relevant_ids"
            ]
        )

        first_rank = first_relevant_rank(
            ranking=ranking,
            relevant_ids=relevant_ids,
        )

        results.append(
            {
                "thesis_id": thesis["id"],
                "topic": thesis["topic"],
                "text": thesis["text"],
                "relevant_ids": thesis[
                    "relevant_ids"
                ],
                "first_relevant_rank": first_rank,
                "ranking": [
                    {
                        "rank": item.rank,
                        "score": item.score,
                        "semantic_unit_id": (
                            item.doc.id
                        ),
                        "topic": (
                            item.doc.topic
                        ),
                        "text": (
                            item.doc.text
                        ),
                        "is_relevant": (
                            item.doc.id
                            in relevant_ids
                        ),
                    }
                    for item in ranking[
                        :top_k
                    ]
                ],
            }
        )

    return {
        "embedding_dim": (
            EMBEDDING_DIM
        ),
        "doc_model_uri": (
            client.doc_model_uri
        ),
        "query_model_uri": (
            client.query_model_uri
        ),
        "semantic_unit_count": (
            len(docs)
        ),
        "thesis_count": (
            len(THESES)
        ),
        "results": results,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-lingual benchmark: "
            "Russian AtomicThesis queries "
            "against English SemanticUnit "
            "documents using Yandex "
            "Text Embeddings v2."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "How many ranked SemanticUnits "
            "to print and save. "
            f"Default: {DEFAULT_TOP_K}."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable local embedding cache "
            "and call Yandex API for every "
            "text."
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional path for a JSON report, "
            "for example "
            "benchmark_report.json."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError(
            "--top-k must be >= 1"
        )

    api_key, folder_id = (
        load_config()
    )

    client = YandexEmbeddingClient(
        api_key=api_key,
        folder_id=folder_id,
        use_cache=(
            not args.no_cache
        ),
    )

    try:
        print(
            "DOC model:   "
            f"{client.doc_model_uri}"
        )

        print(
            "QUERY model: "
            f"{client.query_model_uri}"
        )

        print(
            "Dimension:   "
            f"{EMBEDDING_DIM}"
        )

        print(
            "Cache:       "
            f"{'ON' if not args.no_cache else 'OFF'}"
        )

        print(
            "Documents:   "
            f"{len(SEMANTIC_UNITS)}"
        )

        print(
            "Theses:      "
            f"{len(THESES)}"
        )

        print()

        docs = (
            embed_semantic_units(
                client
            )
        )

        evaluate(
            client=client,
            docs=docs,
            display_top_k=min(
                args.top_k,
                len(docs),
            ),
        )

        if args.report is not None:
            report = build_json_report(
                client=client,
                docs=docs,
                top_k=min(
                    args.top_k,
                    len(docs),
                ),
            )

            args.report.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.report.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print()
            print(
                "JSON report written to: "
                f"{args.report}"
            )

    finally:
        client.close()


if __name__ == "__main__":
    main()
