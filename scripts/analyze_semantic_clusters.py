from __future__ import annotations

import argparse
import asyncio
import json

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from src.infrastructure.elasticsearch.client import es


SEMANTIC_UNITS_INDEX = "semantic_units"


# ============================================================
# TAXONOMY
# ============================================================
#
# Это НЕ научная таксономическая база.
#
# Это только эвристика для кластеризации нашей базы знаний.
# Если позже понадобится более сложная система,
# этот словарь можно заменить отдельным taxonomy service.
# ============================================================


TAXON_PATHS: dict[str, tuple[str, ...]] = {
    "Camelidae": (
        "Camelidae",
    ),

    "Camelini": (
        "Camelidae",
        "Camelini",
    ),

    "Camelus": (
        "Camelidae",
        "Camelini",
        "Camelus",
    ),

    "Camelus dromedarius": (
        "Camelidae",
        "Camelini",
        "Camelus",
        "Camelus dromedarius",
    ),

    "Camelus bactrianus": (
        "Camelidae",
        "Camelini",
        "Camelus",
        "Camelus bactrianus",
    ),

    "Camelus ferus": (
        "Camelidae",
        "Camelini",
        "Camelus",
        "Camelus ferus",
    ),

    "Camelus thomasi": (
        "Camelidae",
        "Camelini",
        "Camelus",
        "Camelus thomasi",
    ),

    "Camelus knoblochi": (
        "Camelidae",
        "Camelini",
        "Camelus",
        "Camelus knoblochi",
    ),

    # Для целей кластеризации связываем только
    # через семейство Camelidae.
    "Paracamelus": (
        "Camelidae",
        "Paracamelus",
    ),

    "Lama": (
        "Camelidae",
        "Lama",
    ),

    "Vicugna": (
        "Camelidae",
        "Vicugna",
    ),
}


# ============================================================
# DATA
# ============================================================


@dataclass(
    frozen=True,
    slots=True,
)
class Unit:
    id: str
    text: str
    source_document_id: str
    taxa: tuple[str, ...]
    embedding: np.ndarray


@dataclass(
    frozen=True,
    slots=True,
)
class Cluster:
    ids: tuple[int, ...]


# ============================================================
# TAXONOMY SCORE
# ============================================================


def taxon_pair_score(
    left: str,
    right: str,
) -> float:
    """
    Эвристическая совместимость двух таксонов.

    1.00 — один и тот же таксон
    0.90 — вид и его род
    0.70 — разные виды одного рода
    0.65 — потомок и его tribe
    0.50 — общий tribe
    0.45 — потомок и family
    0.30 — общее family
    0.00 — связи не установлено
    """

    if left == right:
        return 1.0

    left_path = TAXON_PATHS.get(
        left
    )

    right_path = TAXON_PATHS.get(
        right
    )

    if (
        left_path is None
        or right_path is None
    ):
        return 0.0

    common_length = 0

    for left_item, right_item in zip(
        left_path,
        right_path,
    ):
        if left_item != right_item:
            break

        common_length += 1

    if common_length == 0:
        return 0.0

    left_is_ancestor = (
        len(left_path)
        == common_length
    )

    right_is_ancestor = (
        len(right_path)
        == common_length
    )

    ancestor_relation = (
        left_is_ancestor
        or right_is_ancestor
    )

    # Camelidae
    if common_length == 1:
        if ancestor_relation:
            return 0.45

        return 0.30

    # Camelini
    if common_length == 2:
        if ancestor_relation:
            return 0.65

        return 0.50

    # Camelus
    if common_length == 3:
        if ancestor_relation:
            return 0.90

        # Например:
        # Camelus bactrianus
        # Camelus ferus
        return 0.70

    # Фактически один и тот же вид.
    return 1.0


def taxonomy_score(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> float:
    """
    Для SemanticUnit с несколькими taxa
    берём наиболее близкую пару.
    """

    if not left or not right:
        return 0.0

    best = 0.0

    for left_taxon in left:
        for right_taxon in right:
            score = taxon_pair_score(
                left_taxon,
                right_taxon,
            )

            if score > best:
                best = score

    return best


# ============================================================
# ELASTICSEARCH
# ============================================================


async def load_units() -> list[Unit]:
    units: list[Unit] = []

    response = await es.search(
        index=SEMANTIC_UNITS_INDEX,
        size=500,
        query={
            "exists": {
                "field": "doc_embedding"
            }
        },
        sort=[
            "_doc"
        ],
        scroll="2m",
        source_includes=[
            "id",
            "text",
            "taxa",
            "source_document_id",
            "doc_embedding",
        ],
    )

    scroll_id = response.get(
        "_scroll_id"
    )

    try:
        while True:
            hits = response[
                "hits"
            ][
                "hits"
            ]

            if not hits:
                break

            for hit in hits:
                source = hit[
                    "_source"
                ]

                raw_embedding = source.get(
                    "doc_embedding"
                )

                if not raw_embedding:
                    continue

                units.append(
                    Unit(
                        id=str(
                            source.get(
                                "id",
                                hit["_id"],
                            )
                        ),
                        text=str(
                            source.get(
                                "text",
                                "",
                            )
                        ),
                        source_document_id=str(
                            source.get(
                                "source_document_id",
                                "",
                            )
                        ),
                        taxa=tuple(
                            source.get(
                                "taxa"
                            )
                            or []
                        ),
                        embedding=np.asarray(
                            raw_embedding,
                            dtype=np.float32,
                        ),
                    )
                )

            if not scroll_id:
                break

            response = await es.scroll(
                scroll_id=scroll_id,
                scroll="2m",
            )

            scroll_id = response.get(
                "_scroll_id",
                scroll_id,
            )

    finally:
        if scroll_id:
            try:
                await es.clear_scroll(
                    scroll_id=scroll_id
                )
            except Exception:
                pass

    return units


# ============================================================
# MATRICES
# ============================================================


def build_semantic_matrix(
    units: list[Unit],
) -> np.ndarray:
    matrix = np.stack(
        [
            unit.embedding
            for unit in units
        ]
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    normalized = (
        matrix
        / norms
    )

    similarities = (
        normalized
        @ normalized.T
    )

    return similarities


def build_taxonomy_matrix(
    units: list[Unit],
) -> np.ndarray:
    size = len(
        units
    )

    result = np.zeros(
        (
            size,
            size,
        ),
        dtype=np.float32,
    )

    np.fill_diagonal(
        result,
        1.0,
    )

    for left in range(
        size
    ):
        for right in range(
            left + 1,
            size,
        ):
            score = taxonomy_score(
                units[left].taxa,
                units[right].taxa,
            )

            result[
                left,
                right,
            ] = score

            result[
                right,
                left,
            ] = score

    return result


# ============================================================
# KNN GRAPH
# ============================================================


def build_knn_sets(
    semantic_matrix: np.ndarray,
    *,
    top_k: int,
) -> list[set[int]]:
    count = semantic_matrix.shape[
        0
    ]

    result: list[
        set[int]
    ] = []

    for index in range(
        count
    ):
        row = semantic_matrix[
            index
        ].copy()

        row[
            index
        ] = -np.inf

        k = min(
            top_k,
            count - 1,
        )

        if k <= 0:
            result.append(
                set()
            )
            continue

        nearest = np.argpartition(
            row,
            -k,
        )[
            -k:
        ]

        result.append(
            {
                int(item)
                for item in nearest
            }
        )

    return result


def build_graph(
    *,
    semantic_matrix: np.ndarray,
    taxonomy_matrix: np.ndarray,
    knn_sets: list[set[int]],
    semantic_floor: float,
    combined_threshold: float,
    semantic_weight: float,
    taxonomy_weight: float,
    min_taxonomy: float,
    mutual_knn: bool,
) -> list[set[int]]:
    count = semantic_matrix.shape[
        0
    ]

    graph: list[
        set[int]
    ] = [
        set()
        for _ in range(
            count
        )
    ]

    for left in range(
        count
    ):
        for right in knn_sets[
            left
        ]:
            if right <= left:
                continue

            if (
                mutual_knn
                and left
                not in knn_sets[
                    right
                ]
            ):
                continue

            semantic = float(
                semantic_matrix[
                    left,
                    right,
                ]
            )

            if semantic < semantic_floor:
                continue

            taxonomy = float(
                taxonomy_matrix[
                    left,
                    right,
                ]
            )

            if taxonomy < min_taxonomy:
                continue

            combined = (
                semantic_weight
                * semantic
                +
                taxonomy_weight
                * taxonomy
            )

            if (
                combined
                < combined_threshold
            ):
                continue

            graph[
                left
            ].add(
                right
            )

            graph[
                right
            ].add(
                left
            )

    return graph


# ============================================================
# CONNECTED COMPONENTS
# ============================================================


def connected_components(
    graph: list[set[int]],
    *,
    min_size: int,
) -> list[Cluster]:
    visited: set[int] = set()

    clusters: list[
        Cluster
    ] = []

    for start in range(
        len(
            graph
        )
    ):
        if start in visited:
            continue

        stack = [
            start
        ]

        component: list[
            int
        ] = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            component.append(
                current
            )

            for neighbor in graph[
                current
            ]:
                if neighbor not in visited:
                    stack.append(
                        neighbor
                    )

        if len(
            component
        ) >= min_size:
            clusters.append(
                Cluster(
                    ids=tuple(
                        sorted(
                            component
                        )
                    )
                )
            )

    clusters.sort(
        key=lambda cluster: len(
            cluster.ids
        ),
        reverse=True,
    )

    return clusters


# ============================================================
# REPORTING
# ============================================================


def summarize(
    *,
    clusters: list[Cluster],
    unit_count: int,
) -> dict[str, Any]:
    sizes = [
        len(
            cluster.ids
        )
        for cluster in clusters
    ]

    clustered = sum(
        sizes
    )

    return {
        "cluster_count": len(
            clusters
        ),
        "clustered_units": (
            clustered
        ),
        "unclustered_units": (
            unit_count
            - clustered
        ),
        "coverage": (
            clustered
            / unit_count
            if unit_count
            else 0.0
        ),
        "largest_cluster": (
            max(
                sizes
            )
            if sizes
            else 0
        ),
        "median_cluster_size": (
            float(
                median(
                    sizes
                )
            )
            if sizes
            else 0.0
        ),
    }


def print_summary(
    *,
    threshold: float,
    summary: dict[str, Any],
) -> None:
    print(
        f"{threshold:>9.3f} | "
        f"{summary['cluster_count']:>8} | "
        f"{summary['clustered_units']:>9} | "
        f"{summary['coverage'] * 100:>7.1f}% | "
        f"{summary['largest_cluster']:>7} | "
        f"{summary['median_cluster_size']:>6.1f}"
    )


def cluster_to_json(
    *,
    cluster: Cluster,
    units: list[Unit],
    semantic_matrix: np.ndarray,
    taxonomy_matrix: np.ndarray,
) -> dict[str, Any]:
    indices = list(
        cluster.ids
    )

    semantic_values: list[
        float
    ] = []

    taxonomy_values: list[
        float
    ] = []

    for left_position in range(
        len(
            indices
        )
    ):
        for right_position in range(
            left_position + 1,
            len(
                indices
            ),
        ):
            left = indices[
                left_position
            ]

            right = indices[
                right_position
            ]

            semantic_values.append(
                float(
                    semantic_matrix[
                        left,
                        right,
                    ]
                )
            )

            taxonomy_values.append(
                float(
                    taxonomy_matrix[
                        left,
                        right,
                    ]
                )
            )

    return {
        "size": len(
            indices
        ),
        "average_semantic_similarity": (
            sum(
                semantic_values
            )
            / len(
                semantic_values
            )
            if semantic_values
            else 1.0
        ),
        "average_taxonomy_score": (
            sum(
                taxonomy_values
            )
            / len(
                taxonomy_values
            )
            if taxonomy_values
            else 1.0
        ),
        "source_document_count": len(
            {
                units[
                    index
                ].source_document_id
                for index in indices
            }
        ),
        "taxa": sorted(
            {
                taxon
                for index in indices
                for taxon
                in units[
                    index
                ].taxa
            }
        ),
        "semantic_units": [
            {
                "id": units[
                    index
                ].id,
                "source_document_id": (
                    units[
                        index
                    ].source_document_id
                ),
                "taxa": list(
                    units[
                        index
                    ].taxa
                ),
                "text": units[
                    index
                ].text,
            }
            for index in indices
        ],
    }


# ============================================================
# CLI
# ============================================================


def parse_thresholds(
    raw: str,
) -> list[float]:
    return [
        float(
            item.strip()
        )
        for item in raw.split(
            ","
        )
        if item.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how many thematic clusters can "
            "be formed from Semantic Units."
        )
    )

    parser.add_argument(
        "--semantic-floor",
        type=float,
        default=0.72,
        help=(
            "Minimum cosine similarity for an edge. "
            "Default: 0.72"
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default=(
            "0.72,0.74,0.76,0.78,"
            "0.80,0.82,0.84,0.86"
        ),
        help=(
            "Comma-separated combined thresholds."
        ),
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--taxonomy-weight",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--min-taxonomy",
        type=float,
        default=0.30,
        help=(
            "Minimum taxonomy compatibility. "
            "0 disables taxonomy filtering."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=4,
        help=(
            "Minimum number of Semantic Units "
            "in a cluster."
        ),
    )

    parser.add_argument(
        "--non-mutual",
        action="store_true",
        help=(
            "Use ordinary KNN instead of mutual KNN."
        ),
    )

    parser.add_argument(
        "--export-threshold",
        type=float,
        default=None,
        help=(
            "If specified, export clusters for this "
            "threshold to JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "semantic_clusters_report.json"
        ),
    )

    return parser


async def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    try:
        print(
            "Loading Semantic Units..."
        )

        units = await load_units()

        if len(
            units
        ) < args.min_size:
            raise RuntimeError(
                "Too few Semantic Units with embeddings"
            )

        print(
            f"Loaded: {len(units)}"
        )

        dimensions = {
            len(
                unit.embedding
            )
            for unit in units
        }

        if len(
            dimensions
        ) != 1:
            raise RuntimeError(
                "Embeddings have different dimensions: "
                f"{sorted(dimensions)}"
            )

        print(
            "Building semantic similarity matrix..."
        )

        semantic_matrix = (
            build_semantic_matrix(
                units
            )
        )

        print(
            "Building taxonomy matrix..."
        )

        taxonomy_matrix = (
            build_taxonomy_matrix(
                units
            )
        )

        print(
            "Building KNN graph candidates..."
        )

        knn_sets = build_knn_sets(
            semantic_matrix,
            top_k=args.top_k,
        )

        thresholds = (
            parse_thresholds(
                args.thresholds
            )
        )

        print()
        print(
            "threshold | clusters | clustered | coverage | largest | median"
        )

        print(
            "-" * 69
        )

        for threshold in thresholds:
            graph = build_graph(
                semantic_matrix=(
                    semantic_matrix
                ),
                taxonomy_matrix=(
                    taxonomy_matrix
                ),
                knn_sets=knn_sets,
                semantic_floor=(
                    args.semantic_floor
                ),
                combined_threshold=(
                    threshold
                ),
                semantic_weight=(
                    args.semantic_weight
                ),
                taxonomy_weight=(
                    args.taxonomy_weight
                ),
                min_taxonomy=(
                    args.min_taxonomy
                ),
                mutual_knn=(
                    not args.non_mutual
                ),
            )

            clusters = (
                connected_components(
                    graph,
                    min_size=(
                        args.min_size
                    ),
                )
            )

            summary = summarize(
                clusters=clusters,
                unit_count=len(
                    units
                ),
            )

            print_summary(
                threshold=threshold,
                summary=summary,
            )

        if (
            args.export_threshold
            is not None
        ):
            threshold = (
                args.export_threshold
            )

            graph = build_graph(
                semantic_matrix=(
                    semantic_matrix
                ),
                taxonomy_matrix=(
                    taxonomy_matrix
                ),
                knn_sets=knn_sets,
                semantic_floor=(
                    args.semantic_floor
                ),
                combined_threshold=(
                    threshold
                ),
                semantic_weight=(
                    args.semantic_weight
                ),
                taxonomy_weight=(
                    args.taxonomy_weight
                ),
                min_taxonomy=(
                    args.min_taxonomy
                ),
                mutual_knn=(
                    not args.non_mutual
                ),
            )

            clusters = (
                connected_components(
                    graph,
                    min_size=(
                        args.min_size
                    ),
                )
            )

            report = {
                "parameters": {
                    "semantic_floor": (
                        args.semantic_floor
                    ),
                    "combined_threshold": (
                        threshold
                    ),
                    "semantic_weight": (
                        args.semantic_weight
                    ),
                    "taxonomy_weight": (
                        args.taxonomy_weight
                    ),
                    "min_taxonomy": (
                        args.min_taxonomy
                    ),
                    "top_k": (
                        args.top_k
                    ),
                    "min_size": (
                        args.min_size
                    ),
                    "mutual_knn": (
                        not args.non_mutual
                    ),
                },

                "summary": summarize(
                    clusters=clusters,
                    unit_count=len(
                        units
                    ),
                ),

                "clusters": [
                    cluster_to_json(
                        cluster=cluster,
                        units=units,
                        semantic_matrix=(
                            semantic_matrix
                        ),
                        taxonomy_matrix=(
                            taxonomy_matrix
                        ),
                    )
                    for cluster in clusters
                ],
            }

            args.output.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            print()
            print(
                "Detailed report written to:"
            )

            print(
                args.output.resolve()
            )

    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )