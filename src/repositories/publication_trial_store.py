from __future__ import annotations

import asyncio
import json

from datetime import datetime, timezone
from pathlib import Path

from src.domain.publication_trial import (
    PublicationTrial,
    PublicationTrialDecision,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_DIR = (
    PROJECT_ROOT
    / "var"
    / "publication_trials"
)


class FilePublicationTrialStore:
    """
    Temporary test storage.

    Each generated post is one JSON file. Nothing is written
    to Elasticsearch.
    """

    def __init__(
        self,
        directory: Path = DEFAULT_STORAGE_DIR,
    ) -> None:
        self.directory = directory

    async def save(
        self,
        trial: PublicationTrial,
    ) -> PublicationTrial:
        await asyncio.to_thread(
            self._save_sync,
            trial,
        )
        return trial

    async def get(
        self,
        trial_id: str,
    ) -> PublicationTrial | None:
        return await asyncio.to_thread(
            self._get_sync,
            trial_id,
        )

    async def list_recent(
        self,
        *,
        limit: int = 50,
    ) -> list[PublicationTrial]:
        return await asyncio.to_thread(
            self._list_recent_sync,
            limit,
        )

    async def set_decision(
        self,
        *,
        trial_id: str,
        decision: PublicationTrialDecision,
        note: str | None = None,
    ) -> PublicationTrial:
        trial = await self.get(
            trial_id
        )

        if trial is None:
            raise ValueError(
                "Тестовая публикация не найдена"
            )

        now = datetime.now(
            timezone.utc
        )

        updated = trial.model_copy(
            update={
                "decision": decision,
                "decision_note": (
                    note.strip()
                    if note
                    else None
                ),
                "decided_at": now,
                "updated_at": now,
            }
        )

        # Revalidate after model_copy update.
        updated = PublicationTrial.model_validate(
            updated.model_dump(
                mode="json"
            )
        )

        await self.save(
            updated
        )

        return updated

    def _save_sync(
        self,
        trial: PublicationTrial,
    ) -> None:
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self._path_for(
            trial.id
        )

        temporary = path.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(
                trial.model_dump(
                    mode="json"
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

    def _get_sync(
        self,
        trial_id: str,
    ) -> PublicationTrial | None:
        path = self._path_for(
            trial_id
        )

        if not path.exists():
            return None

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return PublicationTrial.model_validate(
            data
        )

    def _list_recent_sync(
        self,
        limit: int,
    ) -> list[PublicationTrial]:
        if not self.directory.exists():
            return []

        items: list[PublicationTrial] = []

        for path in self.directory.glob(
            "publication_*.json"
        ):
            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                items.append(
                    PublicationTrial.model_validate(
                        data
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
                ValueError,
            ):
                # One broken test file should not break the admin.
                continue

        items.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        return items[
            :max(
                0,
                limit,
            )
        ]

    def _path_for(
        self,
        trial_id: str,
    ) -> Path:
        if (
            not trial_id
            or "/" in trial_id
            or "\\" in trial_id
            or ".." in trial_id
        ):
            raise ValueError(
                "Некорректный ID тестовой публикации"
            )

        return (
            self.directory
            / f"{trial_id}.json"
        )
