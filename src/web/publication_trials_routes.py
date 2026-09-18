from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    Request,
)
from fastapi.responses import (
    RedirectResponse,
)
from fastapi.templating import (
    Jinja2Templates,
)

from src.domain.publication_trial import (
    PublicationTrial,
    PublicationTrialDecision,
)
from src.infrastructure.llm.deepseek_post_generator import (
    DeepSeekPostGenerator,
)
from src.repositories.publication_trial_store import (
    FilePublicationTrialStore,
)
from src.services.topic_discovery_service import (
    TopicDiscoveryService,
)


def create_publication_trials_router(
    *,
    templates: Jinja2Templates,
    discovery_service: TopicDiscoveryService,
    post_generator: DeepSeekPostGenerator,
    trial_store: FilePublicationTrialStore,
) -> APIRouter:
    router = APIRouter()

    async def render_list(
        *,
        request: Request,
        error: str | None = None,
    ):
        trials = await trial_store.list_recent(
            limit=100
        )

        return templates.TemplateResponse(
            request=request,
            name="publication_trials.html",
            context={
                "trials": trials,
                "error": error,
                "decisions": (
                    PublicationTrialDecision
                ),
            },
        )

    @router.get(
        "/publications"
    )
    async def publications_page(
        request: Request,
    ):
        return await render_list(
            request=request
        )

    @router.post(
        "/publications/generate"
    )
    async def generate_publication(
        request: Request,
    ):
        try:
            discovery = (
                await discovery_service.discover()
            )

            generated = (
                await post_generator.generate(
                    evidence=(
                        discovery.evidence
                    )
                )
            )

            selected_ids = [
                item.semantic_unit_id
                for item
                in discovery.evidence
            ]

            trial = PublicationTrial(
                id=(
                    "publication_"
                    + uuid4().hex
                ),
                topic=(
                    generated.post.topic
                ),
                title=(
                    generated.post.title
                ),
                post_text=(
                    generated.post.post_text
                ),
                selected_semantic_unit_ids=(
                    selected_ids
                ),
                used_semantic_unit_ids=(
                    generated
                    .post
                    .used_semantic_unit_ids
                ),
                evidence=(
                    discovery.evidence
                ),
                discovery=(
                    discovery.metrics
                ),
                generation_model=(
                    generated.model
                ),
                prompt_file=(
                    generated.prompt_file
                ),
                prompt_sha256=(
                    generated.prompt_sha256
                ),
                metadata={
                    "llm_calls": 1,
                    "storage": (
                        "local_json"
                    ),
                },
            )

            await trial_store.save(
                trial
            )

            return RedirectResponse(
                (
                    "/publications/"
                    f"{trial.id}"
                ),
                status_code=303,
            )

        except Exception as exc:
            return await render_list(
                request=request,
                error=(
                    "Не удалось сгенерировать "
                    f"публикацию: {exc}"
                ),
            )

    @router.get(
        "/publications/{trial_id}"
    )
    async def publication_detail(
        trial_id: str,
        request: Request,
    ):
        trial = await trial_store.get(
            trial_id
        )

        if trial is None:
            return RedirectResponse(
                "/publications",
                status_code=303,
            )

        used_ids = set(
            trial.used_semantic_unit_ids
        )

        evidence_rows: list[
            dict[str, Any]
        ] = [
            {
                "item": item,
                "used": (
                    item.semantic_unit_id
                    in used_ids
                ),
            }
            for item in trial.evidence
        ]

        return templates.TemplateResponse(
            request=request,
            name=(
                "publication_trial_detail.html"
            ),
            context={
                "trial": trial,
                "evidence_rows": (
                    evidence_rows
                ),
                "decisions": (
                    PublicationTrialDecision
                ),
            },
        )

    @router.post(
        "/publications/{trial_id}/decision"
    )
    async def publication_decision(
        trial_id: str,
        request: Request,
    ):
        form = await request.form()

        try:
            decision = (
                PublicationTrialDecision(
                    str(
                        form[
                            "decision"
                        ]
                    )
                )
            )

            if (
                decision
                == PublicationTrialDecision.PENDING
            ):
                raise ValueError(
                    "Нужно выбрать решение"
                )

            await trial_store.set_decision(
                trial_id=trial_id,
                decision=decision,
                note=(
                    str(
                        form.get(
                            "decision_note",
                            "",
                        )
                    )
                ),
            )

        except (
            ValueError,
            KeyError,
        ):
            return RedirectResponse(
                (
                    "/publications/"
                    f"{trial_id}"
                ),
                status_code=303,
            )

        return RedirectResponse(
            (
                "/publications/"
                f"{trial_id}"
            ),
            status_code=303,
        )

    return router
