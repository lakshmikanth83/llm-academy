from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm_academy.db import get_progress, update_progress, get_stats, grant_xp

router = APIRouter(tags=["progress"])

VALID_STATUSES = {"not_started", "in_progress", "complete", "saved"}

TOPIC_COMPLETE_XP = 80
TOPIC_COMPLETE_GEMS = 10


class ProgressUpdate(BaseModel):
    status: str


@router.get("/progress/{profile_id}")
async def get_all_progress(profile_id: int):
    return {"topics": await get_progress(profile_id)}


@router.post("/progress/{profile_id}/{topic_id}")
async def set_progress(profile_id: int, topic_id: str, body: ProgressUpdate):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")

    was_complete = False
    if body.status == "complete":
        existing = await get_progress(profile_id)
        was_complete = (existing.get(topic_id) or {}).get("status") == "complete"

    await update_progress(profile_id, topic_id, body.status)

    reward = None
    if body.status == "complete" and not was_complete:
        wallet = await grant_xp(
            profile_id, xp=TOPIC_COMPLETE_XP, gems=TOPIC_COMPLETE_GEMS,
            reason="topic_complete", topic_id=topic_id,
        )
        reward = {"xp_gain": TOPIC_COMPLETE_XP, "gems_gain": TOPIC_COMPLETE_GEMS, **wallet}

    return {"ok": True, "reward": reward}


@router.get("/progress/{profile_id}/stats")
async def stats(profile_id: int):
    return await get_stats(profile_id)
