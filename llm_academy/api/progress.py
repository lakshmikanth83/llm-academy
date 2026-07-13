from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm_academy.db import get_progress, update_progress, get_stats

router = APIRouter(tags=["progress"])

VALID_STATUSES = {"not_started", "in_progress", "complete", "saved"}


class ProgressUpdate(BaseModel):
    status: str


@router.get("/progress/{profile_id}")
async def get_all_progress(profile_id: int):
    return {"topics": await get_progress(profile_id)}


@router.post("/progress/{profile_id}/{topic_id}")
async def set_progress(profile_id: int, topic_id: str, body: ProgressUpdate):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")
    await update_progress(profile_id, topic_id, body.status)
    return {"ok": True}


@router.get("/progress/{profile_id}/stats")
async def stats(profile_id: int):
    return await get_stats(profile_id)
