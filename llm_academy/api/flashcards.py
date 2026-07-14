from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_academy.db import update_flashcard, get_flashcards

router = APIRouter(tags=["flashcards"])

VALID_STATUSES = {"know", "review"}


class FlashcardUpdate(BaseModel):
    status: str


@router.post("/flashcards/{profile_id}/{topic_id}/{card_id}")
async def set_flashcard(profile_id: int, topic_id: str, card_id: str, body: FlashcardUpdate):
    if body.status not in VALID_STATUSES:
        raise HTTPException(400, f"status must be one of {VALID_STATUSES}")
    await update_flashcard(profile_id, topic_id, card_id, body.status)
    return {"ok": True}


@router.get("/flashcards/{profile_id}/{topic_id}")
async def list_flashcards(profile_id: int, topic_id: str):
    return await get_flashcards(profile_id, topic_id)
