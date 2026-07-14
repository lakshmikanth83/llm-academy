import json
import pathlib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_academy.db import save_quiz, get_quiz_history, get_quiz_attempt_count, grant_xp

router = APIRouter(tags=["quiz"])

CONTENT_DIR = pathlib.Path(__file__).parent.parent / "content"

PASS_THRESHOLD = 80.0
FIRST_ATTEMPT_PASS_XP, FIRST_ATTEMPT_PASS_GEMS = 120, 15
FIRST_ATTEMPT_FAIL_XP, FIRST_ATTEMPT_FAIL_GEMS = 40, 5
REPEAT_ATTEMPT_XP, REPEAT_ATTEMPT_GEMS = 10, 0


class QuizSubmit(BaseModel):
    answers: list[int]


@router.post("/quiz/{profile_id}/{topic_id}/submit")
async def submit_quiz(profile_id: int, topic_id: str, body: QuizSubmit):
    topic_file = CONTENT_DIR / f"{topic_id}.json"
    if not topic_file.exists():
        raise HTTPException(404, f"Topic {topic_id} not found")

    with open(topic_file) as f:
        topic = json.load(f)

    mcqs = topic.get("mcqs", [])
    if not mcqs:
        raise HTTPException(400, "This topic has no quiz questions")

    if len(body.answers) != len(mcqs):
        raise HTTPException(400, f"Expected {len(mcqs)} answers, got {len(body.answers)}")

    results = []
    score = 0
    for q, given in zip(mcqs, body.answers):
        correct = given == q["correct"]
        if correct:
            score += 1
        results.append({"correct": correct, "explanation": q.get("explanation", "")})

    max_score = len(mcqs)
    percentage = round(score / max_score * 100, 1)

    prior_attempts = await get_quiz_attempt_count(profile_id, topic_id)
    await save_quiz(profile_id, topic_id, score, max_score, body.answers)

    if prior_attempts == 0:
        passed = percentage >= PASS_THRESHOLD
        xp_gain = FIRST_ATTEMPT_PASS_XP if passed else FIRST_ATTEMPT_FAIL_XP
        gems_gain = FIRST_ATTEMPT_PASS_GEMS if passed else FIRST_ATTEMPT_FAIL_GEMS
    else:
        xp_gain, gems_gain = REPEAT_ATTEMPT_XP, REPEAT_ATTEMPT_GEMS

    wallet = await grant_xp(profile_id, xp=xp_gain, gems=gems_gain, reason="quiz_submit", topic_id=topic_id)

    return {
        "score": score,
        "max": max_score,
        "percentage": percentage,
        "results": results,
        "reward": {"xp_gain": xp_gain, "gems_gain": gems_gain, **wallet},
    }


@router.get("/quiz/{profile_id}/{topic_id}/history")
async def quiz_history(profile_id: int, topic_id: str):
    return await get_quiz_history(profile_id, topic_id)
