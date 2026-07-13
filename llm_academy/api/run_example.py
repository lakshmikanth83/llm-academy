from fastapi import APIRouter
from pydantic import BaseModel

from llm_academy.examples.runner import run_topic_example

router = APIRouter(tags=["run"])


class RunRequest(BaseModel):
    api_key: str | None = None


@router.post("/run/{topic_id}")
async def run_example(topic_id: str, body: RunRequest):
    return run_topic_example(topic_id, body.api_key)
