import json
import pathlib

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["content"])

CONTENT_DIR = pathlib.Path(__file__).parent.parent / "content"


@router.get("/content/levels")
async def get_levels():
    levels_file = CONTENT_DIR / "levels.json"
    if not levels_file.exists():
        return []
    with open(levels_file) as f:
        return json.load(f)


@router.get("/content/topics/{topic_id}")
async def get_topic(topic_id: str):
    topic_file = CONTENT_DIR / f"{topic_id}.json"
    if not topic_file.exists():
        raise HTTPException(404, f"Topic {topic_id} not found")
    with open(topic_file) as f:
        return json.load(f)


@router.get("/content/search")
async def search(q: str = Query(..., min_length=1)):
    levels_file = CONTENT_DIR / "levels.json"
    if not levels_file.exists():
        return []
    with open(levels_file) as f:
        levels = json.load(f)

    q_lower = q.lower()
    results = []
    for level in levels:
        for topic in level.get("topics", []):
            if q_lower in topic["title"].lower():
                results.append({
                    "id": topic["id"],
                    "title": topic["title"],
                    "level": level["level"],
                    "level_name": level["name"],
                    "emoji": topic.get("emoji", ""),
                })
    return results
