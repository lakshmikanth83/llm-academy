from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm_academy.db import get_profiles, create_profile, get_profile, delete_profile, touch_profile

router = APIRouter(tags=["profiles"])


class ProfileCreate(BaseModel):
    name: str


@router.get("/profiles")
async def list_profiles():
    return await get_profiles()


@router.post("/profiles", status_code=201)
async def new_profile(body: ProfileCreate):
    if not body.name.strip():
        raise HTTPException(400, "Name cannot be empty")
    return await create_profile(body.name.strip())


@router.get("/profiles/{profile_id}")
async def get_one(profile_id: int):
    p = await get_profile(profile_id)
    if not p:
        raise HTTPException(404, "Profile not found")
    return p


@router.delete("/profiles/{profile_id}")
async def remove_profile(profile_id: int):
    await delete_profile(profile_id)
    return {"ok": True}


@router.put("/profiles/{profile_id}/active")
async def mark_active(profile_id: int):
    await touch_profile(profile_id)
    return {"ok": True}
