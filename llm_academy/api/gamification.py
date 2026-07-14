import json
import pathlib
import datetime as _dt

from fastapi import APIRouter, HTTPException

from llm_academy import db

router = APIRouter(tags=["gamification"])

CONTENT_DIR = pathlib.Path(__file__).parent.parent / "content"

RANKS = ["Curious Mind", "Token Rookie", "Prompt Apprentice", "Prompt Adept",
         "RAG Ranger", "Agent Architect", "Eval Expert", "LLM Sage"]
RANK_STEP = 800
DAILY_GOAL_XP = 200

LOOT_CATALOG = {
    "aurora_theme": {"icon": "🎨", "name": "Aurora Theme", "cost": 200,
                      "desc": "Unlocks an alternate accent theme you can switch to anytime."},
    "dragon_avatar": {"icon": "🐉", "name": "Dragon Avatar", "cost": 500,
                       "desc": "Swaps your profile avatar for a dragon."},
    "streak_freeze": {"icon": "❄️", "name": "Streak Freeze", "cost": 120,
                       "desc": "Forgives one missed day so your streak doesn't reset."},
    "golden_ada": {"icon": "✨", "name": "Golden Ada", "cost": 800,
                   "desc": "Gives your home screen mascot a golden glow."},
}


def _levels():
    levels_file = CONTENT_DIR / "levels.json"
    if not levels_file.exists():
        return []
    with open(levels_file) as f:
        return json.load(f)


def _topic_ids_for_level(level_num: int) -> set:
    for lv in _levels():
        if lv["level"] == level_num:
            return {t["id"] for t in lv.get("topics", [])}
    return set()


async def _badges(profile_id: int, streak: int, rank_index: int):
    completed = await db.get_completed_topic_ids(profile_id)
    perfect = await db.has_perfect_quiz(profile_id)
    night_owl = await db.has_night_owl_activity(profile_id)
    max_daily = await db.get_max_daily_completions(profile_id)
    flashcards_reviewed = await db.get_flashcard_review_count(profile_id)

    rag_ids = _topic_ids_for_level(5)
    agent_ids = _topic_ids_for_level(6)
    safety_ids = _topic_ids_for_level(8)

    defs = [
        ("first_steps", "👣", "First Steps", "Finish your first topic", len(completed) >= 1),
        ("tokenizer", "🧱", "Tokenizer", "Master tokenization", "topic_06" in completed),
        ("week_warrior", "🔥", "Week Warrior", "7-day learning streak", streak >= 7),
        ("flawless", "💯", "Flawless", "Perfect score on a quiz", perfect),
        ("night_owl", "🦉", "Night Owl", "Learn after midnight", night_owl),
        ("fast_learner", "🚀", "Fast Learner", "3 topics in one day", max_daily >= 3),
        ("rag_ready", "📖", "RAG Ready", "Finish the RAG world", bool(rag_ids) and rag_ids.issubset(completed)),
        ("agent_architect", "🦾", "Agent Architect", "Build a multi-agent flow", bool(agent_ids) and agent_ids.issubset(completed)),
        ("guardian", "🛡️", "Guardian", "Ace the Safety world", bool(safety_ids) and safety_ids.issubset(completed)),
        ("capstone", "🎓", "Capstone", "Ship the capstone project", "topic_61" in completed),
        ("deep_thinker", "🧠", "Deep Thinker", "100 flashcards reviewed", flashcards_reviewed >= 100),
        ("llm_sage", "👑", "LLM Sage", "Reach max rank", rank_index >= len(RANKS) - 1),
    ]
    return [{"id": i, "icon": ic, "name": n, "desc": d, "got": g} for i, ic, n, d, g in defs]


async def _quests(profile_id: int, today: str):
    xp_today = await db.get_xp_today(profile_id)

    # Count topics completed today (progress.completed_at date == today).
    progress = await db.get_progress(profile_id)
    completed_today = sum(
        1 for t in progress.values()
        if t.get("status") == "complete" and (t.get("completed_at") or "").startswith(today)
    )

    good_quiz_today = await db.has_good_quiz_today(profile_id)
    claimed = await db.get_claimed_quests(profile_id, today)

    quests_def = [
        ("q1", "📘", "Complete 1 topic", completed_today, 1, 50),
        ("q2", "🎯", "Score 80%+ on a quiz", 1 if good_quiz_today else 0, 1, 80),
        ("q3", "⚡", f"Earn {DAILY_GOAL_XP} XP today", xp_today, DAILY_GOAL_XP, 60),
    ]
    quests = []
    for qid, icon, title, cur, mx, reward in quests_def:
        cur = min(cur, mx)
        done = qid in claimed
        quests.append({
            "id": qid, "icon": icon, "title": title, "cur": cur, "max": mx,
            "reward": reward, "claimed": done, "claimable": (not done and cur >= mx),
        })
    return quests


@router.get("/gamification/{profile_id}")
async def get_gamification(profile_id: int):
    profile = await db.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    wallet = await db.get_wallet(profile_id)
    xp = wallet["xp"]
    gems = wallet["gems"]
    streak = await db.compute_streak(profile_id)
    xp_today = await db.get_xp_today(profile_id)

    rank_index = min(len(RANKS) - 1, xp // RANK_STEP)
    xp_into = xp - rank_index * RANK_STEP
    xp_need = RANK_STEP
    rank_next = RANKS[min(len(RANKS) - 1, rank_index + 1)]

    utc_today = _dt.datetime.utcnow().date()  # matches sqlite's UTC-based datetime('now')
    today = utc_today.isoformat()
    iso_year, iso_week, _ = utc_today.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"

    quests = await _quests(profile_id, today)

    weekly_cur = min(streak, 5)
    weekly_claimed_set = await db.get_claimed_quests(profile_id, week_key)
    weekly_claimed = "weekly_5day" in weekly_claimed_set
    weekly = {
        "id": "weekly_5day", "title": "Learn 5 days in a row", "cur": weekly_cur, "max": 5,
        "reward_gems": 150, "claimed": weekly_claimed, "claimable": (not weekly_claimed and weekly_cur >= 5),
    }

    badges = await _badges(profile_id, streak, rank_index)
    owned = await db.get_loot_owned(profile_id)
    loot = [
        {"id": item_id, **info, "owned": owned.get(item_id, 0) > 0, "qty": owned.get(item_id, 0)}
        for item_id, info in LOOT_CATALOG.items()
    ]

    return {
        "xp": xp, "gems": gems, "streak": streak,
        "xp_today": xp_today, "daily_goal": DAILY_GOAL_XP,
        "rank_index": rank_index, "rank_name": RANKS[rank_index], "rank_next": rank_next,
        "xp_into_rank": xp_into, "xp_need_for_rank": xp_need,
        "quests": quests, "weekly": weekly,
        "badges": badges, "badges_unlocked": sum(1 for b in badges if b["got"]), "badges_total": len(badges),
        "loot": loot,
    }


@router.post("/gamification/{profile_id}/quests/{quest_id}/claim")
async def claim_quest(profile_id: int, quest_id: str):
    profile = await db.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")

    utc_today = _dt.datetime.utcnow().date()  # matches sqlite's UTC-based datetime('now')
    today = utc_today.isoformat()
    iso_year, iso_week, _ = utc_today.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    streak = await db.compute_streak(profile_id)

    if quest_id == "weekly_5day":
        if min(streak, 5) < 5:
            raise HTTPException(400, "Quest not finished yet")
        claimed = await db.claim_quest_db(profile_id, quest_id, week_key)
        if not claimed:
            raise HTTPException(400, "Already claimed")
        wallet = await db.grant_xp(profile_id, xp=0, gems=150, reason="quest:weekly_5day")
        return {"ok": True, **wallet}

    quests = await _quests(profile_id, today)
    match = next((q for q in quests if q["id"] == quest_id), None)
    if not match:
        raise HTTPException(404, "Unknown quest")
    if match["cur"] < match["max"]:
        raise HTTPException(400, "Quest not finished yet")
    if match["claimed"]:
        raise HTTPException(400, "Already claimed")

    claimed = await db.claim_quest_db(profile_id, quest_id, today)
    if not claimed:
        raise HTTPException(400, "Already claimed")
    wallet = await db.grant_xp(profile_id, xp=match["reward"], gems=0, reason=f"quest:{quest_id}")
    return {"ok": True, "xp_gain": match["reward"], "gems_gain": 0, **wallet}


@router.post("/gamification/{profile_id}/loot/{item_id}/purchase")
async def purchase_loot(profile_id: int, item_id: str):
    profile = await db.get_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    item = LOOT_CATALOG.get(item_id)
    if not item:
        raise HTTPException(404, "Unknown item")
    ok = await db.purchase_loot_db(profile_id, item_id, item["cost"])
    if not ok:
        raise HTTPException(400, "Not enough gems")
    wallet = await db.get_wallet(profile_id)
    return {"ok": True, **wallet}
