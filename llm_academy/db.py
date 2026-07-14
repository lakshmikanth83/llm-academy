import pathlib
import json
import datetime as _dt
import aiosqlite

DB_PATH = pathlib.Path.home() / ".llm_academy" / "data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                last_active TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS progress (
                profile_id INTEGER NOT NULL,
                topic_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'not_started',
                completed_at TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (profile_id, topic_id)
            );
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                topic_id TEXT NOT NULL,
                score INTEGER NOT NULL,
                max_score INTEGER NOT NULL,
                answers TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS flashcard_progress (
                profile_id INTEGER NOT NULL,
                topic_id TEXT NOT NULL,
                card_id TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (profile_id, topic_id, card_id)
            );
            CREATE TABLE IF NOT EXISTS xp_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                xp INTEGER NOT NULL DEFAULT 0,
                gems INTEGER NOT NULL DEFAULT 0,
                reason TEXT,
                topic_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS quest_claims (
                profile_id INTEGER NOT NULL,
                quest_id TEXT NOT NULL,
                quest_date TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (profile_id, quest_id, quest_date)
            );
            CREATE TABLE IF NOT EXISTS loot_purchases (
                profile_id INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                qty INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (profile_id, item_id)
            );
        """)
        await db.commit()

        # Additive migration for existing DBs created before gamification was added.
        for stmt in (
            "ALTER TABLE profiles ADD COLUMN xp INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE profiles ADD COLUMN gems INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                await db.execute(stmt)
            except Exception:
                pass  # column already exists
        await db.commit()


def _row(row, keys):
    return dict(zip(keys, row))


async def get_profiles():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT id, name, created_at, last_active, xp, gems FROM profiles ORDER BY last_active DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"id": r[0], "name": r[1], "created_at": r[2], "last_active": r[3], "xp": r[4], "gems": r[5]}
            for r in rows
        ]


async def create_profile(name: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cur = await db.execute(
            "INSERT INTO profiles (name) VALUES (?) RETURNING id, name, created_at, last_active, xp, gems",
            (name,),
        )
        row = await cur.fetchone()
        await db.commit()
        return {"id": row[0], "name": row[1], "created_at": row[2], "last_active": row[3], "xp": row[4], "gems": row[5]}


async def get_profile(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT id, name, created_at, last_active, xp, gems FROM profiles WHERE id=?", (profile_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "created_at": row[2], "last_active": row[3], "xp": row[4], "gems": row[5]}


async def delete_profile(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
        await db.execute("DELETE FROM progress WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM quiz_scores WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM flashcard_progress WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM xp_events WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM quest_claims WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM loot_purchases WHERE profile_id=?", (profile_id,))
        await db.commit()


async def touch_profile(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "UPDATE profiles SET last_active=datetime('now') WHERE id=?", (profile_id,)
        )
        await db.commit()


async def get_progress(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT topic_id, status, completed_at FROM progress WHERE profile_id=?", (profile_id,)
        ) as cur:
            rows = await cur.fetchall()
        return {r[0]: {"status": r[1], "completed_at": r[2]} for r in rows}


async def update_progress(profile_id: int, topic_id: str, status: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        if status == "complete":
            await db.execute(
                """INSERT INTO progress (profile_id, topic_id, status, completed_at, updated_at)
                   VALUES (?, ?, ?, datetime('now'), datetime('now'))
                   ON CONFLICT(profile_id, topic_id) DO UPDATE SET
                   status=excluded.status, completed_at=excluded.completed_at, updated_at=datetime('now')""",
                (profile_id, topic_id, status),
            )
        else:
            await db.execute(
                """INSERT INTO progress (profile_id, topic_id, status, updated_at)
                   VALUES (?, ?, ?, datetime('now'))
                   ON CONFLICT(profile_id, topic_id) DO UPDATE SET
                   status=excluded.status, updated_at=datetime('now')""",
                (profile_id, topic_id, status),
            )
        await db.commit()


async def get_stats(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT status, COUNT(*) FROM progress WHERE profile_id=? GROUP BY status", (profile_id,)
        ) as cur:
            rows = await cur.fetchall()
        counts = {r[0]: r[1] for r in rows}

        async with db.execute(
            "SELECT AVG(CAST(score AS FLOAT) / max_score * 100) FROM quiz_scores WHERE profile_id=?",
            (profile_id,),
        ) as cur:
            avg_row = await cur.fetchone()

    return {
        "total": 64,
        "completed": counts.get("complete", 0),
        "saved": counts.get("saved", 0),
        "in_progress": counts.get("in_progress", 0),
        "quiz_average": round(avg_row[0], 1) if avg_row and avg_row[0] is not None else None,
    }


async def save_quiz(profile_id: int, topic_id: str, score: int, max_score: int, answers: list):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT INTO quiz_scores (profile_id, topic_id, score, max_score, answers) VALUES (?,?,?,?,?)",
            (profile_id, topic_id, score, max_score, json.dumps(answers)),
        )
        await db.commit()


async def get_quiz_history(profile_id: int, topic_id: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT score, max_score, created_at FROM quiz_scores WHERE profile_id=? AND topic_id=? ORDER BY created_at DESC",
            (profile_id, topic_id),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {"score": r[0], "max": r[1], "percentage": round(r[0] / r[1] * 100, 1), "created_at": r[2]}
            for r in rows
        ]


async def get_quiz_attempt_count(profile_id: int, topic_id: str) -> int:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM quiz_scores WHERE profile_id=? AND topic_id=?", (profile_id, topic_id)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def has_good_quiz_today(profile_id: int, min_pct: float = 80.0) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            """SELECT 1 FROM quiz_scores WHERE profile_id=? AND DATE(created_at)=DATE('now')
               AND (CAST(score AS FLOAT) / max_score * 100) >= ? LIMIT 1""",
            (profile_id, min_pct),
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def has_perfect_quiz(profile_id: int) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT 1 FROM quiz_scores WHERE profile_id=? AND score=max_score LIMIT 1", (profile_id,)
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def update_flashcard(profile_id: int, topic_id: str, card_id: str, status: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO flashcard_progress (profile_id, topic_id, card_id, status, updated_at)
               VALUES (?,?,?,?,datetime('now'))
               ON CONFLICT(profile_id, topic_id, card_id) DO UPDATE SET
               status=excluded.status, updated_at=datetime('now')""",
            (profile_id, topic_id, card_id, status),
        )
        await db.commit()


async def get_flashcards(profile_id: int, topic_id: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT card_id, status FROM flashcard_progress WHERE profile_id=? AND topic_id=?",
            (profile_id, topic_id),
        ) as cur:
            rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}


async def get_flashcard_review_count(profile_id: int) -> int:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM flashcard_progress WHERE profile_id=?", (profile_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


# ── Gamification: XP / gems ledger ─────────────────────────────────
async def grant_xp(profile_id: int, xp: int = 0, gems: int = 0, reason: str = "", topic_id: str | None = None):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT INTO xp_events (profile_id, xp, gems, reason, topic_id) VALUES (?,?,?,?,?)",
            (profile_id, xp, gems, reason, topic_id),
        )
        await db.execute(
            "UPDATE profiles SET xp = xp + ?, gems = gems + ? WHERE id=?",
            (xp, gems, profile_id),
        )
        await db.commit()
    return await get_wallet(profile_id)


async def get_wallet(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute("SELECT xp, gems FROM profiles WHERE id=?", (profile_id,)) as cur:
            row = await cur.fetchone()
    return {"xp": row[0] if row else 0, "gems": row[1] if row else 0}


async def get_xp_today(profile_id: int) -> int:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT COALESCE(SUM(xp),0) FROM xp_events WHERE profile_id=? AND DATE(created_at)=DATE('now')",
            (profile_id,),
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def get_completed_topic_ids(profile_id: int) -> set:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT topic_id FROM progress WHERE profile_id=? AND status='complete'", (profile_id,)
        ) as cur:
            rows = await cur.fetchall()
    return {r[0] for r in rows}


async def get_max_daily_completions(profile_id: int) -> int:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            """SELECT COUNT(*) c FROM progress
               WHERE profile_id=? AND completed_at IS NOT NULL
               GROUP BY DATE(completed_at) ORDER BY c DESC LIMIT 1""",
            (profile_id,),
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def has_night_owl_activity(profile_id: int) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            """SELECT 1 FROM progress WHERE profile_id=? AND completed_at IS NOT NULL
               AND CAST(strftime('%H', completed_at) AS INTEGER) BETWEEN 0 AND 4 LIMIT 1""",
            (profile_id,),
        ) as cur:
            row = await cur.fetchone()
    return row is not None


async def get_activity_dates(profile_id: int) -> set:
    dates = set()
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT DISTINCT DATE(completed_at) d FROM progress WHERE profile_id=? AND completed_at IS NOT NULL",
            (profile_id,),
        ) as cur:
            for r in await cur.fetchall():
                if r[0]:
                    dates.add(r[0])
        async with db.execute(
            "SELECT DISTINCT DATE(created_at) d FROM quiz_scores WHERE profile_id=?", (profile_id,)
        ) as cur:
            for r in await cur.fetchall():
                if r[0]:
                    dates.add(r[0])
        async with db.execute(
            "SELECT DISTINCT DATE(updated_at) d FROM flashcard_progress WHERE profile_id=?", (profile_id,)
        ) as cur:
            for r in await cur.fetchall():
                if r[0]:
                    dates.add(r[0])
    return dates


async def compute_streak(profile_id: int) -> int:
    dates = await get_activity_dates(profile_id)
    owned = await get_loot_owned(profile_id)
    freezes_left = owned.get("streak_freeze", 0)

    today = _dt.datetime.utcnow().date()  # matches sqlite's UTC-based datetime('now')
    streak = 0
    d = today
    while True:
        ds = d.isoformat()
        if ds in dates:
            streak += 1
            d -= _dt.timedelta(days=1)
            continue
        if d == today:
            # Today not logged yet — don't break the streak, just skip to yesterday.
            d -= _dt.timedelta(days=1)
            continue
        if freezes_left > 0:
            freezes_left -= 1
            d -= _dt.timedelta(days=1)
            continue
        break
    return streak


# ── Gamification: quests ───────────────────────────────────────────
async def get_claimed_quests(profile_id: int, quest_date: str) -> set:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT quest_id FROM quest_claims WHERE profile_id=? AND quest_date=?", (profile_id, quest_date)
        ) as cur:
            rows = await cur.fetchall()
    return {r[0] for r in rows}


async def claim_quest_db(profile_id: int, quest_id: str, quest_date: str) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cur = await db.execute(
            "INSERT OR IGNORE INTO quest_claims (profile_id, quest_id, quest_date) VALUES (?,?,?)",
            (profile_id, quest_id, quest_date),
        )
        await db.commit()
        return cur.rowcount > 0


# ── Gamification: loot / shop ──────────────────────────────────────
async def get_loot_owned(profile_id: int) -> dict:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT item_id, qty FROM loot_purchases WHERE profile_id=?", (profile_id,)
        ) as cur:
            rows = await cur.fetchall()
    return {r[0]: r[1] for r in rows}


async def purchase_loot_db(profile_id: int, item_id: str, cost_gems: int) -> bool:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute("SELECT gems FROM profiles WHERE id=?", (profile_id,)) as cur:
            row = await cur.fetchone()
        gems = row[0] if row else 0
        if gems < cost_gems:
            return False
        await db.execute("UPDATE profiles SET gems = gems - ? WHERE id=?", (cost_gems, profile_id))
        await db.execute(
            """INSERT INTO loot_purchases (profile_id, item_id, qty, updated_at) VALUES (?,?,1,datetime('now'))
               ON CONFLICT(profile_id, item_id) DO UPDATE SET qty=qty+1, updated_at=datetime('now')""",
            (profile_id, item_id),
        )
        await db.commit()
    return True
