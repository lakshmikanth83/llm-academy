import pathlib
import json
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
        """)
        await db.commit()


def _row(row, keys):
    return dict(zip(keys, row))


async def get_profiles():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute("SELECT id, name, created_at, last_active FROM profiles ORDER BY last_active DESC") as cur:
            rows = await cur.fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2], "last_active": r[3]} for r in rows]


async def create_profile(name: str):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cur = await db.execute(
            "INSERT INTO profiles (name) VALUES (?) RETURNING id, name, created_at, last_active", (name,)
        )
        row = await cur.fetchone()
        await db.commit()
        return {"id": row[0], "name": row[1], "created_at": row[2], "last_active": row[3]}


async def get_profile(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        async with db.execute(
            "SELECT id, name, created_at, last_active FROM profiles WHERE id=?", (profile_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "created_at": row[2], "last_active": row[3]}


async def delete_profile(profile_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
        await db.execute("DELETE FROM progress WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM quiz_scores WHERE profile_id=?", (profile_id,))
        await db.execute("DELETE FROM flashcard_progress WHERE profile_id=?", (profile_id,))
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
    completed_at = "datetime('now')" if status == "complete" else "NULL"
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
