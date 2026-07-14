"""Basic API tests for llm-academy."""
import pytest
from fastapi.testclient import TestClient

from llm_academy.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Profiles ──────────────────────────────────────────────────────
def test_create_and_list_profiles(client):
    r = client.post("/api/profiles", json={"name": "Test User"})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "Test User"
    assert "id" in data

    r2 = client.get("/api/profiles")
    assert r2.status_code == 200
    names = [p["name"] for p in r2.json()]
    assert "Test User" in names


def test_get_profile(client):
    r = client.post("/api/profiles", json={"name": "Get Me"})
    pid = r.json()["id"]

    r2 = client.get(f"/api/profiles/{pid}")
    assert r2.status_code == 200
    assert r2.json()["id"] == pid


def test_delete_profile(client):
    r = client.post("/api/profiles", json={"name": "Delete Me"})
    pid = r.json()["id"]

    r2 = client.delete(f"/api/profiles/{pid}")
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    r3 = client.get(f"/api/profiles/{pid}")
    assert r3.status_code == 404


def test_create_profile_empty_name(client):
    r = client.post("/api/profiles", json={"name": ""})
    assert r.status_code == 400


# ── Content ───────────────────────────────────────────────────────
def test_get_levels(client):
    r = client.get("/api/content/levels")
    assert r.status_code == 200
    levels = r.json()
    # levels.json may not exist during unit tests; accept empty list too
    assert isinstance(levels, list)


def test_get_topic_not_found(client):
    r = client.get("/api/content/topics/topic_99")
    assert r.status_code == 404


def test_search(client):
    r = client.get("/api/content/search?q=AI")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── Progress ──────────────────────────────────────────────────────
def test_progress_lifecycle(client):
    pid = client.post("/api/profiles", json={"name": "Progress Tester"}).json()["id"]

    # Get empty progress
    r = client.get(f"/api/progress/{pid}")
    assert r.status_code == 200
    assert r.json()["topics"] == {}

    # Set in_progress
    r2 = client.post(f"/api/progress/{pid}/topic_01", json={"status": "in_progress"})
    assert r2.status_code == 200
    assert r2.json()["ok"] is True

    # Verify
    r3 = client.get(f"/api/progress/{pid}")
    assert r3.json()["topics"]["topic_01"]["status"] == "in_progress"

    # Mark complete
    client.post(f"/api/progress/{pid}/topic_01", json={"status": "complete"})
    r4 = client.get(f"/api/progress/{pid}")
    assert r4.json()["topics"]["topic_01"]["status"] == "complete"


def test_stats(client):
    pid = client.post("/api/profiles", json={"name": "Stats Tester"}).json()["id"]
    r = client.get(f"/api/progress/{pid}/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 64
    assert data["completed"] == 0


def test_invalid_progress_status(client):
    pid = client.post("/api/profiles", json={"name": "Bad Status"}).json()["id"]
    r = client.post(f"/api/progress/{pid}/topic_01", json={"status": "invalid_status"})
    assert r.status_code == 400


# ── Run example ───────────────────────────────────────────────────
def test_run_example_no_practical(client):
    r = client.post("/api/run/topic_99", json={})
    assert r.status_code == 200
    data = r.json()
    assert "output" in data
    assert data["error"] is None


# ── Gamification ──────────────────────────────────────────────────
def test_gamification_summary_shape(client):
    pid = client.post("/api/profiles", json={"name": "Gamer"}).json()["id"]
    r = client.get(f"/api/gamification/{pid}")
    assert r.status_code == 200
    data = r.json()
    for key in ("xp", "gems", "streak", "rank_name", "quests", "weekly", "badges", "loot"):
        assert key in data
    assert data["xp"] == 0 and data["gems"] == 0
    assert data["badges_total"] == len(data["badges"])
    assert not any(b["got"] for b in data["badges"])


def test_topic_complete_grants_xp_and_unlocks_first_steps_badge(client):
    pid = client.post("/api/profiles", json={"name": "Completer"}).json()["id"]
    r = client.post(f"/api/progress/{pid}/topic_01", json={"status": "complete"})
    assert r.status_code == 200
    reward = r.json()["reward"]
    assert reward["xp_gain"] == 80 and reward["gems_gain"] == 10

    # Re-completing an already-complete topic should not grant XP again.
    r2 = client.post(f"/api/progress/{pid}/topic_01", json={"status": "complete"})
    assert r2.json()["reward"] is None

    summary = client.get(f"/api/gamification/{pid}").json()
    assert summary["xp"] == 80 and summary["gems"] == 10
    first_steps = next(b for b in summary["badges"] if b["id"] == "first_steps")
    assert first_steps["got"] is True


def test_quest_claim_lifecycle(client):
    pid = client.post("/api/profiles", json={"name": "Quester"}).json()["id"]

    # Can't claim before the quest is done.
    r = client.post(f"/api/gamification/{pid}/quests/q1/claim")
    assert r.status_code == 400

    client.post(f"/api/progress/{pid}/topic_02", json={"status": "complete"})

    r2 = client.post(f"/api/gamification/{pid}/quests/q1/claim")
    assert r2.status_code == 200
    assert r2.json()["xp_gain"] == 50

    # Claiming twice should fail.
    r3 = client.post(f"/api/gamification/{pid}/quests/q1/claim")
    assert r3.status_code == 400


def test_loot_purchase_requires_enough_gems(client):
    pid = client.post("/api/profiles", json={"name": "Shopper"}).json()["id"]
    r = client.post(f"/api/gamification/{pid}/loot/streak_freeze/purchase")
    assert r.status_code == 400

    r2 = client.post(f"/api/gamification/{pid}/loot/unknown_item/purchase")
    assert r2.status_code == 404


# ── Flashcards ────────────────────────────────────────────────────
def test_flashcards_lifecycle(client):
    pid = client.post("/api/profiles", json={"name": "Carder"}).json()["id"]
    r = client.get(f"/api/flashcards/{pid}/topic_01")
    assert r.status_code == 200 and r.json() == {}

    r2 = client.post(f"/api/flashcards/{pid}/topic_01/f1", json={"status": "know"})
    assert r2.status_code == 200

    r3 = client.get(f"/api/flashcards/{pid}/topic_01")
    assert r3.json() == {"f1": "know"}

    r4 = client.post(f"/api/flashcards/{pid}/topic_01/f1", json={"status": "bogus"})
    assert r4.status_code == 400


# ── SPA serving ───────────────────────────────────────────────────
def test_spa_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_spa_unknown_route(client):
    r = client.get("/some/unknown/path")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
