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


# ── SPA serving ───────────────────────────────────────────────────
def test_spa_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_spa_unknown_route(client):
    r = client.get("/some/unknown/path")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
