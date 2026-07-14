import pathlib
import threading
import time
import webbrowser
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llm_academy.db import init_db
from llm_academy.api.profile import router as profile_router
from llm_academy.api.progress import router as progress_router
from llm_academy.api.quiz import router as quiz_router
from llm_academy.api.content import router as content_router
from llm_academy.api.run_example import router as run_router
from llm_academy.api.flashcards import router as flashcards_router
from llm_academy.api.gamification import router as gamification_router

STATIC_DIR = pathlib.Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="LLM Academy", version="0.1.0", lifespan=lifespan)

app.include_router(profile_router, prefix="/api")
app.include_router(progress_router, prefix="/api")
app.include_router(quiz_router, prefix="/api")
app.include_router(content_router, prefix="/api")
app.include_router(run_router, prefix="/api")
app.include_router(flashcards_router, prefix="/api")
app.include_router(gamification_router, prefix="/api")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str = ""):
    return FileResponse(STATIC_DIR / "index.html")


def _open_browser():
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")


def run():
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("llm_academy.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
