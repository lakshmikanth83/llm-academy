# llm-academy

An interactive, local web app that teaches AI/LLM concepts end-to-end — from "what is AI?" to building and evaluating production RAG pipelines. Explanations are simple enough for an 8-year-old ("Explain like I'm 8"), with real, runnable code for adults.

No cloud account, no signup, no build step. Install it, run one command, and it opens in your browser.

## Features

- **64 topics** across 11 levels, from Foundations to Capstone & Interview Prep
- **Two-depth theory** for every topic — "Explain like I'm 8" and full technical detail
- **Runnable practicals** — offline-first Python examples you execute right from the browser (26 topics ship with hands-on code; a couple optionally use a real Claude/Ollama call if you provide an API key or run a local server, and gracefully fall back to a simulated walkthrough otherwise)
- **Quizzes** — 6-9 MCQs per topic with instant feedback and explanations
- **Flashcards** — flip-style review cards per topic
- **Curated videos** — 2 hand-picked videos per topic
- **Progress dashboard** — completion tracking, quiz averages, saved-for-later list, per-level progress
- **Multiple profiles** — a family or team can share one machine
- **Apple-inspired UI** — minimal, elegant, light/dark mode, fully responsive

## Install & Run

```bash
pip install git+https://github.com/lakshmikanth83/llm-academy.git
llm-academy
```

This starts a local server at `http://localhost:8000` and opens it in your default browser automatically.

On first launch you'll be asked for a name, which creates a local profile. All progress, quiz scores, and flashcard state are stored locally in SQLite at `~/.llm_academy/data.db` — nothing leaves your machine unless you opt in to a real LLM API call on a specific practical.

### Requirements

- Python 3.10+

### Running from source

```bash
git clone https://github.com/lakshmikanth83/llm-academy.git
cd llm-academy
pip install -e .
llm-academy
```

### Running tests

```bash
pip install -e . pytest
pytest tests/
```

## How it's built

- **Backend:** FastAPI + Uvicorn, SQLite (via `aiosqlite`) for local profile/progress storage
- **Frontend:** a single-page app in plain HTML/CSS/JS — no build step, no framework, no bundler
- **Content:** each topic is a JSON file (`llm_academy/content/topic_XX.json`) with theory, quiz questions, flashcards, and video links
- **Practicals:** each runnable example is a self-contained, offline-first Python script (`llm_academy/examples/topic_XX.py`) executed server-side and streamed back to the browser as plain text output

## Curriculum

1. Foundations — what is AI, ML, neural networks, language models
2. How LLMs Work — tokens, embeddings, transformers, training, context windows, sampling
3. Talking to LLMs — prompt engineering, few-shot, chain-of-thought, structured outputs
4. Using LLMs: APIs & the Tool Landscape — API basics, tool calling, the major model/tooling providers
5. Embeddings, Vector Search & RAG — similarity metrics, vector DBs, chunking, RAG, reranking
6. Agents & Multi-Agent Systems — ReAct, agent memory, multi-agent workflows, MCP, A2A
7. Evaluation & Testing — hallucinations, eval methods, benchmarks, RAGAS, building an eval suite
8. Safety & Security — guardrails, prompt injection, OWASP LLM Top 10, responsible AI
9. Fine-tuning, SLMs & Local Models — LoRA/QLoRA, quantization, running models locally
10. Production & Cloud — productionizing AI apps, observability, cost/routing, scaling
11. Capstone & Interview Prep — a full RAG+agent build, interview questions, system design, portfolio guidance

## License

MIT
