# AI/LLM Learning App — Implementation Plan

> **INSTRUCTIONS FOR THE IMPLEMENTING SESSION (read first)**
> This plan is approved and final. Build the complete app in this folder (`Ai-Learning/`) following every section below — no need to re-ask the user about scope, stack, or curriculum. Execute the Build Phases (Section 8) in order, creating the full project structure (Section 7) with all 64 topics (Section 4), each containing everything in Section 5. Work autonomously; only ask the user if something is truly blocking.
>
> **Confirmed decisions:**
> - App name: **llm-academy**
> - Stack: Python 3.10+ / FastAPI / SQLite / no-build-step SPA (Section 2)
> - UI: Apple-inspired — minimal, elegant, premium feel (Section 6)
> - Practicals: offline-first, optional API key for real LLM calls
> - Navigation: free navigation across all levels, with a suggested path highlighted
> - Git: user will provide the GitHub repo URL; use a placeholder `https://github.com/<USER>/llm-academy` in README/pyproject until given, and prepare the folder git-ready (init, .gitignore, sensible commits) so the user can push
> - First-run flow: ask user's name → create profile → dashboard
>
> **Definition of done:** fresh `pip install` from the repo works; `llm-academy` command launches the server and opens the browser; all 64 topics have kid-friendly + detailed theory, runnable practical, 5–8 MCQs, flashcards, and 2–3 embedded YouTube videos; dashboard tracks progress, completion marks, and save-for-later; tests pass.

## 1. Overview
A local web app that teaches AI/LLM concepts end-to-end, from "what is AI?" to building RAG pipelines and evaluating them. Users install from GitHub with one command, run one command, and the app opens in their browser. Explanations are simple enough for an 8-year-old, with real runnable code for adults.

**Name (proposal):** `llm-academy` (open to change)

## 2. Tech Stack
| Layer | Choice |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Frontend | Single-page app served by FastAPI (HTML/CSS/JS, no build step — keeps install simple) |
| Storage | SQLite (local, per-user profiles + progress) |
| LLM examples | Offline-first (simulated/tiny local logic); optional API key (Anthropic/OpenAI) unlocks real calls |
| Distribution | GitHub repo, `pip install git+https://github.com/<you>/llm-academy.git` → run `llm-academy` |

## 3. Install & Run Experience
```bash
pip install git+https://github.com/<user>/llm-academy.git
llm-academy          # starts server, auto-opens browser at localhost:8000
```
First launch: welcome screen asks the user's name → creates profile → lands on dashboard. Multiple profiles supported (family/team can share one machine).

## 4. Curriculum (Levels → Topics)
Designed as a teacher would sequence it: concepts → conversation → retrieval → agents → quality → safety → customization → production → job readiness. Each level builds only on what came before. ~60 topics; micro-concepts (BPE, RoPE, KV cache) are taught inside parent topics so beginners never feel lost.

**Level 1 — Foundations 🌱** *(What is this magic?)*
1. What is AI? (robot friend analogy)
2. What is Machine Learning? (learning from examples, like riding a bike)
3. Neural Networks (brain made of tiny switches)
4. What is a Language Model? (super-powered autocomplete)
5. LLMs vs. SLMs — big brains vs. small brains (why small can be smart)

**Level 2 — How LLMs Work 🧠** *(Look inside the engine)*
6. Tokens & Tokenization (LEGO bricks; includes BPE)
7. Embeddings (turning words into map coordinates)
8. Transformers & Attention (self-attention, multi-head — paying attention to the right words)
9. Inside the engine: FFN, RoPE, KV Cache, MoE (a friendly peek under the hood)
10. How LLMs are Trained (pretraining → RLHF & DPO — "reading the library, then learning manners")
11. Context Windows & Memory (how much the model can hold in its head)
12. Temperature & Sampling (careful vs. creative answers)
13. Inference & Latency (asking a question vs. waiting for the answer)

**Level 3 — Talking to LLMs 💬** *(The art of asking)*
14. Prompt Engineering basics
15. Zero-shot, One-shot, Few-shot (showing examples first)
16. Chain of Thought & Reasoning (think step by step)
17. Prompt Chaining (breaking big jobs into small prompts)
18. System Prompts & Roles
19. Context Engineering (giving the model the right stuff to read)
20. Structured Outputs (JSON, tables — getting tidy answers)

**Level 4 — Using LLMs: APIs & the Tool Landscape 🧰** *(Meet the players)*
21. Using LLM APIs (keys, messages, parameters)
22. Function/Tool Calling (letting the model use a calculator)
23. The big players: Claude, ChatGPT, Gemini, Grok, Llama (who makes what)
24. Coding assistants: Claude Code, Codex, Copilot, Cursor
25. Enterprise platforms: Azure AI, AWS Bedrock, Google Vertex, Databricks

**Level 5 — Embeddings, Vector Search & RAG 🔍** *(Open-book exams)*
26. Embedding Models & Similarity Metrics (cosine similarity made simple)
27. Vector Databases (a library organized by meaning)
28. Chunking Strategies (cutting books into readable pieces)
29. RAG end-to-end (the open-book exam) — hands-on build
30. Advanced retrieval: Hybrid Search, Metadata Filtering, Reranking, Query Rewriting
31. RAG Failures & Architecture patterns (what goes wrong and how pros fix it)
32. Agentic RAG (RAG that thinks before it searches)

**Level 6 — Agents & Multi-Agent Systems 🤖** *(Assistants that do things)*
33. What is an AI Agent? (perceive → decide → act)
34. ReAct & Planning (reason + act loops)
35. Agent Memory (remembering past conversations)
36. Multi-Agent Systems (agents working as a team)
37. Agentic Workflows (plan, act, observe, iterate)
38. Frameworks: LangChain, LangGraph, AutoGen, CrewAI (when to use which)
39. MCP — Model Context Protocol (a universal plug for tools)
40. Agent-to-Agent protocols (A2A — how agents talk to each other)

**Level 7 — Evaluation & Testing 📏** *(Is it actually good?)*
41. Hallucinations (when AI confidently makes things up)
42. Eval methods: exact match, LLM-as-judge, rubrics
43. Benchmarks & Metrics (report cards for models)
44. RAGAS & evaluating RAG pipelines
45. Testing methodologies for AI apps (unit, integration, regression, red-teaming — why AI testing ≠ normal testing)
46. Build your own eval suite — hands-on

**Level 8 — Safety & Security 🛡️** *(Keeping AI on a leash)*
47. Guardrails (input & output safety checks)
48. Prompt Injection (tricking the AI — and defending against it)
49. OWASP LLM Top 10 (the official danger list)
50. Responsible AI: bias, privacy, ethics

**Level 9 — Fine-tuning, SLMs & Local Models 🔧** *(Making your own model)*
51. Fine-tuning vs. RAG vs. Prompting (when to use what)
52. LoRA & QLoRA (teaching new tricks cheaply)
53. Fine-tuning a Small Language Model — hands-on walkthrough
54. Quantization (shrinking models to fit your laptop)
55. Running models locally: Ollama, vLLM & PagedAttention

**Level 10 — Production & Cloud ☁️** *(Ship it!)*
56. Productionizing AI apps (from notebook to real product: APIs, Docker, CI/CD)
57. Cloud services for AI apps (compute, storage, serverless, managed model endpoints)
58. Observability & Monitoring (watching your AI in the wild)
59. Cost & Model Routing (right model for the right job, latency budgets)
60. Scaling & serving (vLLM, caching, batching)

**Level 11 — Capstone & Interview Prep 🎓** *(Get the job)*
61. Capstone project: build a production-style RAG + agent app end-to-end
62. Interview question bank (concept questions per level, with model answers)
63. System-design for AI apps (whiteboard-style: "design a chatbot over company docs")
64. Portfolio & resume guidance (what to showcase from this course)

## 5. Each Topic Contains
1. **Theory** — two tabs: "Explain like I'm 8" (analogies, pictures/emoji) and "Full detail" (proper depth with examples).
2. **Practical** — runnable code example executed from the UI (e.g., Topic 15 = mini RAG: chunk sample docs → embed with a tiny offline embedder → retrieve → answer; Topic 23 = run a real mini eval and see scores). Offline by default; real-API mode if key is set.
3. **MCQ Quiz** — 5–8 questions, instant feedback, explanation for each answer, score saved.
4. **Flashcards** — flip-style cards, "know it / review again" spaced-repetition-lite.
5. **Videos** — 2–3 curated YouTube videos embedded per topic (e.g., 3Blue1Brown, Andrej Karpathy, freeCodeCamp).
6. **Actions** — Mark Complete ✅ / Save for Later 🔖.

## 6. Dashboard
- Overall progress ring (% complete) + per-level progress bars
- Continue-where-you-left-off card
- Quiz average score, flashcard streak
- "Saved for later" list
- Apple-inspired design language: minimal, elegant, generous whitespace, SF-style typography, frosted-glass cards, subtle shadows, smooth spring animations, light/dark mode, responsive

## 7. Project Structure
```
llm-academy/
├── pyproject.toml            # pip-installable, console entry point
├── README.md                 # install & usage
├── llm_academy/
│   ├── main.py               # FastAPI app + launcher
│   ├── db.py                 # SQLite: profiles, progress, scores
│   ├── api/                  # routes: profile, progress, quiz, run-example
│   ├── content/              # topics as JSON/MD (theory, MCQs, cards, video IDs)
│   ├── examples/             # runnable practicals (rag_demo.py, eval_demo.py, ...)
│   └── static/               # SPA: index.html, app.js, styles.css
└── tests/
```

## 8. Build Phases
1. **Skeleton** — FastAPI app, SQLite, profile creation flow, pip-installable entry point.
2. **Dashboard + navigation** — levels, topic pages, progress tracking, mark complete/later.
3. **Content** — all 64 topics: theory (both depths), MCQs, flashcards, curated video links. Built level by level (1→11) so the app is usable early.
4. **Practicals** — runnable examples (tokenizer demo, temperature playground, RAG, agent, eval suite) with offline + API modes.
5. **Polish** — dark mode, animations, responsive design, empty states.
6. **Ship** — README, tests, verify fresh `pip install` from GitHub works end-to-end.

## 9. Status
All questions resolved — see "Instructions for the implementing session" at the top. Only pending item: the GitHub repo URL (user will supply; placeholder until then).
