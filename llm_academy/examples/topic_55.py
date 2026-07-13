"""
Topic 55: Running Models Locally — Ollama, vLLM & PagedAttention
==================================================================
Demonstrates the concepts behind local LLM inference: quantization,
Ollama's REST API, and vLLM's PagedAttention.

This script FIRST tries a real HTTP connection to a local Ollama server
at http://localhost:11434 (short timeout, stdlib urllib only). If a
server is found it lists real installed models and streams a real chat
response. If not — the overwhelmingly common case — it falls back to a
clearly-labeled SIMULATED walkthrough using the same request/response
shapes Ollama actually returns, so the lesson works fully offline.
"""

from __future__ import annotations

import json
import textwrap
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
THIN = "-" * 70
WRAP_W = 66

OLLAMA_HOST = "http://localhost:11434"
NETWORK_TIMEOUT_S = 2.5  # localhost should answer almost instantly, or not at all


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=WRAP_W, initial_indent=indent,
                          subsequent_indent=indent)


def print_header(title: str) -> None:
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def print_section(title: str) -> None:
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)


# ---------------------------------------------------------------------------
# CONCEPTS: Ollama, quantization, vLLM, PagedAttention
# ---------------------------------------------------------------------------

def demo_explain_concepts() -> None:
    print_section("CONCEPT 1 — Ollama: a local LLM runtime")
    print(wrap(
        "Ollama is a tool that packages a model-inference engine (llama.cpp) "
        "together with a model library, a CLI, and a REST API. Running "
        "`ollama run llama3.2` downloads a model once and starts serving it "
        "from your own machine — no account, no API key, no internet needed "
        "after the download. It listens on http://localhost:11434 and exposes "
        "endpoints like /api/tags (list models) and /api/chat (send messages)."
    ))

    print_section("CONCEPT 2 — Quantization: why a '7B' model fits on a laptop")
    print(wrap(
        "A model's size in memory is (number of parameters) x (bytes per "
        "parameter). A 7-billion-parameter model stored at full precision "
        "(FP16, 2 bytes/param) needs roughly 7B x 2 bytes = 14 GB just for "
        "weights — before any working memory for activations or the KV cache. "
        "Quantization reduces the bytes used per weight, e.g. down to 4 bits "
        "(0.5 bytes/param):"
    ))
    print()
    rows = [
        ("FP16 (16-bit)", "2.0 bytes", "7B x 2.0B  =  ~14.0 GB"),
        ("INT8 (8-bit)", "1.0 bytes", "7B x 1.0B  =  ~7.0 GB"),
        ("Q4 (4-bit, GGUF)", "0.5 bytes", "7B x 0.5B  =  ~3.5 GB"),
    ]
    print(f"  {'Precision':<18} {'Bytes/param':<14} Approx. size for a 7B model")
    print(f"  {'-'*16:<18} {'-'*12:<14} {'-'*32}")
    for precision, bytes_per, size in rows:
        print(f"  {precision:<18} {bytes_per:<14} {size}")
    print()
    print(wrap(
        "This is why Ollama and llama.cpp default to 4-bit GGUF files: a 7B "
        "model shrinks from 14 GB to roughly 3.5-4 GB, fitting comfortably in "
        "8-16 GB of RAM. The tradeoff is a small, usually modest quality loss "
        "versus the full-precision original — the rounded weights carry "
        "slightly less information, but for most everyday tasks the "
        "difference is hard to notice."
    ))

    print_section("CONCEPT 3 — vLLM: a production serving engine")
    print(wrap(
        "Ollama is built for one person on one machine. vLLM solves a "
        "different problem: serving one model to MANY concurrent users with "
        "high throughput, the way a company would run its own hosted API. "
        "vLLM batches many in-flight requests onto a GPU simultaneously and "
        "uses 'continuous batching' — new requests join a running batch as "
        "soon as a slot frees up, instead of waiting for the whole batch to "
        "finish. This keeps the GPU busy and can serve 2-24x more tokens per "
        "second than a naive request-at-a-time server."
    ))

    print_section("CONCEPT 4 — PagedAttention: paging for the KV cache")
    print(wrap(
        "Every generated token requires the model to keep a growing 'KV "
        "cache' (key/value tensors) for that conversation. Naive servers "
        "pre-allocate the MAXIMUM possible cache size for every request up "
        "front — if a request ends early, that reserved memory is wasted and "
        "sits idle. PagedAttention (vLLM's key idea, Kwon et al. 2023) borrows "
        "a trick from operating systems: virtual memory paging. The KV cache "
        "is split into small fixed-size 'blocks' (pages) that are allocated "
        "on demand as a sequence grows, and freed the instant a sequence "
        "finishes — just like an OS pages physical memory in and out for "
        "running processes. No more pre-allocated, wasted memory means many "
        "more sequences can be kept in flight at once, which is where the "
        "2-24x throughput gains come from."
    ))


# ---------------------------------------------------------------------------
# STEP 1 — List local models (real attempt, simulated fallback)
# ---------------------------------------------------------------------------

SIMULATED_TAGS_RESPONSE = {
    "models": [
        {
            "name": "llama3.2:3b",
            "model": "llama3.2:3b",
            "modified_at": "2026-06-02T10:14:00Z",
            "size": 2019393792,
            "digest": "a80c4f23b8e3",
            "details": {
                "family": "llama",
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "phi3:mini",
            "model": "phi3:mini",
            "modified_at": "2026-05-20T08:03:00Z",
            "size": 2176481280,
            "digest": "d1e9a7c40f2b",
            "details": {
                "family": "phi3",
                "parameter_size": "3.8B",
                "quantization_level": "Q4_0",
            },
        },
    ]
}


def fetch_local_models():
    """
    Try a real GET http://localhost:11434/api/tags.
    Returns (ok, models, raw_json). Never raises — any failure (connection
    refused, timeout, DNS, etc.) is caught and reported as ok=False.
    """
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_S) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "?") for m in raw.get("models", [])]
            return True, models, raw
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False, [], None


def demo_list_models():
    print_section("STEP 1 — GET /api/tags  (list locally installed models)")
    print(wrap(f"Attempting real HTTP call to {OLLAMA_HOST}/api/tags "
               f"(timeout={NETWORK_TIMEOUT_S}s)..."))
    print()

    ok, models, raw = fetch_local_models()

    if ok:
        print("  [LIVE] Connected to a real local Ollama server!")
        if models:
            print(f"  Installed models found: {len(models)}")
            for name in models:
                print(f"    - {name}")
        else:
            print("  No models are installed yet (server is running, but empty).")
        return models[0] if models else None

    print("  No local Ollama server detected at localhost:11434 —")
    print("  showing a SIMULATED example of what this would return.")
    print()
    print("  Simulated response body (matches Ollama's real JSON shape):")
    print(THIN)
    pretty = json.dumps(SIMULATED_TAGS_RESPONSE, indent=2)
    for line in pretty.splitlines():
        print(f"  {line}")
    print(THIN)

    sim_models = [m["name"] for m in SIMULATED_TAGS_RESPONSE["models"]]
    print()
    print(f"  Simulated models available: {', '.join(sim_models)}")
    return sim_models[0]


# ---------------------------------------------------------------------------
# STEP 2 — Chat completion (real attempt, simulated fallback + streaming)
# ---------------------------------------------------------------------------

SIMULATED_STREAM_CHUNKS = [
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": "Quant"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": "ization"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " shrinks"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " model"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " weights"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " so they"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " fit on"}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": " a laptop."}, "done": False},
    {"model": "llama3.2:3b", "message": {"role": "assistant", "content": ""},
     "done": True, "total_duration": 812345678, "eval_count": 9},
]


def call_ollama_chat(model: str, prompt: str):
    """
    Try a real streaming POST http://localhost:11434/api/chat.
    Ollama streams newline-delimited JSON objects by default: each line is
    one JSON object with a partial 'message.content' fragment and a 'done'
    boolean; the last line has done=True.
    Returns (ok, assembled_text). Never raises.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        fragments = []
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_S) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                frag = chunk.get("message", {}).get("content", "")
                if frag:
                    fragments.append(frag)
                    print(frag, end="", flush=True)
                if chunk.get("done"):
                    break
        print()
        return True, "".join(fragments)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False, None


def demo_chat_completion(model_hint):
    print_section("STEP 2 — POST /api/chat  (send a prompt, stream the reply)")
    prompt = "In one short sentence, what is quantization?"
    print(wrap(f'Prompt: "{prompt}"'))
    print()

    if model_hint:
        print(f"  Attempting real streaming call using model '{model_hint}'...")
        ok, text = call_ollama_chat(model_hint, prompt)
        if ok:
            print()
            print("  [LIVE] Assembled full response from the real server above.")
            return

    print("  No local Ollama server responded — showing a SIMULATED")
    print("  streamed chat completion instead.")
    print()
    print("  Ollama streams the response as newline-delimited JSON (NDJSON):")
    print("  each line is a full JSON object; concatenate 'message.content'")
    print("  fragments across lines until a line has \"done\": true.")
    print()
    print("  Simulated NDJSON lines as they would arrive over the wire:")
    print(THIN)
    for chunk in SIMULATED_STREAM_CHUNKS[:4]:
        print(f"  {json.dumps(chunk)}")
    print("  ... (more lines) ...")
    print(f"  {json.dumps(SIMULATED_STREAM_CHUNKS[-1])}")
    print(THIN)

    print()
    print("  Assembling the fragments as a client library would:")
    assembled = ""
    for chunk in SIMULATED_STREAM_CHUNKS:
        frag = chunk["message"]["content"]
        assembled += frag
        if frag:
            print(f"    + {frag!r:<14} -> so far: {assembled!r}")
    print()
    print(f"  Final assembled text: {assembled!r}")


# ---------------------------------------------------------------------------
# STEP 3 — Comparison: Ollama vs vLLM vs hosted API
# ---------------------------------------------------------------------------

def demo_comparison():
    print_section("STEP 3 — Ollama vs vLLM vs a hosted API")

    rows = [
        ("Setup complexity", "One command install", "Python/Docker + GPU driver setup", "Just an API key"),
        ("Hardware needed", "Laptop CPU/GPU, 8GB+ RAM", "Datacenter GPU (A10/A100, etc.)", "None — provider's problem"),
        ("Concurrency model", "Sequential, ~1 request", "Continuous batching, many at once", "Provider scales for you"),
        ("Throughput", "~5-20 tok/s", "~200-2000+ tok/s", "Varies, provider-managed"),
        ("Quantization", "GGUF (auto, 4-8 bit)", "AWQ / GPTQ / bitsandbytes", "Not your concern"),
        ("Cost model", "Free after hardware", "Your GPU/cloud bill", "Pay per token"),
        ("Latency (localhost)", "Very low, no network hop", "Low, but network hop to server", "Network + provider queue"),
        ("Privacy", "Data never leaves machine", "Data stays on your infra", "Data sent to third party"),
        ("Best for", "Prototyping, offline, privacy", "Multi-user production services", "Zero-ops, elastic scale"),
    ]

    c0, c1, c2, c3 = 20, 26, 30, 24
    print(f"  {'Dimension':<{c0}} {'Ollama':<{c1}} {'vLLM':<{c2}} {'Hosted API':<{c3}}")
    print("  " + "-" * (c0 + c1 + c2 + c3))
    for dim, ollama, vllm, hosted in rows:
        print(f"  {dim:<{c0}} {ollama:<{c1}} {vllm:<{c2}} {hosted:<{c3}}")

    print()
    print(wrap(
        "Guidance: reach for Ollama when prototyping locally, working "
        "offline, or handling sensitive data on a single machine. Reach for "
        "vLLM when you own the model and need to serve many concurrent users "
        "efficiently on your own GPU hardware. Reach for a hosted API when "
        "you want zero infrastructure to manage, need to scale unpredictably, "
        "or don't want to own any GPUs at all — accepting per-token cost and "
        "sending data to a third party in exchange."
    ))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print_header("TOPIC 55: RUNNING MODELS LOCALLY — OLLAMA, VLLM & PAGEDATTENTION")
    print()
    print(wrap(
        "This demo explains local LLM inference concepts, then attempts a "
        "REAL connection to a local Ollama server before falling back to a "
        "clearly-labeled simulation. Everything below runs with zero errors "
        "whether or not Ollama is installed."
    ))

    demo_explain_concepts()
    model_hint = demo_list_models()
    demo_chat_completion(model_hint)
    demo_comparison()

    print()
    print(DIVIDER)
    print("  KEY TAKEAWAYS")
    print(DIVIDER)
    takeaways = [
        "1. Ollama is the easy-button for local inference: one command, an",
        "   OpenAI-compatible API at localhost:11434, no data leaves your machine.",
        "2. Quantization trades a few bits of precision per weight for a huge",
        "   memory win — a 7B model shrinks from ~14GB (FP16) to ~3.5GB (4-bit).",
        "3. vLLM targets production, multi-user serving with continuous",
        "   batching, not single-user prototyping like Ollama.",
        "4. PagedAttention pages the KV cache in fixed-size blocks (like OS",
        "   virtual memory) instead of pre-allocating worst-case memory per",
        "   request — this is the source of vLLM's 2-24x throughput gains.",
        "5. Ollama's /api/chat streams NDJSON lines; concatenate 'content'",
        "   fragments until a line reports done=true.",
        "6. Choose local (Ollama/vLLM) for privacy, cost control, and offline",
        "   use; choose a hosted API for zero-ops elastic scale.",
    ]
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
