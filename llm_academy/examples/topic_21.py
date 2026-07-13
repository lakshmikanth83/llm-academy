"""
Topic 21: Using LLM APIs
========================
Walkthrough of making a Claude API call: the Messages API request shape,
temperature/max_tokens, a real (or simulated) completion, and streaming.
Runs fully offline by default. If ANTHROPIC_API_KEY is set in the
environment, it will attempt one small real API call and fall back to a
simulated response if that call fails for any reason.
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIVIDER = "=" * 70
THIN = "-" * 70

API_URL = "https://api.anthropic.com/v1/messages"
MODEL_ID = "claude-haiku-4-5-20251001"
REQUEST_TIMEOUT_SECONDS = 8


def wrap(text: str, width: int = 66, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                          subsequent_indent=indent)


def section(title: str) -> None:
    print()
    print(DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def subsection(title: str) -> None:
    print()
    print(THIN)
    print(f"  {title}")
    print(THIN)


# ---------------------------------------------------------------------------
# 1. THE MESSAGES API REQUEST SHAPE
# ---------------------------------------------------------------------------

def build_example_request(prompt: str, temperature: float, max_tokens: int) -> dict:
    """
    Construct the exact JSON body that would be POSTed to
    https://api.anthropic.com/v1/messages for a simple, single-turn request.
    """
    return {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }


def demo_request_shape() -> None:
    subsection("1. The Messages API request shape")
    print(
        "  Every Claude API call is a single HTTP POST to one endpoint:\n"
        "    POST https://api.anthropic.com/v1/messages\n"
        "\n  The request body always has the same core fields:\n"
        "    model       - which Claude model answers this request\n"
        "    messages    - a list of {role, content} turns (user/assistant)\n"
        "    max_tokens  - hard cap on how long the reply may be\n"
        "    temperature - how deterministic vs. creative the reply is\n"
    )

    example = build_example_request(
        prompt="Explain what an API is in one sentence.",
        temperature=0.7,
        max_tokens=100,
    )

    print("  Exact JSON body that would be sent:")
    for line in json.dumps(example, indent=2).splitlines():
        print(f"    {line}")

    print(
        "\n  Required HTTP headers alongside this body:\n"
        "    x-api-key: <your ANTHROPIC_API_KEY>\n"
        "    anthropic-version: 2023-06-01\n"
        "    content-type: application/json"
    )


# ---------------------------------------------------------------------------
# 2. TEMPERATURE AND MAX_TOKENS
# ---------------------------------------------------------------------------

def demo_temperature_and_max_tokens() -> None:
    subsection("2. Temperature and max_tokens")
    print(
        "  temperature (0.0 - 1.0):\n"
        "    0.0  -> deterministic and focused. The model almost always\n"
        "           picks the single highest-probability next token.\n"
        "    1.0  -> more random and creative. The model samples more\n"
        "           broadly across plausible next tokens.\n"
        "\n  max_tokens:\n"
        "    A hard cap on how many tokens the RESPONSE may contain.\n"
        "    It does not affect the size of the input prompt. Too low and\n"
        "    the reply may be cut off mid-sentence; too high and a runaway\n"
        "    generation could cost more than expected."
    )

    prompt = "Describe a rainy city street in one short sentence."
    print(f"\n  Same prompt, two temperature settings: \"{prompt}\"")

    print("\n  [ILLUSTRATIVE / SIMULATED — not a real model call]")
    print("  temperature=0.0 (focused, likely near-identical every run):")
    print(wrap(
        "\"Rain streaked the grey pavement as commuters hurried past "
        "under a row of dim streetlights.\"",
        indent="    ",
    ))

    print("\n  temperature=1.0 (creative, varies more run to run):")
    print(wrap(
        "\"Neon puddles shivered on the cobblestones while a lone "
        "saxophone note dissolved into the drizzle.\"",
        indent="    ",
    ))

    print(
        "\n  Notice the low-temperature version reads like the 'safe, "
        "expected' answer,\n"
        "  while the high-temperature version takes more stylistic risk. "
        "Neither is\n"
        "  'more correct' — the right setting depends on the task "
        "(data extraction\n"
        "  wants low temperature; brainstorming wants higher)."
    )


# ---------------------------------------------------------------------------
# 3. MAKING THE CALL — REAL IF POSSIBLE, SIMULATED OTHERWISE
# ---------------------------------------------------------------------------

SIMULATED_COMPLETION = "Hello there, friend! Wishing you well."

SIMULATED_RESPONSE_ENVELOPE = {
    "id": "msg_01SimulatedExample123",
    "type": "message",
    "role": "assistant",
    "model": MODEL_ID,
    "content": [{"type": "text", "text": SIMULATED_COMPLETION}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 8},
}


def call_claude_api(api_key: str, prompt: str, max_tokens: int = 50) -> dict | None:
    """
    Attempt one small, real request to the Claude Messages API using only
    the urllib stdlib module (no third-party HTTP or Anthropic packages).
    Returns the parsed JSON response on success, or None on any failure —
    this function must never raise.
    """
    body = json.dumps({
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError, OSError):
        # Network failure, invalid key, non-JSON body, timeout, etc.
        # Fall back to the simulated path — never let this crash the demo.
        return None
    except Exception:
        # Belt-and-suspenders: absolutely no exception should escape here.
        return None


def demo_real_or_simulated_call() -> None:
    subsection("3. Making the call — real API if a key is available")

    prompt = "Say hello in exactly 5 words."
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    print(f"  Prompt: \"{prompt}\"")
    print(f"  Model:  {MODEL_ID}")
    print(f"  max_tokens: 50")

    response_json = None
    if api_key:
        print("\n  ANTHROPIC_API_KEY detected — attempting a real API call...")
        response_json = call_claude_api(api_key, prompt, max_tokens=50)
    else:
        print("\n  No ANTHROPIC_API_KEY found in the environment.")

    if response_json is not None:
        try:
            text_blocks = response_json.get("content", [])
            reply_text = "".join(
                block.get("text", "") for block in text_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
            usage = response_json.get("usage", {})
            print("\n  REAL API RESPONSE:")
            print(f"    \"{reply_text}\"")
            print(
                f"    (input_tokens={usage.get('input_tokens', '?')}, "
                f"output_tokens={usage.get('output_tokens', '?')})"
            )
            return
        except Exception:
            # Malformed response shape — fall through to simulated output.
            pass

    print(
        "\n  SIMULATED RESPONSE (no API key detected — showing what a real"
        "\n  call would return):"
    )
    print(f"    \"{SIMULATED_COMPLETION}\"")
    usage = SIMULATED_RESPONSE_ENVELOPE["usage"]
    print(
        f"    (input_tokens={usage['input_tokens']}, "
        f"output_tokens={usage['output_tokens']})"
    )
    print(
        "\n  Full example response envelope (this is the shape you get back"
        "\n  from a real call, whether or not this run made one):"
    )
    for line in json.dumps(SIMULATED_RESPONSE_ENVELOPE, indent=2).splitlines():
        print(f"    {line}")


# ---------------------------------------------------------------------------
# 4. STREAMING RESPONSES
# ---------------------------------------------------------------------------

SIMULATED_STREAM_DELTAS = [
    "Hello", " there", ",", " friend", "!", " Wishing", " you", " well", ".",
]


def demo_streaming() -> None:
    subsection("4. Streaming responses (conceptual)")
    print(
        "  Setting \"stream\": true in the request body changes the response\n"
        "  from a single JSON object into a sequence of Server-Sent Events\n"
        "  (SSE). Instead of waiting for the entire reply, the client reads\n"
        "  small pieces (deltas) as they are generated:\n"
        "\n"
        "    event: message_start        -> reply is beginning\n"
        "    event: content_block_start  -> a text block has opened\n"
        "    event: content_block_delta  -> one chunk of text     (repeats)\n"
        "    event: content_block_stop   -> the text block is done\n"
        "    event: message_delta        -> final stop_reason + usage\n"
        "    event: message_stop         -> the whole message is complete\n"
        "\n  Each content_block_delta event carries a small 'text_delta'\n"
        "  fragment. The client concatenates these fragments in order to\n"
        "  reconstruct the full response as it arrives, which is what makes\n"
        "  chat UIs feel like the model is 'typing' in real time."
    )

    print("\n  [ILLUSTRATIVE / SIMULATED] Assembling text from deltas:")
    assembled = ""
    pieces_preview = []
    for delta in SIMULATED_STREAM_DELTAS:
        assembled += delta
        pieces_preview.append(repr(delta))
    print(f"    deltas received : {' '.join(pieces_preview)}")
    print(f"    assembled text  : \"{assembled}\"")
    print(
        "\n  Time-to-first-token drops dramatically with streaming — the\n"
        "  user sees the first word almost immediately instead of waiting\n"
        "  for the entire response to finish generating."
    )


# ---------------------------------------------------------------------------
# 5. GETTING YOUR OWN KEY AND BEST PRACTICES
# ---------------------------------------------------------------------------

def demo_getting_a_key() -> None:
    subsection("5. How to get your own API key")
    print(
        "  1. Go to https://console.anthropic.com and sign in / sign up.\n"
        "  2. Navigate to \"API Keys\" in the console.\n"
        "  3. Click \"Create Key\", name it, and copy the value immediately\n"
        "     (it is only shown once).\n"
        "  4. Set it as an environment variable rather than pasting it into\n"
        "     code:\n"
        "       export ANTHROPIC_API_KEY=\"sk-ant-...\"\n"
    )

    print("  Best practices:")
    practices = [
        "Never hardcode a key as a string literal in source code.",
        "Never commit a key to version control (add .env to .gitignore).",
        "Use environment variables locally and a secrets manager in prod.",
        "Set spending limits / usage alerts in the console to avoid",
        "  surprise bills from a runaway loop or leaked key.",
        "Rotate a key immediately if you suspect it has been exposed.",
    ]
    for p in practices:
        print(f"    - {p}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(DIVIDER)
    print("  TOPIC 21 — USING LLM APIs")
    print(DIVIDER)
    print(wrap(
        "This demo walks through the shape of a real Claude API request, "
        "what temperature and max_tokens control, how a completion (real "
        "or simulated) comes back, and how streaming changes the response "
        "format. It runs fully offline by default. If ANTHROPIC_API_KEY "
        "is set in the environment, it will attempt one small real API "
        "call using only the Python standard library (urllib) and fall "
        "back to a simulated response if that call fails for any reason.",
        indent="  ",
    ))

    demo_request_shape()
    demo_temperature_and_max_tokens()
    demo_real_or_simulated_call()
    demo_streaming()
    demo_getting_a_key()

    section("KEY TAKEAWAYS")
    takeaways = [
        "1. Every Claude API call is one HTTP POST with model, messages,",
        "   max_tokens, and (optionally) temperature in the JSON body.",
        "2. temperature=0 is deterministic/focused; temperature closer to 1",
        "   is more creative and varied — pick based on the task.",
        "3. max_tokens caps the RESPONSE length only, not the input size;",
        "   too low truncates the reply, too high risks runaway cost.",
        "4. Streaming (stream: true) turns one JSON response into a series",
        "   of SSE events; concatenating text_delta fragments in order",
        "   reconstructs the full reply as it is generated.",
        "5. Get a key at console.anthropic.com and load it from an",
        "   environment variable — never hardcode or commit it.",
        "6. This script works with or without a real key: no key (or any",
        "   failure) triggers a clearly-labeled simulated fallback.",
    ]
    print()
    for t in takeaways:
        print(f"  {t}")
    print()


if __name__ == "__main__":
    main()
