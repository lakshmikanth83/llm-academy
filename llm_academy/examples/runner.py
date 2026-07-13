import importlib.util
import io
import os
import pathlib
import sys

EXAMPLES_DIR = pathlib.Path(__file__).parent


def run_topic_example(topic_id: str, api_key: str | None = None) -> dict:
    example_file = EXAMPLES_DIR / f"{topic_id}.py"

    if not example_file.exists():
        return {
            "output": f"No practical example available for {topic_id} yet.\nCheck back soon — more examples are being added!",
            "error": None,
        }

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    error = None

    try:
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        spec = importlib.util.spec_from_file_location("__main__", example_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    return {"output": output or "(no output)", "error": error}
