"""
llm/ollama_client.py
Thin wrapper around a local Ollama server for chat-style calls that return
strict JSON. Shared by every agent.
"""

import json
import re
import urllib.request
import urllib.error


class OllamaError(Exception):
    pass


def call_ollama(system_prompt: str, user_prompt: str, model: str = "llama3.1",
                 host: str = "http://localhost:11434", temperature: float = 0.2,
                 timeout: int = 300) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
        "format": "json",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {url}. Is `ollama serve` running and "
            f"is the model pulled (`ollama pull {model}`)? Original error: {e}"
        )
    except Exception as e:
        raise OllamaError(f"Unexpected error calling Ollama: {e}")

    if "message" not in body or "content" not in body["message"]:
        raise OllamaError(f"Unexpected Ollama response shape: {body}")
    return body["message"]["content"]


def extract_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise OllamaError("Model did not return parseable JSON:\n" + raw_text[:1500])


def run_agent(system_prompt: str, user_prompt: str, model: str, host: str,
               retries: int = 1) -> dict:
    """Calls the model and parses JSON, retrying once with a stricter
    reminder if parsing fails (cheap insurance against an 8B model's
    occasional formatting slips)."""
    last_err = None
    for attempt in range(retries + 1):
        prompt = user_prompt
        if attempt > 0:
            prompt += "\n\nREMINDER: Respond with ONLY the raw JSON object. No markdown fences, no commentary."
        raw = call_ollama(system_prompt, prompt, model=model, host=host)
        try:
            return extract_json(raw)
        except OllamaError as e:
            last_err = e
    raise last_err
