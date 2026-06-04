"""
CyberTA – LLM Providers (Ollama only)
Models: llama3.2:1b, llama3.2:3b, gemma3:12b
"""

from __future__ import annotations
import os
import requests
from typing import List, Dict

OLLAMA_MODELS: Dict[str, str] = {
    "Llama 3.2 – 1B (fast)": "llama3.2:1b",
    "Llama 3.2 – 3B (balanced)": "llama3.2:3b",
    "Gemma 3 – 12B (best quality)": "gemma3:12b",
}

DEFAULT_MODEL = "Llama 3.2 – 1B (fast)"

SYSTEM_PROMPT = (
    "You are CyberTA, an expert teaching assistant. "
    "Answer the student's question using ONLY the provided context. "
    "If the answer is not in the context, say so honestly. "
    "Be clear, concise, and educational. Cite sources when relevant."
)


def _extract_content(raw) -> str:
    """Handle both string and Gradio-6 list content format."""
    if isinstance(raw, list):
        return " ".join(x.get("text", "") for x in raw if isinstance(x, dict))
    return str(raw) if raw else ""


def _ollama_generate(model_id: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Clean messages: normalize content, merge consecutive same-role
    clean = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content_text = _extract_content(m.get("content", ""))
        if role not in ("user", "assistant"):
            continue
        if not content_text.strip():
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += " " + content_text
        else:
            clean.append({"role": role, "content": content_text})

    # Must start with user
    if not clean or clean[0]["role"] != "user":
        clean = [{"role": "user", "content": "Hello"}] + clean

    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}] + clean,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return resp.json()["message"]["content"].strip()


def generate(display_name: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    model_id = OLLAMA_MODELS.get(display_name)
    if model_id is None:
        raise ValueError(f"Unknown model: {display_name}")
    return _ollama_generate(model_id, system, messages, max_tokens)


def model_list() -> List[str]:
    return list(OLLAMA_MODELS.keys())


def provider_of(display_name: str) -> str:
    return "Ollama (local)"
