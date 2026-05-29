"""
CyberTA – LLM Provider Abstraction
Supports:
  • Hugging Face Inference API  (Gemma, Qwen, Llama, Mistral …)
  • Ollama local server         (any pulled model)
  • OpenAI-compatible endpoint  (GPT-4o-mini, GPT-3.5, etc.)
  • Anthropic Claude            (claude-sonnet-4-20250514 – kept as fallback)

All providers expose a single function:
    generate(system, messages, **kwargs) -> str
"""

from __future__ import annotations
import os
import json
import requests
from typing import List, Dict

# ── Provider constants ────────────────────────────────────────────────────────

# Hugging Face models available via Inference API (free tier)
HF_MODELS: Dict[str, str] = {
    "Gemma-3 1B (HF)"      : "google/gemma-3-1b-it",
    "Gemma-3 4B (HF)"      : "google/gemma-3-4b-it",
    "Qwen2.5 1.5B (HF)"    : "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5 7B (HF)"      : "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.2 1B (HF)"    : "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2 3B (HF)"    : "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral 7B v0.3 (HF)" : "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3.5 mini (HF)"    : "microsoft/Phi-3.5-mini-instruct",
}

# Ollama models (user must have these pulled locally)
OLLAMA_MODELS: Dict[str, str] = {
    "Ollama – Llama3.2:1b" : "llama3.2:1b",
    "Ollama – Llama3.2:3b" : "llama3.2:3b",
    "Ollama – Gemma3:1b"   : "gemma3:1b",
    "Ollama – Qwen2.5:1.5b": "qwen2.5:1.5b",
    "Ollama – Qwen2.5:3b"  : "qwen2.5:3b",
    "Ollama – Phi3.5"      : "phi3.5",
    "Ollama – Mistral"     : "mistral",
}

# OpenAI models (requires OPENAI_API_KEY)
OPENAI_MODELS: Dict[str, str] = {
    "GPT-4o mini (OpenAI)"  : "gpt-4o-mini",
    "GPT-3.5 Turbo (OpenAI)": "gpt-3.5-turbo",
    "GPT-4o (OpenAI)"       : "gpt-4o",
}

# Anthropic models
ANTHROPIC_MODELS: Dict[str, str] = {
    "Claude Sonnet 4 (Anthropic)": "claude-sonnet-4-20250514",
}

ALL_MODELS: Dict[str, str] = {
    **HF_MODELS,
    **OLLAMA_MODELS,
    **OPENAI_MODELS,
    **ANTHROPIC_MODELS,
}

DEFAULT_MODEL = "Gemma-3 1B (HF)"

SYSTEM_PROMPT = (
    "You are CyberTA, an expert teaching assistant. "
    "Answer the student's question using ONLY the provided context. "
    "If the answer is not in the context, say so honestly. "
    "Be clear, concise, and educational. Cite sources when relevant."
)


# ── HuggingFace Inference API ─────────────────────────────────────────────────

def _hf_generate(model_id: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Build chat payload (messages API – supported by TGI/vLLM hosted models)
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    url = f"https://router.huggingface.co/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama_generate(model_id: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _openai_generate(model_id: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ── Anthropic ────────────────────────────────────────────────────────────────

def _anthropic_generate(model_id: str, system: str, messages: List[Dict], max_tokens: int = 1024) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    resp = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return resp.content[0].text.strip()


# ── Unified entry point ───────────────────────────────────────────────────────

def generate(
    display_name: str,
    system: str,
    messages: List[Dict],
    max_tokens: int = 1024,
) -> str:
    """
    Route to the correct backend based on display_name.
    Raises ValueError for unknown names.
    """
    model_id = ALL_MODELS.get(display_name)
    if model_id is None:
        raise ValueError(f"Unknown model: {display_name}")

    if display_name in HF_MODELS:
        return _hf_generate(model_id, system, messages, max_tokens)
    elif display_name in OLLAMA_MODELS:
        return _ollama_generate(model_id, system, messages, max_tokens)
    elif display_name in OPENAI_MODELS:
        return _openai_generate(model_id, system, messages, max_tokens)
    elif display_name in ANTHROPIC_MODELS:
        return _anthropic_generate(model_id, system, messages, max_tokens)
    else:
        raise ValueError(f"No backend mapped for: {display_name}")


def model_list() -> List[str]:
    return list(ALL_MODELS.keys())


def provider_of(display_name: str) -> str:
    if display_name in HF_MODELS:
        return "HuggingFace"
    if display_name in OLLAMA_MODELS:
        return "Ollama (local)"
    if display_name in OPENAI_MODELS:
        return "OpenAI"
    if display_name in ANTHROPIC_MODELS:
        return "Anthropic"
    return "Unknown"
