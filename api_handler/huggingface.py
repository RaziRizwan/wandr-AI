"""
api_handler/huggingface.py
==========================
Hugging Face Inference Providers reasoning layer for Wandr.

This module is the LLM reasoning layer on the hf-inference-experiment branch.
It keeps the same two responsibilities:
  1. Parse a free-form travel query into {"query": ..., "location": ...}
  2. Validate TripAdvisor results against the requested city/country

All factual place data still comes from TripAdvisor.
"""

from __future__ import annotations

import json
import re

import requests


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M:fastest"
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


INTENT_SYSTEM = """You are Wandr, an expert travel guide AI assistant.

For every user message, extract the travel search intent and respond with:

1. A structured search block (ALWAYS first, ALWAYS present):
   <search>{"query": "specific place type", "location": "City, Country"}</search>

2. Then 2-3 vivid, enthusiastic sentences about the destination.

Rules:
- query must be SPECIFIC: "rooftop cafes", "street food stalls", "historic mosques", "night markets"
- location must include country: "Karachi, Pakistan" - never just "Karachi"
- The <search> block must be valid JSON
- Never add text before the <search> block
- If location is unclear, make your best guess based on context
"""


VALIDATION_SYSTEM = """You are a strict geographic location validator for a travel app.

You will receive:
1. A requested location, such as "Karachi, Pakistan"
2. A JSON list of places, each with location_id, name, and address

Your task:
- Include only places genuinely in or immediately adjacent to the requested location
- Exclude places in different cities, regions, or countries
- Be strict: wrong-country results must be excluded

Return ONLY a valid JSON array of approved location_ids.
Example: ["12345678", "87654321"]
Return [] if none are valid.
Do not include any other text or explanation.
"""


def _chat(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 450,
    temperature: float = 0.2,
) -> tuple[str, str | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        return "", "Hugging Face request timed out. Try again."
    except Exception as e:
        return "", f"Hugging Face request failed: {e}"

    if resp.status_code in (401, 403):
        return "", "Hugging Face token is invalid or missing Inference Providers permission."
    if resp.status_code == 429:
        return "", "Hugging Face rate limit hit. Wait a moment and retry."
    if resp.status_code >= 400:
        return "", f"Hugging Face API error (HTTP {resp.status_code})."

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"], None
    except Exception:
        return "", "Hugging Face returned an unexpected response format."


def parse_intent(messages: list, api_key: str, model: str = DEFAULT_MODEL) -> tuple:
    """
    Parse user travel intent with a Hugging Face chat-completion model.

    Returns:
        (params_dict, narrative_str, error_str_or_None)
    """
    hf_messages = [{"role": "system", "content": INTENT_SYSTEM}]
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        hf_messages.append({"role": role, "content": m.get("content", "")})

    text, err = _chat(
        api_key,
        model,
        hf_messages,
        max_tokens=500,
        temperature=0.35,
    )
    if err:
        return None, "", err

    match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    params = None
    if match:
        try:
            params = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            params = None

    narrative = re.sub(r"<search>.*?</search>", "", text, flags=re.DOTALL).strip()
    if not params:
        return None, "", "Hugging Face could not parse the travel query. Try mentioning a city explicitly."
    return params, narrative, None


def validate_locations(
    requested_location: str,
    spots: list,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> list:
    """
    Validate TripAdvisor results with a Hugging Face chat-completion model.

    Fail-open: if validation fails, return the original TripAdvisor results.
    """
    if not spots:
        return spots

    place_payload = [
        {
            "location_id": s.get("location_id", ""),
            "name": s.get("name", ""),
            "address": s.get("address", ""),
        }
        for s in spots
    ]
    user_prompt = (
        f"Requested location: {requested_location}\n\n"
        f"Places JSON:\n{json.dumps(place_payload, ensure_ascii=False)}"
    )
    text, err = _chat(
        api_key,
        model,
        [
            {"role": "system", "content": VALIDATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=220,
        temperature=0,
    )
    if err:
        return spots

    try:
        clean_text = re.sub(r"```json|```", "", text).strip()
        array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
        valid_ids = set(json.loads(array_match.group(0) if array_match else clean_text))
    except Exception:
        return spots

    filtered = [s for s in spots if s.get("location_id") in valid_ids]
    return filtered if filtered else spots
