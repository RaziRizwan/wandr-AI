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


DEFAULT_MODEL = "openai/gpt-oss-20b:fastest"
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


SEARCH_PLAN_SYSTEM = """You are the search supervisor for Wandr.

You receive a user's parsed travel intent:
{"query": "...", "location": "City, Country"}

Generate 6 to 8 diverse TripAdvisor search queries that will uncover both
popular results and hidden gems in that exact destination.

Rules:
- Return ONLY a valid JSON array of query strings.
- Each query must be short, specific, and suitable for TripAdvisor search.
- Include the original intent first.
- Add variants for hidden gems, local favorites, neighborhoods, culture,
  food, nature, attractions, and underrated places where relevant.
- Do not include the city/country in each query; the app appends location.
- Do not invent place names.
"""


SENTIMENT_AUDIT_SYSTEM = """You are a strict review sentiment auditor for a travel app.

You receive a JSON list of places. Each place has:
- location_id
- name
- local_sentiment from a local ML model
- up to 5 review texts with ratings

For each place, audit whether the local sentiment is reasonable. Use the actual
review text as the source of truth. Return ONLY a valid JSON object:

{
  "location_id": {
    "sentiment_score": 0.0-1.0,
    "sentiment_label": "Excellent"|"Good"|"Mixed"|"Poor"|"Unknown",
    "positive_pct": 0-100,
    "review_count_analyzed": integer
  }
}

Rules:
- If reviews are missing, return Unknown with sentiment_score 0.5.
- Be conservative. Mixed praise and complaints should be Mixed.
- Do not include explanations or markdown.
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
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:240]
        suffix = f" {detail}" if detail else ""
        return "", f"Hugging Face API error (HTTP {resp.status_code}).{suffix}"

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


def plan_search_queries(
    params: dict,
    api_key: str,
    model: str = DEFAULT_MODEL,
    *,
    max_queries: int = 8,
) -> list[str]:
    """
    Ask Hugging Face to supervise TripAdvisor search expansion.

    Returns a deduplicated list of short query strings. The first item is always
    the parsed user intent query so the pipeline preserves the user's request.
    """
    base_query = (params or {}).get("query", "tourist attractions").strip()
    location = (params or {}).get("location", "").strip()
    fallback = _fallback_search_queries(base_query)

    text, err = _chat(
        api_key,
        model,
        [
            {"role": "system", "content": SEARCH_PLAN_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"query": base_query, "location": location},
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=260,
        temperature=0.25,
    )
    if err:
        candidates = fallback
    else:
        try:
            clean_text = re.sub(r"```json|```", "", text).strip()
            array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
            parsed = json.loads(array_match.group(0) if array_match else clean_text)
            candidates = [str(q).strip() for q in parsed if str(q).strip()]
        except Exception:
            candidates = fallback

    queries = []
    seen = set()
    for query in [base_query, *candidates, *fallback]:
        normalized = re.sub(r"\s+", " ", query.strip()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(query.strip())
        if len(queries) >= max_queries:
            break
    return queries


def _fallback_search_queries(base_query: str) -> list[str]:
    normalized = base_query.lower()
    queries = [
        base_query,
        f"hidden gems {base_query}",
        "local favorites",
        "things to do",
    ]
    if any(term in normalized for term in [
        "food", "restaurant", "cafe", "street", "dining", "market", "bakery",
        "breakfast", "lunch", "dinner", "dessert",
    ]):
        queries.extend([
            "local restaurants",
            "food markets",
            "traditional food",
            "cafes",
            "cheap eats",
        ])
    elif any(term in normalized for term in [
        "nature", "park", "beach", "garden", "mountain", "lake", "waterfall",
    ]):
        queries.extend([
            "nature spots",
            "parks",
            "gardens",
            "outdoor attractions",
            "scenic places",
        ])
    else:
        queries.extend([
            "attractions",
            "cultural sites",
            "historic places",
            "museums",
            "landmarks",
        ])
    return queries


def validate_locations(
    requested_location: str,
    spots: list,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> list:
    """
    Validate TripAdvisor results with a Hugging Face chat-completion model.

    Fail-closed: if Hugging Face can parse a validation response, return only
    approved IDs. If the API call itself fails, return the original list so the
    app remains usable.
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

    return [s for s in spots if s.get("location_id") in valid_ids]


def audit_sentiments(
    spots: list,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> list:
    """
    Use Hugging Face as a sentiment supervisor after local DistilBERT analysis.

    This is one batched LLM call for the whole result set, not one call per
    place. If the audit fails, local sentiment values are preserved.
    """
    if not spots:
        return spots

    payload = []
    for spot in spots[:30]:
        reviews = [
            {
                "text": str(r.get("text", ""))[:500],
                "rating": r.get("rating"),
            }
            for r in spot.get("raw_reviews", [])[:5]
            if r.get("text")
        ]
        payload.append({
            "location_id": spot.get("location_id", ""),
            "name": spot.get("name", ""),
            "local_sentiment": spot.get("sentiment", {}),
            "reviews": reviews,
        })

    text, err = _chat(
        api_key,
        model,
        [
            {"role": "system", "content": SENTIMENT_AUDIT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_tokens=1800,
        temperature=0,
    )
    if err:
        return spots

    try:
        clean_text = re.sub(r"```json|```", "", text).strip()
        object_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        audit = json.loads(object_match.group(0) if object_match else clean_text)
    except Exception:
        return spots

    valid_labels = {"Excellent", "Good", "Mixed", "Poor", "Unknown"}
    for spot in spots:
        result = audit.get(str(spot.get("location_id", "")))
        if not isinstance(result, dict):
            continue
        try:
            score = max(0.0, min(1.0, float(result.get("sentiment_score", 0.5))))
            positive_pct = max(0.0, min(100.0, float(result.get("positive_pct", 0))))
            label = str(result.get("sentiment_label", "Unknown"))
            if label not in valid_labels:
                label = "Unknown"
            local = spot.get("sentiment", {})
            spot["sentiment"] = {
                **local,
                "sentiment_score": round(score, 3),
                "sentiment_label": label,
                "positive_pct": round(positive_pct, 1),
                "review_count_analyzed": int(result.get(
                    "review_count_analyzed",
                    local.get("review_count_analyzed", 0),
                )),
                "audited_by_hf": True,
            }
        except Exception:
            continue

    return spots
