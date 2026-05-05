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
- Preserve the requested category strictly. If the user asked for food, every
  query must be food/restaurant/cafe/market related. If they asked for hotels,
  every query must be lodging related. If they asked for nature, every query
  must be nature/outdoor related.
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


CATEGORY_AUDIT_SYSTEM = """You are a strict travel-result category auditor.

You receive:
1. The user's requested intent category
2. A JSON list of TripAdvisor places with location_id, name, category, cuisine,
   subcategories, address, description, and review snippets.

Return ONLY a valid JSON array of location_ids that match the requested intent.

Strict rules:
- If requested intent is food, include ONLY restaurants, cafes, bakeries, food
  markets, street-food places, dessert shops, bars, or clearly food-focused
  listings. Exclude landmarks, hotels, tours, mosques, forts, museums, parks,
  shops, and generic attractions even if they are popular.
- If requested intent is hotel, include ONLY lodging/hotels.
- If requested intent is nature, include ONLY parks, gardens, beaches, lakes,
  mountains, forests, waterfalls, outdoor scenic places, and nature attractions.
- If requested intent is attraction/culture/history, include ONLY relevant
  attractions and exclude unrelated restaurants/hotels unless the user asked
  for food or lodging.
- Be strict. If unsure, exclude it.
"""


FOOD_TERMS = {
    "food", "restaurant", "restaurants", "cafe", "cafes", "coffee", "bar",
    "bars", "bistro", "grill", "pizza", "bakery", "bakeries", "dining",
    "street food", "food market", "food markets", "dessert", "breakfast", "lunch",
    "dinner", "eat", "eats", "cuisine", "tea", "rooftop cafe",
    "dhaba", "bbq", "barbecue", "kitchen", "tandoor", "biryani", "burger",
    "shawarma", "sweets", "ice cream", "snacks",
}
HOTEL_TERMS = {"hotel", "hotels", "lodging", "stay", "resort", "hostel"}
NATURE_TERMS = {
    "nature", "park", "parks", "beach", "garden", "gardens", "lake",
    "mountain", "forest", "waterfall", "outdoor", "scenic",
}

FOOD_NEGATIVE_TERMS = {
    "fort", "mosque", "museum", "tour", "guided", "airport", "park",
    "garden", "mall", "landmark", "monument", "temple", "church", "hotel",
    "resort", "hostel", "spa", "zoo", "tomb", "palace", "gate",
}


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
    requested_category = infer_requested_category(params)
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
        if not _query_matches_requested_category(query, requested_category):
            continue
        normalized = re.sub(r"\s+", " ", query.strip()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append(query.strip())
        if len(queries) >= max_queries:
            break
    if not queries:
        queries = fallback[:max_queries]
    return queries


def _fallback_search_queries(base_query: str) -> list[str]:
    normalized = base_query.lower()
    queries = [
        base_query,
        f"hidden gems {base_query}",
    ]
    if any(term in normalized for term in [
        "food", "restaurant", "cafe", "street", "dining", "market", "bakery",
        "breakfast", "lunch", "dinner", "dessert",
    ]):
        queries.extend([
            "local food favorites",
            "local restaurants",
            "street food",
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


def _query_matches_requested_category(query: str, requested_category: str) -> bool:
    normalized = query.lower()
    if requested_category == "food":
        return any(term in normalized for term in FOOD_TERMS)
    if requested_category == "hotel":
        return any(term in normalized for term in HOTEL_TERMS)
    if requested_category == "nature":
        return any(term in normalized for term in NATURE_TERMS)
    if requested_category == "attraction":
        return not any(term in normalized for term in FOOD_TERMS | HOTEL_TERMS)
    return True


def infer_requested_category(params: dict) -> str:
    query = (params or {}).get("query", "").lower()
    if any(term in query for term in FOOD_TERMS):
        return "food"
    if any(term in query for term in HOTEL_TERMS):
        return "hotel"
    if any(term in query for term in NATURE_TERMS):
        return "nature"
    return "attraction"


def _spot_text(spot: dict) -> str:
    parts = [
        spot.get("name", ""),
        spot.get("category", ""),
        spot.get("cuisine", ""),
        spot.get("description", ""),
        " ".join(spot.get("subcategories", []) or []),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def _deterministic_category_match(spot: dict, requested_category: str) -> bool:
    cat = str(spot.get("category", "")).lower()
    cuisine = str(spot.get("cuisine", "")).lower()
    text = _spot_text(spot)
    subs = " ".join(str(s).lower() for s in (spot.get("subcategories", []) or []))

    if requested_category == "food":
        if any(term in text for term in ["tour", "guided", "sightseeing"]):
            return False
        if cat == "restaurants":
            return True
        if cuisine:
            return True
        has_food_signal = any(term in text for term in FOOD_TERMS)
        food_place_name = any(term in str(spot.get("name", "")).lower() for term in [
            "food", "restaurant", "cafe", "barbecue", "bbq", "kitchen",
            "bakery", "sweets", "biryani", "burger", "dhaba",
        ])
        has_negative_signal = any(term in text for term in FOOD_NEGATIVE_TERMS)
        if food_place_name:
            return True
        return has_food_signal and not has_negative_signal

    if requested_category == "hotel":
        return cat == "hotels" or any(term in text for term in HOTEL_TERMS)

    if requested_category == "nature":
        return cat == "geos" or any(term in subs or term in text for term in NATURE_TERMS)

    if requested_category == "attraction":
        return cat in {"attractions", "geos"} and cat != "restaurants"

    return True


def filter_by_requested_category(
    params: dict,
    spots: list,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> list:
    """
    Strictly keep only places matching the user's requested category.

    Deterministic category rules run first. Hugging Face then audits the
    remaining candidate set for stricter semantic matching.
    """
    if not spots:
        return spots

    requested_category = infer_requested_category(params)
    deterministic = [
        spot for spot in spots
        if _deterministic_category_match(spot, requested_category)
    ]
    if not deterministic:
        return []

    audit_payload = [
        {
            "location_id": s.get("location_id", ""),
            "name": s.get("name", ""),
            "category": s.get("category", ""),
            "cuisine": s.get("cuisine", ""),
            "subcategories": s.get("subcategories", []),
            "address": s.get("address", ""),
            "description": s.get("description", ""),
            "reviews": [
                str(r.get("text", ""))[:220]
                for r in s.get("raw_reviews", [])[:2]
                if r.get("text")
            ],
        }
        for s in deterministic[:36]
    ]

    text, err = _chat(
        api_key,
        model,
        [
            {"role": "system", "content": CATEGORY_AUDIT_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({
                    "requested_category": requested_category,
                    "places": audit_payload,
                }, ensure_ascii=False),
            },
        ],
        max_tokens=600,
        temperature=0,
    )
    if err:
        return deterministic

    try:
        clean_text = re.sub(r"```json|```", "", text).strip()
        array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
        valid_ids = set(json.loads(array_match.group(0) if array_match else clean_text))
    except Exception:
        return deterministic

    return [s for s in deterministic if s.get("location_id") in valid_ids]


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
