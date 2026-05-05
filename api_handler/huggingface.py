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
   <search>{"query": "specific place type", "location": "requested geographic scope"}</search>

2. Then 2-3 vivid, enthusiastic sentences about the destination.

Rules:
- query must be SPECIFIC: "rooftop cafes", "street food stalls", "historic mosques", "night markets"
- location is the user's requested geographic scope. It may be a city,
  region, country, continent, island, mountain range, or abstract travel area.
- Preserve broad scopes instead of narrowing them. Examples:
  "castles across Europe" -> {"query": "castles", "location": "Europe"}
  "beaches in Southeast Asia" -> {"query": "beaches", "location": "Southeast Asia"}
  "northern areas to visit in Pakistan" -> {"query": "scenic valleys and mountain attractions", "location": "Northern Pakistan, Pakistan"}
- For a city, include country: "Karachi, Pakistan" - never just "Karachi"
- The <search> block must be valid JSON
- Never add text before the <search> block
- If the user gives a broad place, keep it broad. Do not substitute a famous
  city, capital, or nearby tourist hub unless the user asked for that city.
- If no place is stated or clearly implied, use an empty location string.
"""


VALIDATION_SYSTEM = """You are a strict geographic location validator for a travel app.

You will receive:
1. A requested location, such as "Karachi, Pakistan", "Europe", or
   "Northern Pakistan, Pakistan"
2. A JSON list of places, each with location_id, name, and address

Your task:
- Include only places genuinely inside the requested geographic scope.
- For a city/locality, allow only that city or immediately adjacent places.
- For a region, country, or continent, allow results anywhere inside that
  broader scope and do not collapse validation to one city.
- Exclude places in unrelated cities, regions, countries, or continents.
- Be strict: wrong-country or wrong-continent results must be excluded.

Return ONLY a valid JSON array of approved location_ids.
Example: ["12345678", "87654321"]
Return [] if none are valid.
Do not include any other text or explanation.
"""


SEARCH_PLAN_SYSTEM = """You are the search supervisor for Wandr.

You receive a user's parsed travel intent:
{"query": "...", "location": "requested geographic scope"}

Generate 6 to 8 diverse TripAdvisor search queries that will uncover both
popular results and hidden gems in that exact destination or region.

Rules:
- Return ONLY a valid JSON array of query strings.
- Each query must be short, specific, and suitable for TripAdvisor search.
- Include the original intent first.
- Add variants for hidden gems, local favorites, neighborhoods, culture,
  food, nature, attractions, and underrated places where relevant.
- If location is broad, such as Europe, Asia, a whole country, or a large
  region, diversify across representative subregions/countries in the query
  text. Example for "castles" in "Europe": "castles France", "castles Germany",
  "castles Czech Republic", "castles Romania".
- Preserve the requested category strictly. If the user asked for food, every
  query must be food/restaurant/cafe/market related. If they asked for hotels,
  every query must be lodging related. If they asked for nature, every query
  must be nature/outdoor related.
- For city-level searches, do not include the city/country in each query; the
  app appends location. For broad regional searches, country/subregion words
  may be included to spread results intelligently.
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
    "mountain", "mountains", "valley", "valleys", "forest", "waterfall",
    "outdoor", "scenic", "meadow", "meadows", "national park",
}

FOOD_NEGATIVE_TERMS = {
    "fort", "mosque", "museum", "tour", "guided", "airport", "park",
    "garden", "mall", "landmark", "monument", "temple", "church", "hotel",
    "resort", "hostel", "spa", "zoo", "tomb", "palace", "gate",
}

PLACE_NEGATIVE_TERMS = {
    "hotel", "resort", "hostel", "inn", "tour", "tours", "guided",
    "travel", "travels", "expedition", "expeditions", "airport",
}

NORTHERN_PAKISTAN_REQUEST_TERMS = {
    "northern areas", "northern area", "north pakistan", "northern pakistan",
    "north of pakistan", "gilgit baltistan", "gilgit-baltistan",
}

NORTHERN_PAKISTAN_TERMS = {
    "gilgit", "gilgit-baltistan", "gilgit baltistan", "baltistan", "hunza",
    "skardu", "naran", "kaghan", "swat", "kalam", "chitral", "kumrat",
    "deosai", "fairy meadows", "khunjerab", "passu", "attabad", "rakaposhi",
    "naltar", "shogran", "siri paye", "neelum", "azad kashmir", "ajk",
    "muzaffarabad", "murree", "nathia gali", "nathiagali", "ayubia",
    "malam jabba", "dir", "karakoram", "himalaya", "himalayan",
    "k2", "babusar", "batakundi", "ali abad", "aliabad", "gulmit",
    "khaplu", "kachura", "shigar", "astore", "rama",
}

NON_NORTHERN_PAKISTAN_TERMS = {
    "sindh", "karachi", "hyderabad", "thatta", "mohenjo", "moenjodaro",
    "lahore", "multan", "faisalabad", "bahawalpur", "balochistan",
    "quetta", "gwadar", "makran", "punjab", "thar",
}

BROAD_SCOPE_PATTERNS = {
    "Eastern Europe": [r"\beastern europe\b"],
    "Western Europe": [r"\bwestern europe\b"],
    "Central Europe": [r"\bcentral europe\b"],
    "Southern Europe": [r"\bsouthern europe\b"],
    "Northern Europe": [r"\bnorthern europe\b"],
    "North America": [r"\bnorth america\b"],
    "South America": [r"\bsouth america\b"],
    "Southeast Asia": [r"\bsoutheast asia\b"],
    "Middle East": [r"\bmiddle east\b"],
    "Scandinavia": [r"\bscandinavia\b"],
    "Balkans": [r"\bbalkans\b"],
    "Europe": [r"\beurope\b"],
    "Asia": [r"\basia\b"],
    "Africa": [r"\bafrica\b"],
    "Oceania": [r"\boceania\b"],
}

EUROPE_COUNTRY_TERMS = {
    "albania", "andorra", "austria", "belgium", "bosnia", "bulgaria",
    "croatia", "cyprus", "czech republic", "czechia", "denmark", "england",
    "estonia", "finland", "france", "germany", "greece", "hungary",
    "iceland", "ireland", "italy", "latvia", "lithuania", "luxembourg",
    "malta", "monaco", "montenegro", "netherlands", "norway", "poland",
    "portugal", "romania", "scotland", "serbia", "slovakia", "slovenia",
    "spain", "sweden", "switzerland", "ukraine", "united kingdom", "wales",
}

EUROPE_CITY_TERMS = {
    "amsterdam", "athens", "barcelona", "berlin", "bratislava", "brussels",
    "bucharest", "budapest", "copenhagen", "dubrovnik", "edinburgh",
    "florence", "krakow", "lisbon", "london", "madrid", "munich", "oslo",
    "paris", "porto", "prague", "reykjavik", "riga", "rome", "salzburg",
    "stockholm", "tallinn", "venice", "vienna", "vilnius", "warsaw",
    "zagreb", "zurich",
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
    params = normalize_regional_intent(params, messages[-1].get("content", ""))
    return params, narrative, None


def normalize_regional_intent(params: dict, user_text: str) -> dict:
    """
    Correct broad regional phrases that chat models often collapse to one city.

    Example: "northern areas to visit in Pakistan" should not become Murree
    only. It should stay a northern-Pakistan regional search.
    """
    query_text = f"{user_text} {params.get('query', '')} {params.get('location', '')}".lower()
    explicit_scope = _explicit_broad_scope(user_text)
    if explicit_scope:
        params = dict(params)
        params["location"] = explicit_scope

    if any(term in query_text for term in NORTHERN_PAKISTAN_REQUEST_TERMS):
        params = dict(params)
        params["location"] = "Northern Pakistan, Pakistan"
        if not any(term in params.get("query", "").lower() for term in FOOD_TERMS | HOTEL_TERMS):
            params["query"] = "scenic valleys and mountain attractions"
    return params


def _explicit_broad_scope(user_text: str) -> str:
    normalized = user_text.lower()
    for scope in sorted(BROAD_SCOPE_PATTERNS, key=len, reverse=True):
        if any(re.search(pattern, normalized) for pattern in BROAD_SCOPE_PATTERNS[scope]):
            return scope
    return ""


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
    fallback = _fallback_search_queries(base_query, location)

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

    query_pool = [base_query, *candidates, *fallback]
    if is_broad_scope(location):
        query_pool = [base_query, *fallback, *candidates]

    queries = []
    seen = set()
    for query in query_pool:
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


def _fallback_search_queries(base_query: str, location: str = "") -> list[str]:
    normalized = base_query.lower()
    location_normalized = location.lower()
    queries = [
        base_query,
        f"hidden gems {base_query}",
    ]
    if is_europe_scope(location_normalized):
        queries.extend(_europe_fallback_queries(normalized))
        return queries
    if is_northern_pakistan_scope(location_normalized):
        queries.extend([
            "Hunza Valley",
            "Skardu attractions",
            "Gilgit Baltistan scenic places",
            "Naran Kaghan valley",
            "Swat Valley",
            "Kumrat Valley",
            "Deosai National Park",
            "Fairy Meadows",
        ])
        return queries
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


def _europe_fallback_queries(normalized_query: str) -> list[str]:
    if any(term in normalized_query for term in ["castle", "castles", "fortress", "fortresses"]):
        return [
            "castles France",
            "castles Germany",
            "castles Czech Republic",
            "castles Romania",
            "castles Spain",
            "castles Portugal",
            "castles Austria",
            "castles Scotland",
        ]
    if any(term in normalized_query for term in FOOD_TERMS):
        return [
            "local food Italy",
            "food markets Spain",
            "traditional restaurants France",
            "cafes Austria",
            "street food Germany",
        ]
    if any(term in normalized_query for term in NATURE_TERMS):
        return [
            "national parks Switzerland",
            "scenic lakes Italy",
            "mountain attractions Austria",
            "waterfalls Iceland",
            "parks Slovenia",
        ]
    return [
        "historic places France",
        "cultural sites Italy",
        "landmarks Germany",
        "museums Netherlands",
        "old towns Czech Republic",
    ]


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


def is_northern_pakistan_scope(location: str) -> bool:
    normalized = location.lower()
    return any(term in normalized for term in [
        "northern pakistan", "northern areas", "gilgit", "baltistan", "hunza",
        "skardu", "swat", "naran", "kaghan", "kashmir",
    ])


def is_broad_scope(location: str) -> bool:
    normalized = location.lower().strip()
    if is_europe_scope(normalized) or is_northern_pakistan_scope(normalized):
        return True
    return any(scope.lower() == normalized for scope in BROAD_SCOPE_PATTERNS)


def is_europe_scope(location: str) -> bool:
    return bool(re.search(r"\beurope\b", location.lower()))


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
        if any(term in text for term in PLACE_NEGATIVE_TERMS):
            return False
        return cat == "geos" or any(term in subs or term in text for term in NATURE_TERMS)

    if requested_category == "attraction":
        if any(term in text for term in PLACE_NEGATIVE_TERMS):
            return False
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

    deterministic_spots = filter_by_location_scope(requested_location, spots)
    if not deterministic_spots:
        return []

    place_payload = [
        {
            "location_id": s.get("location_id", ""),
            "name": s.get("name", ""),
            "address": s.get("address", ""),
        }
        for s in deterministic_spots
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
        return deterministic_spots

    try:
        clean_text = re.sub(r"```json|```", "", text).strip()
        array_match = re.search(r"\[.*\]", clean_text, re.DOTALL)
        valid_ids = set(json.loads(array_match.group(0) if array_match else clean_text))
    except Exception:
        return deterministic_spots

    return [s for s in deterministic_spots if s.get("location_id") in valid_ids]


def filter_by_location_scope(requested_location: str, spots: list) -> list:
    """
    Deterministically reject impossible geography before LLM validation.

    This prevents regional Pakistan searches from returning India, Vietnam, or
    non-northern Pakistani provinces when TripAdvisor search is noisy.
    """
    requested = requested_location.lower()
    if is_europe_scope(requested):
        europe_spots = [s for s in spots if _is_europe_spot(s)]
        return europe_spots or spots
    if is_northern_pakistan_scope(requested):
        return [s for s in spots if _is_northern_pakistan_spot(s)]
    if "pakistan" in requested:
        return [s for s in spots if "pakistan" in _location_text(s)]
    return spots


def _location_text(spot: dict) -> str:
    parts = [
        spot.get("name", ""),
        spot.get("address", ""),
        spot.get("description", ""),
        " ".join(spot.get("subcategories", []) or []),
    ]
    return " ".join(str(part).lower() for part in parts if part)


def _is_northern_pakistan_spot(spot: dict) -> bool:
    text = _location_text(spot)
    if "pakistan" not in text:
        return False
    if any(term in text for term in NON_NORTHERN_PAKISTAN_TERMS):
        return False
    return any(term in text for term in NORTHERN_PAKISTAN_TERMS)


def _is_europe_spot(spot: dict) -> bool:
    text = _location_text(spot)
    return any(term in text for term in EUROPE_COUNTRY_TERMS | EUROPE_CITY_TERMS)


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
