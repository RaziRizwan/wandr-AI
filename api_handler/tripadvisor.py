"""
api_handler/tripadvisor.py
==========================
Handles all communication with the TripAdvisor Content API.

WHY TRIPADVISOR API (not Google Places)?
-----------------------------------------
Google Places API was the original choice, but it has geographic restrictions
that make it unavailable or unreliable in certain regions (e.g. Pakistan).
TripAdvisor Content API was chosen because:

  1. Global coverage  — works in all countries without geo-restrictions.
  2. Rich review data — returns up to 5 full review texts per location,
                        which we feed into our ML sentiment model.
  3. Subcategory data — returns cuisine type, subcategory labels (e.g.
                        "Historic Site", "Seafood"), which our theme engine
                        uses for visual card categorisation.
  4. Photo CDN        — high-quality photos with multiple size variants.
  5. Free tier        — 5,000 API calls/month at no cost.

TRADE-OFFS:
  - TripAdvisor returns at most 10 results per search query.
  - Review counts per API call are capped at 5 (sufficient for ML analysis).
  - Results occasionally include wrong-location entries (handled by Hugging Face
    location validation in api_handler/huggingface.py).

API PIPELINE PER SEARCH:
  1. search_locations()    → GET /location/search  → list of location_ids
  2. fetch_spot_details()  → (per location_id, in parallel via ThreadPoolExecutor)
       GET /location/{id}/details  → rating, reviews count, description, cuisine
       GET /location/{id}/photos   → photo CDN URL
       GET /location/{id}/reviews  → up to 5 review texts for ML analysis
  3. Results merged into a spot dict passed to ml_model/ for scoring.
"""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import clean, safe_url, safe_photo, truncate_words

# Base URL for all TripAdvisor Content API v1 endpoints
BASE = "https://api.content.tripadvisor.com/api/v1"

# Required header for all requests
HEADERS = {"accept": "application/json"}


class TripAdvisorRateLimitError(Exception):
    """Raised when TripAdvisor rejects detail/photo/review calls for quota."""


def _get(endpoint: str, params: dict, timeout: int = 8) -> dict:
    """
    Internal helper: perform a GET request to a TripAdvisor endpoint.

    Silently returns an empty dict on any error (timeout, bad status, etc.)
    so that a single failed detail fetch does not crash the whole pipeline.

    Args:
        endpoint: API path, e.g. "/location/123456/details"
        params:   Query parameters dict (must include 'key': api_key)
        timeout:  Seconds before the request is abandoned.

    Returns:
        Parsed JSON dict, or {} on failure.
    """
    try:
        r = requests.get(
            f"{BASE}{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=timeout,
        )
        if r.status_code == 429:
            raise TripAdvisorRateLimitError(
                "TripAdvisor detail/photo/review quota is exhausted. "
                "Wait for the quota to reset or use a key with available quota."
            )
        return r.json() if r.status_code == 200 else {}
    except TripAdvisorRateLimitError:
        raise
    except Exception:
        return {}


def search_locations(query: str, location: str, api_key: str, limit: int = 10) -> tuple:
    """
    Search TripAdvisor for places matching a query string + location.

    Combines the user's intent query ("street food markets") with the
    validated city name ("Karachi, Pakistan") into a single searchQuery.
    This improves result relevance over querying by location alone.

    Args:
        query:    The type of place to search for (from Hugging Face intent parsing).
        location: The validated city+country string (from Hugging Face).
        api_key:  TripAdvisor API key (passed at call time, never hardcoded).
        limit:    Max results to return (default 10, TripAdvisor max is 10).

    Returns:
        Tuple of (results_list, error_string_or_None).
        results_list contains raw TripAdvisor search result dicts.
    """
    params = {
        "key": api_key,
        "language": "en",
        "searchQuery": f"{query} {location}",
    }
    try:
        resp = requests.get(
            f"{BASE}/location/search",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code == 401:
            return None, "TripAdvisor API key is invalid. Check your key."
        if resp.status_code == 429:
            return None, "TripAdvisor rate limit hit. Wait a moment and retry."
        if resp.status_code != 200:
            return None, f"TripAdvisor API error (HTTP {resp.status_code})."
        results = resp.json().get("data", [])[:limit]
        return results, None
    except requests.exceptions.Timeout:
        return None, "TripAdvisor request timed out. Check your internet connection."
    except Exception as e:
        return None, f"Unexpected error fetching locations: {e}"


def fetch_spot_details(result: dict, api_key: str, location: str) -> dict:
    """
    Fetch comprehensive details for a single TripAdvisor location.

    Makes 3 parallel-friendly sub-calls per location:
      1. /details  — rating, review count, description, cuisine, price level
      2. /photos   — first available photo URL (large → medium → original)
      3. /reviews  — up to 5 most recent reviews (text + rating + author)

    The reviews are stored in two forms:
      - raw_reviews: full list passed to ml_model/sentiment.py for analysis
      - display_review: the first review, truncated for card display

    All text fields are passed through clean() to strip any embedded HTML
    before being stored. URLs are validated with safe_url() / safe_photo().

    Args:
        result:   One item from search_locations() result list.
        api_key:  TripAdvisor API key.
        location: City string used as fallback when address is missing.

    Returns:
        A fully populated spot dict ready for ML processing and rendering.
    """
    base_params = {"key": api_key, "language": "en"}
    loc_id = result.get("location_id", "")

    # ── Basic info from search result ──────────────────────────────────────────
    name = clean(result.get("name", ""))
    addr_obj = result.get("address_obj", {})
    full_addr = clean(
        addr_obj.get("address_string", "")
        or addr_obj.get("city", "")
        or location
    )
    # TripAdvisor top-level category: "restaurants", "hotels", "attractions", "geos"
    category = result.get("category", {}).get("key", "attractions")

    # ── Detailed information ────────────────────────────────────────────────────
    det = _get(f"/location/{loc_id}/details", {**base_params, "currency": "USD"})

    rating      = det.get("rating")          # float e.g. 4.5
    num_reviews = det.get("num_reviews", 0)  # int total review count on TripAdvisor
    description = clean(det.get("description", "") or "")
    price_level = clean(det.get("price_level", "") or "")  # "$", "$$", "$$$", "$$$$"

    # Cuisine list (for restaurants): extract up to 2 cuisine types
    cuisine = ", ".join(
        clean(c.get("localized_name", ""))
        for c in det.get("cuisine", [])[:2]
    )

    # Subcategories give finer detail: "Historic Site", "Seafood", "Park", etc.
    subcats = [clean(s.get("name", "")) for s in det.get("subcategory", [])]

    # Validated TripAdvisor listing URL
    ta_url = safe_url(det.get("web_url", ""))

    # Fall back to details address if search result address was empty
    if not full_addr:
        full_addr = clean(det.get("address_obj", {}).get("address_string", "") or location)

    # ── Photo ───────────────────────────────────────────────────────────────────
    # Request only 1 photo. Prefer 'large' for quality, fall back to smaller sizes.
    photo_url = ""
    photos = _get(f"/location/{loc_id}/photos", {**base_params, "limit": 1})
    plist = photos.get("data", [])
    if plist:
        imgs = plist[0].get("images", {})
        raw = (
            imgs.get("large") or imgs.get("medium") or imgs.get("original") or {}
        ).get("url", "")
        photo_url = safe_photo(raw)

    # ── Reviews ─────────────────────────────────────────────────────────────────
    # Fetch up to 5 reviews — more reviews = more reliable ML sentiment signal.
    # Each review dict: { text, rating (1-5), author }
    raw_reviews = []
    rvs = _get(f"/location/{loc_id}/reviews", {**base_params, "limit": 5})
    for rv in rvs.get("data", []):
        txt = clean(rv.get("text", ""))
        if txt:
            raw_reviews.append({
                "text": txt,
                "rating": rv.get("rating"),
                "author": clean(rv.get("user", {}).get("username", "Traveler")),
            })

    # Display review: first review, word-truncated for card readability
    display_review = {}
    if raw_reviews:
        rv0 = raw_reviews[0]
        truncated, was_cut = truncate_words(rv0["text"], max_words=80)
        display_review = {
            "text": truncated,
            "full_text": rv0["text"],  # stored for potential "Read more"
            "was_truncated": was_cut,
            "author": rv0["author"],
            "rating": rv0["rating"],
        }

    return {
        "location_id": loc_id,
        "name": name,
        "address": full_addr,
        "rating": float(rating) if rating else None,
        "num_reviews": int(num_reviews) if num_reviews else 0,
        "category": category,
        "subcategories": subcats,
        "cuisine": cuisine,
        "description": description[:200] + "…" if len(description) > 200 else description,
        "price_level": price_level,
        "ta_url": ta_url,
        "photo_url": photo_url,
        "raw_reviews": raw_reviews,       # → fed to ml_model/sentiment.py
        "display_review": display_review,  # → rendered on card
    }


def fetch_all_spots(
    query: str,
    location: str,
    api_key: str,
    max_results: int = 10,
) -> tuple:
    """
    Full TripAdvisor data pipeline: search → parallel detail fetch.

    Uses ThreadPoolExecutor to fetch details for all locations concurrently.
    This reduces total wait time from O(n * single_request) to roughly
    O(single_request) since all 10 detail fetches run simultaneously.

    Args:
        query:       Search intent string from Hugging Face (e.g. "street food stalls").
        location:    City + country string from Hugging Face (e.g. "Lahore, Pakistan").
        api_key:     TripAdvisor API key.
        max_results: Maximum spots to return (capped at 10 by TripAdvisor).

    Returns:
        Tuple of (spots_list, error_string_or_None).
        spots_list: list of fully populated spot dicts.
    """
    # Step 1: Get list of matching location IDs
    results, err = search_locations(query, location, api_key, max_results)
    if err:
        return None, err
    if not results:
        return [], None

    # Step 2: Fetch full details for each location in parallel (6 workers)
    spots = []
    first_error = None
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(fetch_spot_details, r, api_key, location): r
            for r in results
        }
        for future in as_completed(futures):
            try:
                spots.append(future.result())
            except TripAdvisorRateLimitError as e:
                first_error = str(e)
            except Exception:
                # Skip any spot that fails — don't crash the whole result set
                pass

    if first_error and not spots:
        return None, first_error
    return spots, None


def fetch_supervised_spots(
    queries: list[str],
    location: str,
    api_key: str,
    max_results_per_query: int = 10,
    max_candidates: int = 36,
) -> tuple:
    """
    Run multiple TripAdvisor searches from a Hugging Face search plan.

    TripAdvisor caps one search at 10 results, so this function broadens recall
    by searching several targeted query variants, deduplicating by location_id,
    then fetching details for the combined candidate set in parallel.
    """
    if not queries:
        return [], None

    deduped_results = {}
    first_error = None

    for query in queries:
        results, err = search_locations(
            query=query,
            location=location,
            api_key=api_key,
            limit=max_results_per_query,
        )
        if err and not first_error:
            first_error = err
        if not results:
            continue
        for result in results:
            loc_id = result.get("location_id")
            if loc_id and loc_id not in deduped_results:
                result["matched_query"] = query
                deduped_results[loc_id] = result
            if len(deduped_results) >= max_candidates:
                break
        if len(deduped_results) >= max_candidates:
            break

    if not deduped_results:
        return (None, first_error) if first_error else ([], None)

    spots = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(fetch_spot_details, r, api_key, location): r
            for r in deduped_results.values()
        }
        for future in as_completed(futures):
            try:
                spots.append(future.result())
            except TripAdvisorRateLimitError as e:
                first_error = str(e)
            except Exception:
                pass

    if first_error and not spots:
        return None, first_error
    return spots, None
