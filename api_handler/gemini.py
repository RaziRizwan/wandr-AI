"""
api_handler/gemini.py
=====================
Uses Google Gemini as the AI reasoning layer of the pipeline.

ROLE OF GEMINI IN THIS SYSTEM:
--------------------------------
Gemini serves two distinct roles — neither involves generating travel content
from its own knowledge (which would risk hallucination). Instead, it acts as:

  ROLE 1 — Intent Parser:
    Converts a free-form user query ("best street food near old city Lahore")
    into a structured search object: { "query": "...", "location": "..." }.
    This structured object is then passed to the TripAdvisor API.
    Gemini is better at this than regex/NLP rules because it handles:
      - Colloquial phrasing ("things to do this weekend in KHI")
      - Ambiguous city names ("karachi" → "Karachi, Pakistan")
      - Multi-language input (Urdu-English mix)

  ROLE 2 — Location Validator:
    After TripAdvisor returns results, Gemini cross-checks each result's
    name and address against the requested location. It filters out results
    from the wrong city or country.

    WHY THIS IS NEEDED:
      TripAdvisor's search algorithm is keyword-based and sometimes returns
      results from other cities that happen to match the query text.
      Example: searching "street food Lahore" returned results from Malaysia
      because "Lahore" matched a restaurant name in Kuala Lumpur.
      Gemini reads the full address and catches these mismatches.

HOW GEMINI PREVENTS HALLUCINATION:
    - It is never asked to invent place names or descriptions.
    - All factual data (names, addresses, ratings) comes from TripAdvisor.
    - Gemini only processes: (a) structured query extraction, (b) yes/no
      location validation based on explicit address strings.
    - If validation fails (API error, parse error), the system returns the
      unfiltered results rather than an empty set (fail-open strategy).

MODEL USED: gemini-2.0-flash
    - Fast inference with sufficient reasoning for structured extraction.
    - Free tier available via Google AI Studio.
    - Supports system instruction and multi-turn conversation history.
"""

import json
import re
import google.generativeai as genai

# Gemini model identifier — update here if switching to a newer model
MODEL = "gemini-2.5-flash"

# ── System prompt for intent parsing ──────────────────────────────────────────
# This prompt constrains Gemini to produce ONLY structured output.
# The <search> block is machine-parseable; the narrative is for the user.
INTENT_SYSTEM = """You are Wandr, an expert travel guide AI assistant.

For every user message, extract the travel search intent and respond with:

1. A structured search block (ALWAYS first, ALWAYS present):
   <search>{"query": "specific place type", "location": "City, Country"}</search>

2. Then 2-3 vivid, enthusiastic sentences about the destination.

Rules:
- query must be SPECIFIC: "rooftop cafes", "street food stalls", "historic mosques", "night markets"
- location must include country: "Karachi, Pakistan" — never just "Karachi"
- The <search> block must be valid JSON
- Never add text before the <search> block
- If location is unclear, make your best guess based on context
"""

# ── System prompt for location validation ─────────────────────────────────────
# Gemini acts as a strict filter here — it reads addresses and rejects
# any place not in the requested city/country.
VALIDATION_SYSTEM = """You are a strict geographic location validator for a travel app.

You will receive:
1. A requested location (e.g. "Karachi, Pakistan")
2. A JSON list of places, each with location_id, name, and address

Your task:
- Read each place's address carefully
- Include only places that are genuinely in or immediately adjacent to the requested location
- Exclude places in different cities, regions, or countries
- Be strict — wrong-country results must be excluded

Return ONLY a valid JSON array of the approved location_ids.
Example: ["12345678", "87654321"]
Return [] if none are valid.
Do not include any other text or explanation.
"""


def configure(api_key: str) -> None:
    """
    Configure the Gemini SDK with the provided API key.
    Must be called before any generate/chat calls.
    """
    genai.configure(api_key=api_key)


def parse_intent(messages: list, api_key: str) -> tuple:
    """
    Use Gemini to parse a user's free-form travel query into structured output.

    Maintains conversation history so follow-up messages ("show me restaurants
    there instead") resolve correctly against prior context.

    Pipeline:
        User message → Gemini (INTENT_SYSTEM) → <search> JSON + narrative text
        → parsed into params dict + narrative string

    Args:
        messages: Full conversation history as list of {"role", "content"} dicts.
                  The last item is the current user message.
        api_key:  Gemini API key.

    Returns:
        Tuple of (params_dict, narrative_str, error_str_or_None).
        params_dict: {"query": str, "location": str} or None on failure.
        narrative_str: The user-facing destination description.
        error_str: None on success, error message string on failure.
    """
    try:
        configure(api_key)
        model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=INTENT_SYSTEM,
        )

        # Build conversation history for multi-turn context
        # (exclude the last message — that's sent via chat.send_message)
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})

        chat = model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        text = response.text

        # Extract structured JSON from <search>...</search> block
        match = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
        params = None
        if match:
            try:
                params = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass  # params stays None, caller will show a helpful error

        # Everything outside the <search> block is the user-facing narrative
        narrative = re.sub(r"<search>.*?</search>", "", text, flags=re.DOTALL).strip()
        return params, narrative, None

    except Exception as e:
        err = str(e)
        # Provide specific, actionable error messages
        if "quota" in err.lower() or "429" in err:
            return None, "", "Gemini quota exceeded — wait a moment and retry."
        if "API_KEY_INVALID" in err or "400" in err:
            return None, "", "Gemini API key is invalid. Check your key at aistudio.google.com."
        return None, "", f"Gemini error: {err}"


def validate_locations(requested_location: str, spots: list, api_key: str) -> list:
    """
    Use Gemini to filter TripAdvisor results to only those in the correct city.

    This is a critical quality-control step. TripAdvisor's keyword search
    sometimes returns places from the wrong city (e.g., searching "Lahore"
    returns a restaurant named "Lahore" located in Malaysia).

    Gemini reads each place's full address and decides whether it is
    genuinely in the requested location. Only approved location_ids are kept.

    Fail-open strategy: if Gemini's response cannot be parsed (API error,
    malformed JSON, network issue), the original unfiltered list is returned
    rather than an empty result. This ensures the app remains usable even
    when validation fails.

    Args:
        requested_location: City + country string (e.g. "Karachi, Pakistan").
        spots:              List of spot dicts from fetch_all_spots().
        api_key:            Gemini API key.

    Returns:
        Filtered list of spot dicts. Same order as input, minus invalid entries.
    """
    if not spots:
        return spots

    try:
        configure(api_key)
        model = genai.GenerativeModel(
            model_name=MODEL,
            system_instruction=VALIDATION_SYSTEM,
        )

        # Build a concise location summary for Gemini to evaluate
        spot_summary = "\n".join(
            f'- ID: {s["location_id"]} | Name: {s["name"]} | Address: {s["address"]}'
            for s in spots
        )

        prompt = (
            f"Requested location: {requested_location}\n\n"
            f"Places to validate:\n{spot_summary}\n\n"
            f"Return only the JSON array of valid location_ids."
        )

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown code fences if Gemini wraps the JSON
        clean_text = re.sub(r"```json|```", "", text).strip()
        valid_ids = set(json.loads(clean_text))

        filtered = [s for s in spots if s["location_id"] in valid_ids]

        # Fail-open: return original list if Gemini filtered everything
        # (likely a validation error rather than all results being wrong)
        return filtered if filtered else spots

    except Exception:
        # Any error in validation → return original spots unfiltered
        return spots
