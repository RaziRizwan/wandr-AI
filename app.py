"""
app.py — Wandr: AI-Powered Tourist Guide
=========================================
Streamlit application entry point and pipeline orchestrator.

PROJECT OVERVIEW:
    Wandr is a portfolio-grade AI/ML travel recommendation system that:
      1. Uses Hugging Face LLM inference to parse natural language queries and validate results
      2. Fetches real-time place data from the TripAdvisor Content API
      3. Applies DistilBERT ML sentiment analysis to traveler reviews
      4. Runs a custom hidden gem detection scoring model
      5. Presents results in a categorised, ML-annotated card UI

FULL PIPELINE (executed per user query):
    ┌─────────────────────────────────────────────────────────────┐
    │  User types: "best street food in Lahore"                  │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 1: Hugging Face Intent Parsing (api_handler/huggingface.py) │
    │  → Extracts: query="street food stalls"                    │
    │              location="Lahore, Pakistan"                   │
    │  → Generates: user-facing narrative paragraph              │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 2: TripAdvisor Fetch (api_handler/tripadvisor.py)    │
    │  → Search API: returns up to 10 location_ids               │
    │  → Parallel detail fetch (ThreadPoolExecutor, 6 workers):  │
    │       /details → rating, description, cuisine, price       │
    │       /photos  → CDN photo URL                             │
    │       /reviews → up to 5 review texts for ML              │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 3: Hugging Face Location Validation                  │
    │  → Reads each result's address                             │
    │  → Removes results from wrong cities/countries            │
    │  → Fail-open: returns unfiltered if validation fails      │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 4: DistilBERT Sentiment Analysis (ml_model/sentiment)│
    │  → Processes review text through NLP model                 │
    │  → Outputs: sentiment_score, label, positive_pct          │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 5: Hidden Gem Detection (ml_model/gem_detector.py)   │
    │  → gem_score = 0.40×rating + 0.35×sentiment + 0.25×hidden │
    │  → is_gem = score ≥ 0.68 AND rating ≥ 4.0                 │
    │  → Sorts results: hidden gems first                        │
    └────────────────────────┬────────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 6: UI Render (frontend/components.py)                │
    │  → Category filter buttons                                 │
    │  → Themed cards with photos, reviews, ML badges            │
    │  → "Read more" expanders for long reviews                  │
    └─────────────────────────────────────────────────────────────┘

API KEYS:
    Loaded from Streamlit secrets or environment variables:
      HF_TOKEN              — from huggingface.co/settings/tokens
      HF_MODEL              — optional Hugging Face chat-completion model
      TRIPADVISOR_API_KEY   — from tripadvisor.com/developers
"""

import os
import sys
import tomllib
import importlib

# ── Python path fix ────────────────────────────────────────────────────────────
# Ensures all submodules (frontend/, api_handler/, ml_model/, utils/) are
# importable regardless of which directory Streamlit is launched from.
# This is necessary on Windows where the working directory may differ from
# the script's directory when using VS Code's Streamlit runner.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script.
st.set_page_config(
    page_title="Wandr — AI Tourist Guide",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── API Key management ─────────────────────────────────────────────────────────
def get_project_secret(name: str) -> str:
    secrets_path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return ""
    try:
        with open(secrets_path, "rb") as f:
            content = f.read().decode("utf-8-sig")
        return str(tomllib.loads(content).get(name, ""))
    except Exception:
        return ""


def get_secret(name: str) -> str:
    """Read API keys from Streamlit, env vars, or this project's secrets file."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, "") or get_project_secret(name)


if "hf_key" not in st.session_state:
    st.session_state["hf_key"] = get_secret("HF_TOKEN") or get_secret("HUGGINGFACE_HUB_TOKEN")
if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = get_secret("HF_MODEL") or "openai/gpt-oss-20b:fastest"
if "ta_key" not in st.session_state:
    st.session_state["ta_key"] = get_secret("TRIPADVISOR_API_KEY")

# Shorthand references used throughout this file
HF_KEY     = st.session_state["hf_key"]
HF_MODEL   = st.session_state["hf_model"]
TA_KEY     = st.session_state["ta_key"]

# ── Module imports ─────────────────────────────────────────────────────────────
# Deferred until after page config and key setup to avoid import errors
# from interfering with the Streamlit page config call.
import frontend.components as ui_components
ui_components = importlib.reload(ui_components)
render_css = ui_components.render_css
render_theme_toggle = ui_components.render_theme_toggle
render_hero = ui_components.render_hero
render_filters = ui_components.render_filters
render_cards = ui_components.render_cards
from api_handler.tripadvisor import fetch_supervised_spots
from api_handler.huggingface import (
    audit_sentiments,
    filter_by_requested_category,
    is_broad_scope,
    parse_intent,
    plan_search_queries,
    validate_locations,
)
from ml_model.sentiment import analyze_reviews
from ml_model.gem_detector import rank_places
from utils.helpers import esc

# ── CSS and Hero ───────────────────────────────────────────────────────────────
theme_param = st.query_params.get("theme")
if isinstance(theme_param, list):
    theme_param = theme_param[0] if theme_param else None

if theme_param in ("light", "dark"):
    st.session_state["theme_mode"] = theme_param
elif "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"

render_css(st.session_state["theme_mode"])
render_theme_toggle()
render_hero()

# ── Session state initialisation ──────────────────────────────────────────────
# These keys persist across reruns within the same browser session.
if "messages"    not in st.session_state: st.session_state.messages    = []
if "last_spots"  not in st.session_state: st.session_state.last_spots  = []
if "filter_cat"  not in st.session_state: st.session_state.filter_cat  = "All"

search_slot = st.container()

# ── Chat history display ───────────────────────────────────────────────────────
# Replays all previous messages so the conversation is visible after reruns.
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{esc(msg["content"])}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="assistant-label">✦ WANDR GUIDE</div>', unsafe_allow_html=True)
        if msg.get("narrative"):
            st.markdown(msg["narrative"])
        if msg.get("error"):
            st.markdown(
                f'<div class="swarn">⚠️ {esc(msg["error"])}</div>',
                unsafe_allow_html=True,
            )

# ── Category filters and card grid ────────────────────────────────────────────
# Shown persistently once spots have been loaded for the current session.
if st.session_state.last_spots:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    filtered = render_filters(st.session_state.last_spots)
    gem_count = sum(1 for s in filtered if s.get("gem", {}).get("is_gem"))
    st.markdown(
        f'<p class="spot-count">'
        f'Showing {len(filtered)} spots'
        f'{f" · 💎 {gem_count} hidden gems" if gem_count else ""}'
        f'</p>',
        unsafe_allow_html=True,
    )
    render_cards(filtered)

# ── Input bar ──────────────────────────────────────────────────────────────────
with search_slot:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 5, 1])

with c1:
    # New button: clears the current session for a fresh city search
    if st.button("🔄 New"):
        st.session_state.messages   = []
        st.session_state.last_spots = []
        st.session_state.filter_cat = "All"
        st.rerun()

with c2:
    user_input = st.text_input(
        " ",
        placeholder="🌍 Where's the adventure? e.g. 'street food in Karachi'",
        key="user_input",
        label_visibility="collapsed",
    )

with c3:
    send = st.button("GO! →")

# ── Handle user query submission ───────────────────────────────────────────────
if send and user_input.strip():

    # Validate keys before doing any API work
    if not HF_KEY:
        st.markdown('<div class="serr">🔑 Hugging Face token missing. Add HF_TOKEN to Streamlit secrets.</div>', unsafe_allow_html=True)
        st.stop()
    if not TA_KEY:
        st.markdown('<div class="serr">🔑 TripAdvisor key missing. Restart the app.</div>', unsafe_allow_html=True)
        st.stop()

    user_msg = user_input.strip()

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.markdown(f'<div class="user-bubble">{esc(user_msg)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="assistant-label">✦ WANDR GUIDE</div>', unsafe_allow_html=True)

    # ── STEP 1: Hugging Face intent parsing ────────────────────────────────────
    with st.spinner("🤖 Understanding your request..."):
        params, narrative, ai_err = parse_intent(st.session_state.messages, HF_KEY, HF_MODEL)

    if ai_err:
        st.markdown(f'<div class="serr">❌ {esc(ai_err)}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": "", "error": ai_err})
        st.stop()

    # Display the AI-generated destination narrative
    st.markdown(narrative)

    loc = (params or {}).get("location", "")
    if not loc:
        err = "Could not detect a city. Please mention a city name explicitly."
        st.markdown(f'<div class="swarn">🗺 {err}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": narrative, "error": err})
        st.stop()

    # ── STEP 2: Hugging Face supervised TripAdvisor search ─────────────────────
    with st.spinner(f"📍 Planning a broader search for {loc}..."):
        search_queries = plan_search_queries(params, HF_KEY, HF_MODEL)

    broad_scope = is_broad_scope(loc)
    max_results_per_query = 6 if broad_scope else 8
    max_candidates = 18 if broad_scope else 24

    with st.spinner(f"📍 Finding more possible gems in {loc}..."):
        spots, ta_err = fetch_supervised_spots(
            queries=search_queries,
            location=loc,
            api_key=TA_KEY,
            max_results_per_query=max_results_per_query,
            max_candidates=max_candidates,
        )

    if ta_err:
        st.markdown(f'<div class="swarn">⚠️ {esc(ta_err)}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": narrative, "error": ta_err})
        st.stop()

    if not spots:
        err = f"No results found for {loc}. Try a different query or city name."
        st.markdown(f'<div class="swarn">🔍 {err}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": narrative, "error": err})
        st.stop()

    # ── STEP 3: Hugging Face location validation ───────────────────────────────
    with st.spinner(f"✅ Verifying results are actually in {loc}..."):
        spots = validate_locations(loc, spots, HF_KEY, HF_MODEL)

    if not spots:
        err = f"No verified results found inside {loc}. Try a broader query."
        st.markdown(f'<div class="swarn">🗺 {err}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": narrative, "error": err})
        st.stop()

    with st.spinner("✅ Filtering results to match your exact request..."):
        spots = filter_by_requested_category(params, spots, HF_KEY, HF_MODEL)

    if not spots:
        err = "No results matched the requested category after strict filtering. Try a broader category."
        st.markdown(f'<div class="swarn">🔍 {err}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": narrative, "error": err})
        st.stop()

    # ── STEP 4: ML sentiment analysis ──────────────────────────────────────────
    with st.spinner("🧠 Analysing reviews with ML sentiment model..."):
        for spot in spots:
            # Pass raw review texts to DistilBERT; result stored in spot["sentiment"]
            spot["sentiment"] = analyze_reviews(spot.get("raw_reviews", []))

    with st.spinner("🧠 Auditing review sentiment with Hugging Face..."):
        spots = audit_sentiments(spots, HF_KEY, HF_MODEL)

    # ── STEP 5: Hidden gem detection and ranking ────────────────────────────────
    with st.spinner("💎 Detecting hidden gems..."):
        spots = rank_places(spots)

    # ── STEP 6: Store and render ────────────────────────────────────────────────
    st.session_state.last_spots = spots
    st.session_state.filter_cat = "All"

    gem_count = sum(1 for s in spots if s.get("gem", {}).get("is_gem"))
    st.markdown(
        f'<p class="spot-count">'
        f'Found {len(spots)} spots in {esc(loc)}'
        f'{f" · 💎 {gem_count} hidden gems detected" if gem_count else ""}'
        f'</p>',
        unsafe_allow_html=True,
    )

    render_filters(spots)   # render filter buttons (All / Restaurants / ...)
    render_cards(spots)     # render card grid + Read more expanders

    # Save assistant turn to conversation history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   narrative,
        "narrative": narrative,
        "error":     "",
        "location":  loc,
    })

# ── Empty state prompt chips ───────────────────────────────────────────────────
# Shown only when no conversation has started yet — gives users a quick start.
if not st.session_state.messages:
    st.markdown('<p class="chip-label">✦ start your adventure</p>', unsafe_allow_html=True)
    chips = [
        "🗼 Paris Night out",
        "🍜 Bangkok street food",
        "🏯 Kyoto hidden gems",
        "🌊 Best beaches in Bali",
        "🕌 Things to do in Karachi",
        "🌮 Istanbul food scene",
    ]
    cols = st.columns(3)
    for i, chip in enumerate(chips):
        with cols[i % 3]:
            if st.button(chip, key=f"chip_{i}"):
                # Simulate typing the chip text — triggers next rerun as a prefill
                st.session_state["prefill"] = chip
                st.rerun()

# Handle prefill from chip click (runs on the subsequent rerun)
if st.session_state.get("prefill"):
    pf = st.session_state.pop("prefill")
    st.session_state.messages.append({"role": "user", "content": pf})
    st.rerun()
