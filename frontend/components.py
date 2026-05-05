"""
frontend/components.py
======================
All Streamlit UI rendering lives here.

This module owns:
  - Global CSS (light-mode-first, high-contrast, readable in all modes)
  - Hero section
  - Category filter bar (with session state persistence)
  - Card builder (per-theme, word-truncated reviews, ML badges)
  - Card grid renderer

LIGHT MODE DESIGN DECISIONS:
  - Background: #f8f9fa (off-white, easier on eyes than pure #fff)
  - Card background: #ffffff with 1px border, subtle shadow
  - All text colours explicitly set with WCAG AA contrast ratios
  - Theme accent colours are dark enough on white (contrast ≥ 4.5:1)
  - No colour is defined only in a media query — default IS light mode

REVIEW TRUNCATION:
  - Reviews are truncated to 80 words in api_handler/tripadvisor.py
  - Cards show the truncated version by default
  - A "read more" expander (st.expander) is shown if was_truncated=True
  - This keeps all cards the same height in the grid

CATEGORY THEMES:
  Each TripAdvisor category maps to a visual theme (CSS class prefix "t-"):
    food    → orange  (warm, appetising)
    attr    → blue    (trustworthy, landmark-like)
    hotel   → purple  (luxury, hospitality)
    nature  → green   (organic, outdoors)
"""

import streamlit as st
from utils.helpers import esc


# ── Category filter configuration ─────────────────────────────────────────────
# Maps display label → internal theme key used in CSS and filtering logic
CATEGORIES = {
    "All":             None,
    "🍽 Restaurants":  "food",
    "🏛 Attractions":  "attr",
    "💎 Hidden Gems":  "gem",
    "🏨 Hotels":       "hotel",
    "🌿 Nature":       "nature",
}


def get_theme(category: str, subcategories: list, cuisine: str) -> tuple:
    """
    Map a TripAdvisor category to a visual theme tuple.

    Uses TripAdvisor's own category key as the primary signal, with
    cuisine strings and subcategory names as secondary signals for
    finer categorisation (e.g., a cafe inside "restaurants" category).

    Args:
        category:      TripAdvisor top-level key: "restaurants", "hotels",
                       "attractions", "geos"
        subcategories: List of subcategory name strings (e.g. ["Historic Site"])
        cuisine:       Comma-separated cuisine string (e.g. "Pizza, Italian")

    Returns:
        Tuple of (theme_key: str, emoji: str, display_label: str)
        theme_key is used as CSS class prefix (e.g. "food" → class "t-food")
    """
    cat  = (category or "").lower()
    subs = " ".join(s.lower() for s in (subcategories or []))
    cu   = (cuisine or "").lower()

    # Food: TripAdvisor restaurants category, or cuisine field mentions food types
    if cat == "restaurants" or any(
        w in cu for w in ["restaurant", "food", "cafe", "bar", "bistro", "grill", "pizza", "bakery", "dining"]
    ):
        return "food", "🍽️", "Food & Dining"

    # Hotels: direct category match
    if cat == "hotels":
        return "hotel", "🏨", "Hotel"

    # Nature: "geos" category or subcategory names mention outdoor features
    if cat == "geos" or any(
        w in subs for w in ["park", "beach", "nature", "garden", "lake", "mountain", "forest", "waterfall"]
    ):
        return "nature", "🌿", "Nature"

    # Default: attractions (museums, temples, markets, historical sites, etc.)
    return "attr", "🏛️", "Attraction"


def render_css() -> None:
    """
    Inject all global CSS styles into the Streamlit app.

    Design principles applied:
      1. Light-mode-first: all base colours are suitable for light backgrounds.
         No colour relies on dark mode to be readable.
      2. Explicit contrast: all text-on-background combinations meet WCAG AA
         (minimum 4.5:1 ratio for normal text, 3:1 for large text).
      3. Theme isolation: category theme CSS uses descendant selectors
         (.t-food .wcard-name) so theme styles never bleed between cards.
      4. No dark-mode-only rules: the app works perfectly without any
         prefers-color-scheme media query.
    """
    st.markdown("""
<style>
/* ── Fonts ───────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Nunito:wght@400;600;700;800&display=swap');

/* ── Global reset (light-mode-first) ─────────────────────────────────────── */
/* Overrides Streamlit's default which can be dark in some browser configs   */
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #f8f9fa !important;  /* Off-white — softer than pure white */
    color: #1a1a2e !important;             /* Near-black — high contrast on white */
}
.stApp { background-color: #f8f9fa !important; }

/* Native Streamlit text can inherit white from the active app theme.
   Keep Streamlit-generated markdown/widget text dark while custom UI classes
   below keep their own explicit colours. */
[data-testid="stMarkdownContainer"] > p,
[data-testid="stMarkdownContainer"] > ul,
[data-testid="stMarkdownContainer"] > ol,
[data-testid="stMarkdownContainer"] > div:not([class]),
[data-testid="stMarkdownContainer"] > h1,
[data-testid="stMarkdownContainer"] > h2,
[data-testid="stMarkdownContainer"] > h3,
[data-testid="stMarkdownContainer"] > h4,
[data-testid="stMarkdownContainer"] > h5,
[data-testid="stMarkdownContainer"] > h6,
[data-testid="stMarkdownContainer"] > blockquote,
[data-testid="stMarkdownContainer"] > pre,
[data-testid="stMarkdownContainer"] > code,
[data-testid="stText"],
[data-testid="stCaptionContainer"],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] > p {
    color: #1a1a2e !important;
}

[data-testid="stMarkdownContainer"] > p a,
[data-testid="stMarkdownContainer"] > ul a,
[data-testid="stMarkdownContainer"] > ol a {
    color: #1d4ed8 !important;
}

/* Hide Streamlit chrome (menu, footer, header) */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ──────────────────────────────────────────────────────────── */
/* Orange-to-yellow gradient works well on both light and dark backgrounds   */
.hero {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 40%, #ffcd3c 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(255, 107, 53, 0.25);
}
/* Decorative emoji watermark behind the title */
.hero::before {
    content: "✈ 🗺 🧭 🏔 🌊 🏛 🌴 🎡";
    position: absolute;
    top: 10px; left: 0; right: 0;
    font-size: 1.4rem;
    opacity: 0.15;
    letter-spacing: 12px;
    white-space: nowrap;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    color: #ffffff;                      /* White on orange — contrast > 4.5:1 */
    margin: 0;
    line-height: 1;
    letter-spacing: 4px;
    text-shadow: 3px 3px 0 rgba(0, 0, 0, 0.15);
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.95);   /* Near-white on orange — readable */
    margin: 0.4rem 0 0;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.hero-badges {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.25);
    color: #ffffff;
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
}

/* ── Chat bubbles ─────────────────────────────────────────────────────────── */
.user-bubble {
    background: #1a1a2e;                 /* Dark navy — high contrast for white text */
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 16px;
    margin: 8px 0;
    max-width: 68%;
    margin-left: auto;
    font-size: 0.92rem;
    font-weight: 600;
    box-shadow: 3px 3px 0 #ff6b35;
}
.assistant-label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #ff6b35;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 6px;
}

/* ── Cards grid ───────────────────────────────────────────────────────────── */
/* auto-fill with 260px min creates 4 columns on wide screens, 2 on tablet   */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 18px;
    margin: 12px 0 24px;
}

/* ── Base card ────────────────────────────────────────────────────────────── */
.wcard {
    background: #ffffff;                /* Pure white card on off-white page */
    border: 1.5px solid #e2e8f0;       /* Light grey border — visible but subtle */
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);   /* Soft shadow for depth */
    transition: transform 0.15s, box-shadow 0.15s;
    display: flex;
    flex-direction: column;
}
.wcard:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

/* ── Card photo ───────────────────────────────────────────────────────────── */
.wcard img {
    width: 100%;
    height: 170px;
    object-fit: cover;
    display: block;
}
/* Placeholder shown when photo URL is missing or fails to load */
.wcard-ph {
    width: 100%;
    height: 170px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
}

/* ── Coloured accent strip between photo and body ─────────────────────────── */
.wcard-strip { height: 4px; width: 100%; }

/* ── Card body ────────────────────────────────────────────────────────────── */
.wcard-body {
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

/* Category badge (small pill above the place name) */
.wcard-badge {
    display: inline-block;
    font-size: 0.60rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 20px;
    margin-bottom: 2px;
    width: fit-content;
}

/* Place name — large, bold, using display font */
.wcard-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.35rem;
    margin: 0;
    letter-spacing: 1.5px;
    line-height: 1.15;
}

/* Address line */
.wcard-addr {
    font-size: 0.74rem;
    color: #64748b;                     /* Slate — sufficient contrast on white */
    margin: 2px 0 0;
    line-height: 1.4;
}

/* Review quote block */
.wcard-review {
    font-size: 0.78rem;
    font-style: italic;
    line-height: 1.55;
    padding: 9px 11px;
    border-radius: 8px;
    margin: 8px 0 0;
    color: inherit;                     /* Inherits theme colour set per .t-* */
    border-left: 3px solid currentColor;
}

/* Review attribution line */
.wcard-reviewer {
    font-size: 0.67rem;
    font-style: normal;
    font-weight: 700;
    opacity: 0.65;
    margin-top: 3px;
}

/* Description text (shown when no review is available) */
.wcard-desc {
    font-size: 0.77rem;
    color: #475569;                     /* Slate-600 — good contrast on white */
    line-height: 1.5;
    margin: 7px 0 0;
}

/* Meta row (rating pill, price pill, TripAdvisor link) */
.wcard-meta {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 10px;
    align-items: center;
}

/* Generic pill style */
.wpill {
    border-radius: 20px;
    padding: 3px 9px;
    font-size: 0.69rem;
    font-weight: 700;
    border: 1px solid rgba(0, 0, 0, 0.1);
}
.wpill-price {
    background: #f1f5f9;               /* Light slate — neutral price indicator */
    color: #334155;
}
/* TripAdvisor link pill — dark background ensures contrast for white text */
.wpill-ta {
    background: #1a1a2e;
    color: #ffffff !important;
    border: none;
    font-weight: 700;
    font-size: 0.69rem;
    padding: 3px 9px;
    border-radius: 20px;
    text-decoration: none;
}
.wpill-ta:hover { background: #2d2d4e; }

/* ── ML insight badges ────────────────────────────────────────────────────── */
/* All badge text colours meet WCAG AA contrast on their background colours  */
.ml-row {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin-top: 5px;
}
.ml-pill {
    border-radius: 20px;
    padding: 2px 8px;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
/* Hidden gem badge: amber background, dark amber text */
.ml-gem { background: #fef3c7; color: #78350f; border: 1px solid #d97706; }
/* Positive sentiment: green background, dark green text */
.ml-pos { background: #dcfce7; color: #14532d; border: 1px solid #16a34a; }
/* Mixed sentiment: yellow background, dark text */
.ml-mix { background: #fef9c3; color: #713f12; border: 1px solid #ca8a04; }
/* Poor sentiment: red background, dark red text */
.ml-neg { background: #fee2e2; color: #7f1d1d; border: 1px solid #dc2626; }
/* Score badge: light grey */
.ml-score { background: #f1f5f9; color: #334155; border: 1px solid #94a3b8; }

/* ══ CATEGORY THEMES ══════════════════════════════════════════════════════════
   Each theme sets: border colour, strip gradient, badge colours, name colour,
   review block background, rating pill, photo placeholder background.
   All text colours are dark enough for WCAG AA on their respective backgrounds.
   ══════════════════════════════════════════════════════════════════════════ */

/* 🍽 Food — orange (warm, appetite-stimulating) */
.t-food { border-color: #fb923c; }
.t-food:hover { box-shadow: 0 8px 24px rgba(251, 146, 60, 0.3); }
.t-food .wcard-strip { background: linear-gradient(90deg, #ff6b35, #ffcd3c); }
.t-food .wcard-badge { background: #fff7ed; color: #9a3412; }          /* Orange-900 text on Orange-50 bg */
.t-food .wcard-name  { color: #9a3412; }
.t-food .wcard-review { background: #fff7ed; color: #7c2d12; border-color: #fb923c; }
.t-food .wpill-r { background: #ff6b35; color: #fff; border: none; }
.t-food .wcard-ph { background: linear-gradient(135deg, #ff6b35, #ffcd3c); }

/* 🏛 Attraction — blue (trustworthy, cultural, landmark) */
.t-attr { border-color: #60a5fa; }
.t-attr:hover { box-shadow: 0 8px 24px rgba(96, 165, 250, 0.3); }
.t-attr .wcard-strip { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.t-attr .wcard-badge { background: #eff6ff; color: #1e40af; }          /* Blue-800 text on Blue-50 bg */
.t-attr .wcard-name  { color: #1e3a8a; }
.t-attr .wcard-review { background: #eff6ff; color: #1e3a8a; border-color: #60a5fa; }
.t-attr .wpill-r { background: #1d4ed8; color: #fff; border: none; }
.t-attr .wcard-ph { background: linear-gradient(135deg, #1d4ed8, #60a5fa); }

/* 🏨 Hotel — purple (luxury, hospitality) */
.t-hotel { border-color: #a78bfa; }
.t-hotel:hover { box-shadow: 0 8px 24px rgba(167, 139, 250, 0.3); }
.t-hotel .wcard-strip { background: linear-gradient(90deg, #6d28d9, #a78bfa); }
.t-hotel .wcard-badge { background: #f5f3ff; color: #4c1d95; }         /* Purple-900 text on Purple-50 bg */
.t-hotel .wcard-name  { color: #4c1d95; }
.t-hotel .wcard-review { background: #f5f3ff; color: #4c1d95; border-color: #a78bfa; }
.t-hotel .wpill-r { background: #6d28d9; color: #fff; border: none; }
.t-hotel .wcard-ph { background: linear-gradient(135deg, #6d28d9, #a78bfa); }

/* 🌿 Nature — green (organic, outdoors, parks) */
.t-nature { border-color: #4ade80; }
.t-nature:hover { box-shadow: 0 8px 24px rgba(74, 222, 128, 0.3); }
.t-nature .wcard-strip { background: linear-gradient(90deg, #15803d, #4ade80); }
.t-nature .wcard-badge { background: #f0fdf4; color: #14532d; }        /* Green-900 text on Green-50 bg */
.t-nature .wcard-name  { color: #14532d; }
.t-nature .wcard-review { background: #f0fdf4; color: #14532d; border-color: #4ade80; }
.t-nature .wpill-r { background: #15803d; color: #fff; border: none; }
.t-nature .wcard-ph { background: linear-gradient(135deg, #15803d, #4ade80); }

/* ── Input elements ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;    /* Slate-300 — light but visible */
    border-radius: 12px !important;
    color: #1a1a2e !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;                 /* Slate-400 — subtle but readable */
    font-weight: 400 !important;
}

/* Buttons */
.stButton > button {
    background: #ff6b35 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    padding: 10px 18px !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.35) !important;
    transition: transform 0.1s, box-shadow 0.1s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(255, 107, 53, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 1px 4px rgba(255, 107, 53, 0.3) !important;
}

/* ── Alert / status banners ───────────────────────────────────────────────── */
.serr  { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; border-radius: 10px; padding: 12px 16px; font-weight: 700; margin: 8px 0; }
.swarn { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; border-radius: 10px; padding: 12px 16px; font-weight: 700; margin: 8px 0; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
.chip-label { text-align: center; color: #94a3b8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin: 1.5rem 0 0.75rem; }
.spot-count { font-size: 0.78rem; color: #64748b; margin: 4px 0 12px; font-weight: 600; }
.filter-label { font-size: 0.75rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


def render_hero() -> None:
    """Render the top hero banner with app title and feature badges."""
    st.markdown("""
<div class="hero">
  <h1 class="hero-title">WANDR</h1>
  <p class="hero-sub">🧭 AI-powered travel guide · Discover smarter · Anywhere on Earth</p>
  <div class="hero-badges">
    <span class="hero-badge">🏛 Landmarks</span>
    <span class="hero-badge">🍜 Food & Dining</span>
    <span class="hero-badge">💎 Hidden Gems</span>
    <span class="hero-badge">🌿 Nature</span>
    <span class="hero-badge">🤖 ML-Powered</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_filters(spots: list) -> list:
    """
    Render category filter buttons and return the filtered spot list.

    Filter state is stored in st.session_state["filter_cat"] so it persists
    across Streamlit reruns (each button click triggers a rerun).

    Filter logic:
      - "All"         → return all spots unfiltered
      - "Hidden Gems" → return only spots where gem["is_gem"] is True
      - Other cats    → match by theme key from get_theme()

    Args:
        spots: Full list of scored spot dicts.

    Returns:
        Filtered sub-list of spot dicts based on selected category.
    """
    if not spots:
        return spots

    st.markdown('<p class="filter-label">Filter by category</p>', unsafe_allow_html=True)
    cols = st.columns(len(CATEGORIES))
    selected = st.session_state.get("filter_cat", "All")

    # Render one button per category — clicking sets session state and reruns
    for i, (label, _) in enumerate(CATEGORIES.items()):
        with cols[i]:
            if st.button(label, key=f"filter_{label}"):
                st.session_state["filter_cat"] = label
                selected = label

    # Apply the active filter to the spot list
    if selected == "All":
        return spots

    theme_key = CATEGORIES[selected]

    # Hidden gem filter: use ML gem detection result
    if theme_key == "gem":
        return [s for s in spots if s.get("gem", {}).get("is_gem")]

    # Category filter: match by theme key from get_theme()
    return [
        s for s in spots
        if get_theme(s["category"], s.get("subcategories", []), s["cuisine"])[0] == theme_key
    ]


def _ml_badges(spot: dict) -> str:
    """
    Build the HTML for ML insight badge row on a card.

    Badges displayed:
      💎 Hidden Gem  — if gem_detector classified this place as a gem
      Sentiment      — Excellent/Good/Mixed/Poor with percentage
      Score          — numeric gem_score for transparency

    Args:
        spot: Spot dict containing "gem" and "sentiment" sub-dicts.

    Returns:
        HTML string for the badge row, or empty string if no badges apply.
    """
    badges = []
    gem  = spot.get("gem", {})
    sent = spot.get("sentiment", {})

    # Hidden gem badge — highest priority, shown first
    if gem.get("is_gem"):
        badges.append('<span class="ml-pill ml-gem">💎 Hidden Gem</span>')

    # Sentiment label badge with percentage
    label = sent.get("sentiment_label", "")
    pct   = sent.get("positive_pct", 0)
    if label == "Excellent":
        badges.append(f'<span class="ml-pill ml-pos">😊 {pct:.0f}% positive</span>')
    elif label == "Good":
        badges.append(f'<span class="ml-pill ml-pos">👍 {pct:.0f}% positive</span>')
    elif label == "Mixed":
        badges.append(f'<span class="ml-pill ml-mix">😐 Mixed reviews</span>')
    elif label == "Poor":
        badges.append(f'<span class="ml-pill ml-neg">⚠ Low sentiment</span>')

    # Numeric gem score for portfolio/demo transparency
    score = gem.get("gem_score")
    if score is not None:
        badges.append(f'<span class="ml-pill ml-score">Score {score:.2f}</span>')

    return '<div class="ml-row">' + "".join(badges) + "</div>" if badges else ""


def build_card(spot: dict) -> str:
    """
    Build the complete HTML for a single place card.

    Each card is built as a self-contained HTML string. All text content
    is passed through esc() before injection to prevent XSS from API data.
    URLs are pre-validated by safe_url() / safe_photo() in the API handler.

    Review display:
      - Shows the truncated display_review text (max 80 words from tripadvisor.py)
      - If was_truncated=True, the caller (render_cards) adds a Streamlit expander
        with the full text AFTER the card HTML (outside the HTML block)

    Args:
        spot: Fully populated spot dict from the complete pipeline:
              tripadvisor.py → validate_locations → sentiment.py → gem_detector.py

    Returns:
        HTML string for one card (no outer container — caller wraps in .cards-grid)
    """
    theme, emoji, label = get_theme(
        spot["category"], spot.get("subcategories", []), spot["cuisine"]
    )
    t = f"t-{theme}"  # CSS class prefix, e.g. "t-food"

    # ── Safe text fields ───────────────────────────────────────────────────────
    cuisine_lbl = esc(spot["cuisine"]) if spot["cuisine"] else label.upper()
    name        = esc(spot["name"])
    addr        = esc(spot["address"])
    desc        = esc(spot["description"])
    ta_url      = spot["ta_url"]    # pre-validated by safe_url() in api_handler
    photo       = spot["photo_url"] # pre-validated by safe_photo() in api_handler
    rating      = spot.get("rating")
    reviews     = spot.get("num_reviews", 0)
    price       = esc(spot.get("price_level", ""))

    # Display review (already word-truncated in fetch_spot_details)
    drv      = spot.get("display_review", {})
    rv_txt   = esc(drv.get("text", ""))
    rv_auth  = esc(drv.get("author", ""))
    rv_rat   = drv.get("rating")

    # ── Photo / placeholder ────────────────────────────────────────────────────
    if photo:
        img = (
            f'<img src="{photo}" alt="{name}" '
            f'style="width:100%;height:170px;object-fit:cover;display:block" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<div class="wcard-ph {t}" style="display:none">{emoji}</div>'
        )
    else:
        img = f'<div class="wcard-ph {t}">{emoji}</div>'

    # ── Pills ──────────────────────────────────────────────────────────────────
    r_pill  = f'<span class="wpill wpill-r">★ {rating} ({int(reviews):,})</span>' if rating else ""
    p_pill  = f'<span class="wpill wpill-price">{price}</span>' if price else ""
    ta_link = f'<a href="{ta_url}" target="_blank" class="wpill-ta">TripAdvisor ↗</a>'

    # ── Review or description block ────────────────────────────────────────────
    if rv_txt:
        stars = "★" * min(int(rv_rat), 5) if rv_rat else ""
        extra = (
            f'<div class="wcard-review">'
            f'"{rv_txt}"'
            f'<div class="wcard-reviewer">— {rv_auth} {stars}</div>'
            f'</div>'
        )
    elif desc:
        extra = f'<p class="wcard-desc">{desc}</p>'
    else:
        extra = ""

    ml_html = _ml_badges(spot)

    return (
        f'<div class="wcard {t}">'
        f'{img}'
        f'<div class="wcard-strip"></div>'
        f'<div class="wcard-body">'
        f'<span class="wcard-badge">{cuisine_lbl}</span>'
        f'<p class="wcard-name">{name}</p>'
        f'<p class="wcard-addr">📍 {addr}</p>'
        f'{extra}'
        f'{ml_html}'
        f'<div class="wcard-meta">{r_pill}{p_pill}{ta_link}</div>'
        f'</div>'
        f'</div>'
    )


def render_cards(spots: list) -> None:
    """
    Render the full card grid with optional "Read more" expanders.

    For spots with truncated reviews (was_truncated=True), a Streamlit
    st.expander is rendered BELOW the HTML card grid with the full review text.
    This approach is necessary because Streamlit expanders cannot be embedded
    inside raw HTML — they must be rendered as native Streamlit components.

    Args:
        spots: Filtered list of scored spot dicts to display.
    """
    if not spots:
        st.markdown(
            '<div class="swarn">No spots match this filter — try another category.</div>',
            unsafe_allow_html=True,
        )
        return

    # Render all cards as one HTML block (fast, single DOM update)
    html = '<div class="cards-grid">' + "".join(build_card(s) for s in spots) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Render "Read more" expanders for truncated reviews
    # These must be native Streamlit components — cannot be inside HTML block
    truncated_spots = [
        s for s in spots
        if s.get("display_review", {}).get("was_truncated")
    ]
    if truncated_spots:
        st.markdown(
            '<p style="font-size:0.75rem;color:#94a3b8;margin:4px 0 8px">'
            '📖 Full reviews available below</p>',
            unsafe_allow_html=True,
        )
        for spot in truncated_spots:
            drv = spot.get("display_review", {})
            full_text = drv.get("full_text", "")
            author    = drv.get("author", "Traveler")
            if full_text:
                with st.expander(f"📖 Full review — {esc(spot['name'])}"):
                    st.write(f'*"{full_text}"*')
                    st.caption(f"— {author}")
