"""
ml_model/gem_detector.py
========================
Hidden Gem Detection — the primary original ML contribution of this project.

WHAT IS A "HIDDEN GEM"?
    A hidden gem is a place that is:
      (a) High quality: strong star rating AND positive ML sentiment score
      (b) Underrated:   relatively few reviews compared to well-known places
      (c) Authentic:    not dominated by tourist-trap hype

    The intuition: a place with 4.6 stars and 45 reviews is more "hidden"
    than a place with 4.6 stars and 45,000 reviews — equally good, but
    undiscovered. Our model detects and surfaces these places.

THE SCORING MODEL:
    This is a custom unsupervised weighted composite scoring model.
    "Unsupervised" means it requires no labelled training data (no dataset
    of pre-labelled "gems" and "non-gems"). The score is computed analytically
    from three normalized components:

        gem_score = W_RATING × norm_rating
                  + W_SENTIMENT × sentiment_score
                  + W_HIDDEN × (1 − log_norm_reviews)

    Component breakdown:

    1. norm_rating (weight 0.40):
       The TripAdvisor star rating (0–5) normalised to [0, 1].
       This is the largest weight because quality is the primary criterion.
       A place must be genuinely good to be a gem.

    2. sentiment_score (weight 0.35):
       The ML sentiment score from sentiment.py, already in [0, 1].
       This augments the star rating with language-level signal from reviews.
       It catches places where the rating is high but reviewers express
       dissatisfaction in their text (or vice versa).

    3. (1 − log_norm_reviews) "hidden score" (weight 0.25):
       Measures how undiscovered a place is.
       We use LOG normalization (log1p) rather than linear normalization because
       review counts span several orders of magnitude (10 to 100,000+).
       Linear normalization would make a 1,000-review place look "almost as
       hidden as" a 100,000-review place, which isn't meaningful.
       With log scale:
         10 reviews   → very hidden (score ≈ 0.87)
         100 reviews  → moderately hidden (score ≈ 0.70)
         1,000 reviews → less hidden (score ≈ 0.53)
         10,000 reviews → popular (score ≈ 0.35)

CLASSIFICATION THRESHOLD:
    is_gem = True if gem_score ≥ 0.68 AND rating ≥ 4.0
    The rating floor (≥ 4.0) prevents a highly obscure but low-quality place
    from being classified as a gem purely due to its low review count.

WEIGHTS JUSTIFICATION:
    W_RATING=0.40, W_SENTIMENT=0.35, W_HIDDEN=0.25
    Quality dominates (0.75 combined) because a hidden gem must first be
    genuinely good. Hiddenness is secondary (0.25) — it distinguishes between
    two equally good places, not between a good and bad place.

RANKING:
    All spots in a result set are scored and sorted by gem_score descending.
    This means hidden gems appear first in the UI, followed by well-known
    good places, followed by lower-scoring spots.
"""

from __future__ import annotations
import math
import numpy as np

# ── Scoring weights — must sum to 1.0 ─────────────────────────────────────────
W_RATING    = 0.40  # Star rating quality component
W_SENTIMENT = 0.35  # NLP sentiment score component
W_HIDDEN    = 0.25  # Inverse log-popularity component

# Gem classification threshold (0.0–1.0 scale)
GEM_THRESHOLD = 0.68

# Minimum star rating to qualify as a gem (prevents bad-but-obscure places)
MIN_RATING_FOR_GEM = 4.0


def score_place(
    rating: float | None,
    num_reviews: int,
    sentiment_score: float,
    max_reviews_in_set: int = 10_000,
) -> dict:
    """
    Compute the hidden gem score for a single place.

    All three input components are normalized to [0, 1] before combining.
    The result is clipped to [0, 1] using numpy to handle any floating-point
    edge cases.

    Args:
        rating:             TripAdvisor star rating (0–5), or None if unavailable.
        num_reviews:        Total review count on TripAdvisor for this place.
        sentiment_score:    ML sentiment output from sentiment.py (0.0–1.0).
        max_reviews_in_set: Maximum review count seen in the current result set,
                            used as the normalization ceiling. Defaults to 10,000
                            if not provided.

    Returns:
        Dict with:
          gem_score   (float 0–1): composite quality + hiddenness score
          is_gem      (bool):      True if score ≥ GEM_THRESHOLD and rating ≥ 4.0
          confidence  (str):       "High" / "Medium" / "Low"
          components  (dict):      individual normalized component scores
    """

    # ── Component 1: Normalise star rating to [0, 1] ──────────────────────────
    # If rating is None (not yet rated), assume neutral midpoint (0.5)
    norm_rating = (float(rating) / 5.0) if rating else 0.5

    # ── Component 2: Log-normalise review count ───────────────────────────────
    # log1p(x) = log(1 + x), which handles x=0 gracefully (log(1) = 0)
    # Divide by log1p(max) to get a value in [0, 1]
    safe_max = max(max_reviews_in_set, 1)  # prevent division by zero
    norm_reviews = math.log1p(num_reviews) / math.log1p(safe_max)
    hidden_score = 1.0 - norm_reviews      # invert: fewer reviews = more hidden

    # ── Composite gem score (weighted sum) ────────────────────────────────────
    gem_score = (
        W_RATING    * norm_rating +
        W_SENTIMENT * float(sentiment_score) +
        W_HIDDEN    * hidden_score
    )
    # Clip to valid range to handle any floating-point overflow
    gem_score = round(float(np.clip(gem_score, 0.0, 1.0)), 3)

    # ── Classification ────────────────────────────────────────────────────────
    is_gem = (gem_score >= GEM_THRESHOLD) and ((rating or 0) >= MIN_RATING_FOR_GEM)

    # Confidence label for UI display
    if gem_score >= 0.80:
        confidence = "High"
    elif gem_score >= GEM_THRESHOLD:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "gem_score":  gem_score,
        "is_gem":     is_gem,
        "confidence": confidence,
        "components": {
            "rating_score":    round(norm_rating, 3),
            "sentiment_score": round(float(sentiment_score), 3),
            "hidden_score":    round(hidden_score, 3),
        },
    }


def rank_places(spots: list[dict]) -> list[dict]:
    """
    Score all spots in a result set and sort them by gem_score descending.

    This function:
      1. Finds the max review count in the set (for log normalization context)
      2. Calls score_place() for each spot using its ML sentiment output
      3. Attaches the gem scoring result to each spot dict as spot["gem"]
      4. Sorts the list so hidden gems appear first in the UI

    Each spot dict must already contain a "sentiment" key (from sentiment.py)
    with at minimum a "sentiment_score" float.

    Args:
        spots: List of spot dicts from api_handler/tripadvisor.py,
               each already enriched with spot["sentiment"] from sentiment.py.

    Returns:
        Same list, with spot["gem"] added to each item, sorted by gem_score.
    """
    if not spots:
        return spots

    # Use the maximum review count in this result set as the normalization ceiling
    # This makes scoring relative within the current result set, not global
    max_reviews = max((s.get("num_reviews", 0) for s in spots), default=1)

    for spot in spots:
        sentiment = spot.get("sentiment", {})
        gem_info = score_place(
            rating=spot.get("rating"),
            num_reviews=spot.get("num_reviews", 0),
            sentiment_score=sentiment.get("sentiment_score", 0.5),
            max_reviews_in_set=max_reviews,
        )
        spot["gem"] = gem_info

    # Sort descending by gem_score — hidden gems float to the top
    spots.sort(key=lambda s: s["gem"]["gem_score"], reverse=True)
    return spots
