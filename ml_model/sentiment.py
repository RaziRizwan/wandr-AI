"""
ml_model/sentiment.py
=====================
Sentiment analysis on TripAdvisor review text using a pretrained NLP model.

WHAT THIS MODULE DOES:
    Converts free-form traveler review text into a quantitative sentiment
    signal (score 0.0–1.0) that is used by gem_detector.py to rank places
    beyond their raw star ratings.

WHY SENTIMENT ANALYSIS (not just star ratings)?
    Star ratings are noisy — a 4-star review might say "beautiful location but
    terrible service and we won't return." The star doesn't capture the nuance.
    Sentiment analysis reads the actual text and classifies the emotional tone,
    giving a more reliable quality signal.

    Example where this matters:
      Place A: 4.2 stars, 50 reviews, 95% positive sentiment → genuinely great
      Place B: 4.2 stars, 50 reviews, 55% positive sentiment → inflated rating

WHICH MODEL IS USED:
    Model: distilbert-base-uncased-finetuned-sst-2-english
    Source: HuggingFace Transformers (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

    WHY THIS MODEL:
      - DistilBERT is a compressed (distilled) version of BERT, retaining 97%
        of BERT's performance at 60% the size and 2x the speed.
      - Fine-tuned on SST-2 (Stanford Sentiment Treebank), a benchmark dataset
        of movie/restaurant review sentences — directly applicable to travel reviews.
      - Binary classification: POSITIVE or NEGATIVE, each with a confidence score.
      - Runs efficiently on CPU — no GPU required, making it suitable for a
        Streamlit deployment on a standard laptop.
      - Downloaded once and cached locally by HuggingFace's transformers library.

HOW IT PROCESSES REVIEW DATA:
    Input:  List of review dicts → [{"text": str, "rating": int}, ...]
    Process:
      1. Extract text from each review (up to 512 tokens per review)
      2. Pass all texts through the DistilBERT pipeline in one batch
      3. Each result is {"label": "POSITIVE"|"NEGATIVE", "score": float}
      4. Convert to uniform 0–1 scale:
           POSITIVE with score 0.92 → sentiment = 0.92
           NEGATIVE with score 0.88 → sentiment = 1 - 0.88 = 0.12
      5. Average across all reviews → single sentiment_score for the place

HOW SENTIMENT INFLUENCES RANKINGS:
    sentiment_score feeds into gem_detector.py's composite scoring formula:
      gem_score = 0.40 × rating + 0.35 × sentiment_score + 0.25 × hidden_score
    A high sentiment_score boosts a place's gem_score even if its star rating
    is modest (e.g., a 4.1 ★ place with 90% positive sentiment outranks a
    4.3 ★ place with 55% positive sentiment in the hidden gem ranking).

LAZY LOADING:
    The DistilBERT model (~250MB) is loaded only on the first call to
    analyze_reviews(), not at app startup. This prevents slow startup times.
    The model is cached in the module-level _pipeline variable thereafter.
"""

from __future__ import annotations
import numpy as np

# Module-level cache: the pipeline is loaded once and reused across all calls
_pipeline = None


def _get_pipeline():
    """
    Lazily load and cache the HuggingFace sentiment pipeline.

    First call: downloads DistilBERT model (~250MB) and loads into memory.
    Subsequent calls: returns the cached pipeline instance immediately.

    Returns:
        Initialized HuggingFace pipeline object ready for inference.
    """
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,    # Silently truncate reviews > 512 tokens
            max_length=512,     # DistilBERT's maximum input length
        )
    return _pipeline


def analyze_reviews(reviews: list[dict]) -> dict:
    """
    Run sentiment analysis on a list of review dicts.

    Each review dict must have at minimum a "text" key.
    The "rating" key (1–5 star) is not used in ML inference but is available
    for future correlation analysis.

    ML Pipeline per call:
        reviews → extract texts → DistilBERT batch inference →
        convert to 0-1 scale → aggregate stats → return summary dict

    Args:
        reviews: List of dicts with keys: "text" (str), "rating" (int|None),
                 "author" (str). From api_handler/tripadvisor.py.

    Returns:
        Dict with these keys:
          sentiment_score    (float 0.0–1.0): mean positivity across all reviews
          sentiment_label    (str): "Excellent" / "Good" / "Mixed" / "Poor"
          positive_pct       (float): percentage of reviews classified positive
          review_count_analyzed (int): how many reviews were actually processed
          top_positive       (str): snippet from the most positive review
          top_concern        (str): snippet from the most negative review (if any)
    """
    if not reviews:
        return _empty_sentiment()

    # Extract and truncate review texts (DistilBERT max input = 512 tokens)
    texts = [r["text"][:512] for r in reviews if r.get("text")]
    if not texts:
        return _empty_sentiment()

    try:
        pipe = _get_pipeline()
        # Batch inference — all reviews processed in a single forward pass
        results = pipe(texts)
    except Exception:
        # If model inference fails for any reason, return neutral defaults
        return _empty_sentiment()

    # ── Convert model outputs to uniform 0–1 sentiment scores ─────────────────
    # DistilBERT output: {"label": "POSITIVE"|"NEGATIVE", "score": confidence}
    # We want: 1.0 = very positive, 0.0 = very negative
    scores = []
    for res in results:
        if res["label"] == "POSITIVE":
            scores.append(res["score"])          # e.g. POSITIVE 0.97 → 0.97
        else:
            scores.append(1.0 - res["score"])    # e.g. NEGATIVE 0.91 → 0.09

    # ── Aggregate statistics across all reviews ────────────────────────────────
    sentiment_score = float(np.mean(scores))
    positive_pct    = float(np.mean([1 if s > 0.5 else 0 for s in scores])) * 100

    # ── Map score to human-readable label ─────────────────────────────────────
    if sentiment_score >= 0.80:
        label = "Excellent"   # > 80% mean positivity
    elif sentiment_score >= 0.65:
        label = "Good"        # 65–80%
    elif sentiment_score >= 0.45:
        label = "Mixed"       # 45–65%
    else:
        label = "Poor"        # < 45%

    # ── Extract representative snippets ───────────────────────────────────────
    # Sort reviews by sentiment score to find most positive and most concerning
    indexed = sorted(zip(scores, texts), key=lambda x: x[0], reverse=True)
    top_positive = _snippet(indexed[0][1]) if indexed else ""
    # Only report a concern if the worst review is genuinely negative (< 0.5)
    top_concern = (
        _snippet(indexed[-1][1])
        if len(indexed) > 1 and indexed[-1][0] < 0.5
        else ""
    )

    return {
        "sentiment_score":       round(sentiment_score, 3),
        "sentiment_label":       label,
        "positive_pct":          round(positive_pct, 1),
        "review_count_analyzed": len(texts),
        "top_positive":          top_positive,
        "top_concern":           top_concern,
    }


def _snippet(text: str, limit: int = 120) -> str:
    """
    Return a short readable snippet from a review text.

    Used for the top_positive and top_concern fields in the returned dict.
    These snippets are surfaced in the UI to explain the ML assessment.

    Args:
        text:  Full review text (already cleaned).
        limit: Character limit for the snippet.

    Returns:
        Truncated text with ellipsis if over limit.
    """
    text = text.strip()
    return text[:limit] + "…" if len(text) > limit else text


def _empty_sentiment() -> dict:
    """
    Return a neutral/unknown sentiment dict when analysis cannot be performed.

    Used when: no reviews exist, all reviews have empty text, or model fails.
    The 0.5 sentiment_score is intentionally neutral — it does not boost
    or penalise the gem_score for unanalyzed places.
    """
    return {
        "sentiment_score":       0.5,
        "sentiment_label":       "Unknown",
        "positive_pct":          0.0,
        "review_count_analyzed": 0,
        "top_positive":          "",
        "top_concern":           "",
    }
