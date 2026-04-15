"""
utils/helpers.py
================
Utility functions shared across all modules.

Purpose:
    Centralises text sanitization and URL validation to prevent HTML injection
    attacks from untrusted API responses (TripAdvisor review text, descriptions,
    and URLs sometimes contain embedded HTML or partial markup).

Why this matters:
    Streamlit renders markdown with unsafe_allow_html=True, so any raw HTML from
    an API response injected into a card string will execute in the browser.
    Every piece of API text MUST pass through clean() + esc() before rendering.
"""

import re
import html as htmllib


def clean(text: str) -> str:
    """
    Strip all HTML tags from a string and decode HTML entities.

    Example:
        clean("<b>Café &amp; Restaurant</b>")  → "Café & Restaurant"

    Args:
        text: Raw string possibly containing HTML tags or entities.

    Returns:
        Plain text with tags removed and entities decoded.
    """
    if not text:
        return ""
    # Remove anything that looks like an HTML tag (e.g. <br>, <div class="...">)
    text = re.sub(r"<[^>]+>", "", str(text))
    # Decode HTML entities (&amp; → &, &quot; → ", etc.)
    text = htmllib.unescape(text)
    return text.strip()


def esc(text: str) -> str:
    """
    Clean a string then HTML-escape it for safe injection into Streamlit markdown.

    Use this for ALL user-visible text injected into f-string HTML blocks.
    Calling clean() first strips tags; htmllib.escape() then prevents the
    cleaned text from being interpreted as HTML in the output.

    Args:
        text: Any string from an API response or user input.

    Returns:
        A safe string ready for insertion into HTML templates.
    """
    return htmllib.escape(clean(text))


def safe_url(url: str) -> str:
    """
    Validate a URL from the TripAdvisor API before using it in an href.

    TripAdvisor's web_url field occasionally contains partial HTML or
    malformed strings. We reject anything that:
    - Doesn't start with http (could be javascript:// or data://)
    - Contains < or > characters (embedded HTML)

    Args:
        url: Raw URL string from the API.

    Returns:
        The original URL if valid, otherwise the TripAdvisor homepage.
    """
    u = clean(url)
    if u.startswith("http") and "<" not in u and ">" not in u:
        return u
    return "https://www.tripadvisor.com"


def safe_photo(url: str) -> str:
    """
    Validate a photo URL from the TripAdvisor photos endpoint.

    Photo URLs from TripAdvisor CDN are always short HTTPS URLs.
    Reject anything longer than 500 chars (likely injected HTML content)
    or containing angle brackets.

    Args:
        url: Raw photo URL from the API images dict.

    Returns:
        Valid CDN URL string, or empty string if invalid.
    """
    u = clean(url)
    if u.startswith("http") and "<" not in u and ">" not in u and len(u) < 500:
        return u
    return ""


def truncate_words(text: str, max_words: int = 80) -> tuple[str, bool]:
    """
    Truncate text to a maximum word count.

    Used to keep review snippets concise in cards. Returns both the
    truncated string and a boolean indicating whether truncation occurred
    (so the UI can show a "Read more" toggle).

    Args:
        text:      Input text (already cleaned).
        max_words: Maximum number of words to keep (default 80).

    Returns:
        Tuple of (truncated_text, was_truncated).
        If was_truncated is True, UI should offer a "Read more" option.
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + "…", True
