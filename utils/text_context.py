"""Small text-matching helpers shared by the API and the tests.

Kept free of heavy imports (no torch / fastapi) so it can be unit-tested
anywhere.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional, Pattern, Tuple


@lru_cache(maxsize=16)
def _compile_keywords(keywords: Tuple[str, ...]) -> Optional[Pattern[str]]:
    """Compile keywords into one whole-word, case-insensitive regex.

    A keyword only matches as a standalone word (an optional plural ``s`` /
    ``es`` is allowed), never as a fragment inside a longer word.  Plain
    substring matching used to flag "cont*acting*", "f*actor*y",
    "de*script*ion" and "*prop*erty" as acting / actor / script / prop context.
    """
    cleaned = [kw.strip() for kw in keywords if kw and kw.strip()]
    if not cleaned:
        return None

    # Longest first so multi-word phrases win over their own prefixes.
    cleaned.sort(key=len, reverse=True)
    alternation = "|".join(re.escape(kw).replace(r"\ ", r"\s+") for kw in cleaned)
    return re.compile(
        rf"(?<![a-z0-9])(?:{alternation})(?:s|es)?(?![a-z0-9])",
        re.IGNORECASE,
    )


def contains_keyword(text: Optional[str], keywords: Iterable[str]) -> bool:
    """Return True if ``text`` contains any keyword as a whole word/phrase."""
    if not text:
        return False
    pattern = _compile_keywords(tuple(keywords))
    return bool(pattern and pattern.search(text))
