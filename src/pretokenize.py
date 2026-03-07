# pretokenize.py
from __future__ import annotations

import re
from typing import List

# Fixed rule from the lab sheet
PRETOKENIZE_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]")


def pretokenize(text: str) -> List[str]:
    """
    Deterministic pre-tokenization:
      - lowercase
      - regex findall with PRETOKENIZE_PATTERN
    """
    text = text.lower()
    return PRETOKENIZE_PATTERN.findall(text)
