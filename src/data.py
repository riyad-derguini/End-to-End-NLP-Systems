# data.py
from __future__ import annotations

from typing import List, Tuple


def load_tsv(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a TSV file with format: label<TAB>text
    Returns (texts, labels).
    """
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected 2 columns (label<TAB>text)")
            y_str, text = parts
            y = int(y_str)
            if y not in (0, 1):
                raise ValueError(f"{path}:{line_no}: label must be 0 or 1, got {y}")
            texts.append(text)
            labels.append(y)
    return texts, labels
