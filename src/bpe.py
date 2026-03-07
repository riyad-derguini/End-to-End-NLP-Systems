# bpe.py
from __future__ import annotations

from typing import Dict, List, Tuple

from constants import EOW_TOKEN


Pair = Tuple[str, str]
Merges = List[Pair]
WordFreq = Dict[str, int]


def build_word_freq(pretokenized_sentences: List[List[str]]) -> WordFreq:
    """
    Count word (pre-token) frequencies from pretokenized training data.
    """
    freq: WordFreq = {}
    for sent in pretokenized_sentences:
        for w in sent:
            freq[w] = freq.get(w, 0) + 1
    return freq


def init_representation(word: str) -> List[str]:
    """
    rho_0(word): list of characters + EOW marker.
    Example: "good" -> ["g","o","o","d","</w>"]
    """
    # TODO
    raise NotImplementedError


def merge_pair_in_sequence(seq: List[str], a: str, b: str, new_sym: str) -> List[str]:
    """
    Replace every non-overlapping occurrence of adjacent pair (a,b) by new_sym,
    scanning left-to-right.

    Example:
      seq = ["a","b","a","b"], merge (a,b)->"ab" => ["ab","ab"]
    """
    # TODO
    raise NotImplementedError


def count_pairs(word_freq: WordFreq, reps: Dict[str, List[str]]) -> Dict[Pair, int]:
    """
    Compute weighted adjacent pair counts:
      count(a,b) = sum_w freq(w) * (# occurrences of (a,b) in reps[w])
    """
    # TODO
    raise NotImplementedError


def choose_best_pair(pair_counts: Dict[Pair, int]) -> Pair:
    """
    Deterministic choice:
      1) maximum count
      2) if tied, prefer pairs that do NOT contain EOW_TOKEN
      3) if still tied, lexicographically smallest pair
    """
    # TODO
    raise NotImplementedError


def train_bpe(word_freq: WordFreq, merges: int) -> Merges:
    """
    Train word-level BPE and return the merge list [(a1,b1),...,(aM,bM)].

    Notes:
      - Represent each word as chars + </w>
      - Each merge creates new_sym = a+b (string concatenation)
      - Apply merge to ALL word representations
    """
    # TODO
    raise NotImplementedError


def encode_word(word: str, merges_list: Merges) -> List[str]:
    """
    Encode a single pre-token word using learned merges.
    Must remove the standalone </w> in the final output.
    """
    # TODO
    raise NotImplementedError


def encode_sentence(words: List[str], merges_list: Merges) -> List[str]:
    """
    Encode a sentence (list of pre-token words) as a flat list of BPE tokens.
    """
    # TODO
    raise NotImplementedError
