# main.py
from __future__ import annotations

import argparse
from typing import List

from data import load_tsv
from pretokenize import pretokenize
from bpe import build_word_freq, train_bpe, encode_sentence
from vectorize import build_vocab, vectorize_corpus, Vector
from models import MajorityBaseline, LexiconHeuristic
from metrics import accuracy, macro_f1, confusion_matrix


def _pretokenize_corpus(texts: List[str]) -> List[List[str]]:
    return [pretokenize(t) for t in texts]


def _bpe_encode_corpus(pre_sents: List[List[str]], merges_list):
    return [encode_sentence(words, merges_list) for words in pre_sents]


def _print_metrics(split_name: str, y_true: List[int], y_pred: List[int]) -> None:
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    print(f"{split_name} accuracy: {accuracy(y_true, y_pred):.6f}")
    print(f"{split_name} macro_f1: {macro_f1(y_true, y_pred):.6f}")
    print(f"{split_name} confusion: TP={tp} FP={fp} TN={tn} FN={fn}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to data directory containing train/dev/test.tsv")
    ap.add_argument("--model", type=str, required=True, choices=["majority", "lexicon"])
    ap.add_argument("--merges", type=int, default=200)
    ap.add_argument("--vocab_size", type=int, default=5000)
    args = ap.parse_args()

    train_texts, train_y = load_tsv(f"{args.data}/train.tsv")
    dev_texts, dev_y = load_tsv(f"{args.data}/dev.tsv")
    test_texts, test_y = load_tsv(f"{args.data}/test.tsv")

    # 1) Pre-tokenize
    train_words = _pretokenize_corpus(train_texts)
    dev_words = _pretokenize_corpus(dev_texts)
    test_words = _pretokenize_corpus(test_texts)

    # 2) Train BPE on TRAIN only
    word_freq = build_word_freq(train_words)
    merges_list = train_bpe(word_freq, merges=args.merges)

    # 3) Encode with BPE (train/dev/test)
    train_bpe_toks = _bpe_encode_corpus(train_words, merges_list)
    dev_bpe_toks = _bpe_encode_corpus(dev_words, merges_list)
    test_bpe_toks = _bpe_encode_corpus(test_words, merges_list)

    # 4) Build vocab on TRAIN only, vectorize all splits
    vocab = build_vocab(train_bpe_toks, vocab_size=args.vocab_size)
    X_train = vectorize_corpus(train_bpe_toks, vocab)
    X_dev = vectorize_corpus(dev_bpe_toks, vocab)
    X_test = vectorize_corpus(test_bpe_toks, vocab)

    # 5) Train + predict
    if args.model == "majority":
        model = MajorityBaseline()
        model.fit(train_y)
    else:
        model = LexiconHeuristic(pos_words=["great", "good"], neg_words=["bad", "boring"])
        model.fit(merges_list, vocab)

    dev_pred = model.predict(X_dev)
    test_pred = model.predict(X_test)

    # 6) Print required outputs
    _print_metrics("DEV ", dev_y, dev_pred)
    _print_metrics("TEST", test_y, test_pred)


if __name__ == "__main__":
    main()
