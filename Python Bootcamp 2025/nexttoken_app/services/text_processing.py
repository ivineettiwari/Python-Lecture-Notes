"""Tokenizer, vocabulary and training-set construction."""
import re
from collections import Counter

import numpy as np

import config

PAD, UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def build_vocab(text: str) -> dict:
    counts = Counter(tokenize(text))
    words = [w for w, _ in counts.most_common(config.MAX_VOCAB - 2)]
    return {w: i + 2 for i, w in enumerate(words)}   # 0 = pad, 1 = unk


def encode(tokens, vocab: dict):
    return [vocab.get(t, UNK) for t in tokens]


def make_dataset(text: str, vocab: dict, ctx_len: int):
    """Sliding context windows per sentence -> (X, y) with sparse integer labels."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    X, y = [], []
    for sentence in sentences:
        ids = encode(tokenize(sentence), vocab)
        for i in range(1, len(ids)):
            ctx = ids[max(0, i - ctx_len):i]
            X.append([PAD] * (ctx_len - len(ctx)) + ctx)   # pre-padding
            y.append(ids[i])
    X = np.array(X, dtype="int32")
    y = np.array(y, dtype="int32")
    if len(X) > config.MAX_SAMPLES:
        idx = np.random.RandomState(42).choice(len(X), config.MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
    return X, y
