import json
import os
from functools import lru_cache

import numpy as np

from .text_processing import PAD, UNK, encode, tokenize


@lru_cache(maxsize=4)
def _load(path: str, mtime: float):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def clear_cache() -> None:
    _load.cache_clear()


def generate_text(session, seed: str, n_tokens: int, temperature: float) -> str:
    """Generate n_tokens after seed. temperature=0 -> greedy argmax."""
    path = session["model_path"]
    model = _load(path, os.path.getmtime(path))
    vocab = json.loads(session["vocab_json"])
    ctx_len = session["ctx_len"]
    inverse = {i: w for w, i in vocab.items()}

    out = tokenize(seed)
    rng = np.random.default_rng()

    for _ in range(int(n_tokens)):
        ids = encode(out, vocab)[-ctx_len:]
        x = np.array([[PAD] * (ctx_len - len(ids)) + ids], dtype="int32")
        probs = model.predict(x, verbose=0)[0].astype("float64")
        probs[[PAD, UNK]] = 0                      # never emit pad / unknown
        if probs.sum() == 0:
            break
        if temperature <= 0:
            nxt = int(np.argmax(probs))
        else:
            probs = np.exp(np.log(probs + 1e-9) / temperature)
            probs /= probs.sum()
            nxt = int(rng.choice(len(probs), p=probs))
        out.append(inverse.get(nxt, ""))
    return " ".join(out)
