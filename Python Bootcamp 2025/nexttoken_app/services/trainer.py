from typing import Callable, Optional

import config
from database import repository
from . import generator
from .model_builder import build_model
from .text_processing import build_vocab, make_dataset


def train_session(
    session_id: int,
    model_type: str,
    epochs: int,
    ctx_len: int,
    on_epoch: Optional[Callable[[int, int, dict], None]] = None,
) -> int:
    """Train a model on the session's document. Returns number of training samples."""
    import tensorflow as tf

    session = repository.get_session(session_id)
    vocab = build_vocab(session["text"])
    X, y = make_dataset(session["text"], vocab, ctx_len)
    if len(X) < 20:
        raise ValueError("Not enough text in this document to train a model.")

    model = build_model(model_type, len(vocab) + 2, ctx_len)

    class _Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch:
                on_epoch(epoch + 1, epochs, logs or {})

    history = model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.1,
                        verbose=0, callbacks=[_Progress()])

    path = config.MODEL_DIR / f"session_{session_id}.keras"
    model.save(path)
    repository.save_training(
        session_id, vocab, ctx_len, model_type, epochs,
        float(history.history["loss"][-1]), float(history.history["accuracy"][-1]), path,
    )
    generator.clear_cache()
    return len(X)
