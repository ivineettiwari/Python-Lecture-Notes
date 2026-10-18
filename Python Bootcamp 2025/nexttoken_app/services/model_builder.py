def build_model(model_type: str, vocab_size: int, ctx_len: int):
    from tensorflow.keras import layers, models

    rnn_layers = {"SimpleRNN": layers.SimpleRNN, "GRU": layers.GRU, "LSTM": layers.LSTM}
    if model_type not in rnn_layers:
        raise ValueError(f"Unknown model type: {model_type}")

    model = models.Sequential([
        layers.Input(shape=(ctx_len,)),
        layers.Embedding(vocab_size, 64),
        rnn_layers[model_type](128),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
