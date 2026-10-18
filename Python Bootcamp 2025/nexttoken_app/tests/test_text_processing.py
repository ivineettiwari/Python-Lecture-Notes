from services.text_processing import build_vocab, encode, make_dataset, tokenize

TEXT = "machine learning is fun. deep learning is fun too.\nlearning never stops here."


def test_tokenize():
    assert tokenize("Hello, World 42!") == ["hello", "world", "42"]


def test_vocab_reserves_pad_unk():
    vocab = build_vocab(TEXT)
    assert min(vocab.values()) == 2
    assert encode(["zzz"], vocab) == [1]


def test_dataset_shapes():
    vocab = build_vocab(TEXT)
    X, y = make_dataset(TEXT, vocab, ctx_len=5)
    assert X.shape[1] == 5 and len(X) == len(y) > 0
