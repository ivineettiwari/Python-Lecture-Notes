"""
=============================================================================
PROJECT 4: EMBEDDING LAYER IN NATURAL LANGUAGE PROCESSING
=============================================================================
GOAL: Learn what a Word Embedding is, why it is much better than BoW/TF-IDF,
      and how to USE an Embedding Layer inside a deep-learning model (Keras).

WHAT IS A WORD EMBEDDING?
  An embedding is a DENSE numeric vector (e.g. 50, 100, 300 numbers) that
  represents a word's MEANING. Words that are similar end up CLOSE in this
  vector space.
       king   - man   + woman   ~=  queen
       paris  - france + italy  ~=  rome

WHY IS IT BETTER THAN TF-IDF?
  - Captures SEMANTIC meaning (not just word frequency)
  - Much smaller dimension (TF-IDF can have 100,000 cols, embeddings 100)
  - Can be LEARNED automatically by the neural network

EMBEDDING LAYER (in Keras / TensorFlow):
  Input  : an integer (the index of the word in the vocabulary)
  Output : a dense vector of shape (embedding_dim,)
  These vectors are LEARNED during training, just like other weights.
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import nltk
import spacy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# spaCy already gives us PRE-TRAINED word vectors via en_core_web_md/lg
# Note: en_core_web_sm has NO real vectors. To see real vectors:
#       python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_sm")


# =============================================================================
# PART A: PRE-TRAINED EMBEDDINGS WITH SPACY
# =============================================================================
print("=" * 70)
print("PART A: PRE-TRAINED EMBEDDINGS  (spaCy)")
print("=" * 70)

words = ["king", "queen", "man", "woman", "apple", "banana", "computer"]

# Each word in spaCy has a .vector property (300-dim if using md/lg model)
print("\nWord vector shapes (spaCy):")
for w in words:
    token = nlp(w)[0]
    # token.vector -> the numeric embedding
    print(f"  {w:<10} vector shape = {token.vector.shape}")

# --- WORD SIMILARITY ---
# spaCy can compute cosine similarity between vectors directly.
# (For meaningful results, use en_core_web_md or en_core_web_lg)
print("\nWord Similarity (closer to 1 = more similar):")
pairs = [("king", "queen"), ("king", "apple"), ("man", "woman"), ("dog", "cat")]
for w1, w2 in pairs:
    t1 = nlp(w1)[0]
    t2 = nlp(w2)[0]
    print(f"  {w1:<8} vs {w2:<8} similarity = {t1.similarity(t2):.3f}")


# =============================================================================
# PART B: BUILDING OUR OWN EMBEDDING LAYER WITH KERAS
# =============================================================================
print("\n" + "=" * 70)
print("PART B: LEARNING EMBEDDINGS WITH A KERAS EMBEDDING LAYER")
print("=" * 70)

# --- Step 1: A small dataset (movie reviews) ---
sentences = [
    "I love this movie",
    "This film was great",
    "Best movie ever",
    "Loved every second",
    "What a fantastic story",
    "I hated this film",
    "Worst movie ever",
    "Boring and dull",
    "Terrible acting and plot",
    "I do not like this movie",
]
labels = np.array([1, 1, 1, 1, 1,    # positive
                   0, 0, 0, 0, 0])   # negative


# --- Step 2: TOKENIZE the text into integer sequences ---
# Each unique word will be assigned an integer ID.
VOCAB_SIZE = 50            # max number of words we keep
MAX_LEN = 6                # all sequences will be padded/truncated to this length
EMBED_DIM = 8              # each word will be represented by an 8-dim vector

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

print("\nWord -> Integer mapping (first 10):")
for word, idx in list(tokenizer.word_index.items())[:10]:
    print(f"  {word:<12} -> {idx}")

# Convert sentences to lists of integers
sequences = tokenizer.texts_to_sequences(sentences)
print(f"\nExample sequence: '{sentences[0]}' -> {sequences[0]}")

# Pad / truncate so that ALL sequences have the same length
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
print(f"\nPadded matrix shape: {X.shape}\n{X}")


# --- Step 3: BUILD THE MODEL with an Embedding layer ---
# Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)
#   - input_dim  : size of the vocabulary (rows of the embedding matrix)
#   - output_dim : embedding dimension (cols of the embedding matrix)
#   - input_length: length of each input sequence
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBED_DIM,
              input_length=MAX_LEN,
              name="embedding_layer"),
    GlobalAveragePooling1D(),    # average all word vectors -> sentence vector
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),   # binary output (positive/negative)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel architecture:")
model.summary()


# --- Step 4: TRAIN ---
print("\nTraining the model...")
history = model.fit(X, labels, epochs=50, verbose=0)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.2f}")


# --- Step 5: INSPECT THE LEARNED EMBEDDINGS ---
embedding_layer = model.get_layer("embedding_layer")
embedding_matrix = embedding_layer.get_weights()[0]
# embedding_matrix shape = (VOCAB_SIZE, EMBED_DIM)
print(f"\nLearned embedding matrix shape: {embedding_matrix.shape}")

print("\nLearned vector for the word 'movie':")
movie_idx = tokenizer.word_index['movie']
print(f"  index = {movie_idx}")
print(f"  vector = {embedding_matrix[movie_idx]}")


# --- Step 6: PREDICT on new sentences ---
new_reviews = [
    "I love this story",
    "This was terrible and boring",
]
new_seq = tokenizer.texts_to_sequences(new_reviews)
new_X = pad_sequences(new_seq, maxlen=MAX_LEN, padding='post')
preds = model.predict(new_X, verbose=0)

print("\nPredictions on new reviews:")
for review, prob in zip(new_reviews, preds):
    sentiment = "POSITIVE" if prob[0] > 0.5 else "NEGATIVE"
    print(f"  '{review}'  ->  {sentiment}  (score={prob[0]:.2f})")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Two ways to use embeddings:
  1. PRE-TRAINED  (spaCy / Word2Vec / GloVe / FastText)
       -> Quick, no training needed, captures general meaning
  2. LEARNED inside your model (Keras Embedding layer)
       -> Tailored to YOUR task, but needs more data

The Embedding Layer is the foundation of every modern NLP model
(LSTM, GRU, Transformer, BERT, GPT — they all start with embeddings).
""")
