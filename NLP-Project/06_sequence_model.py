"""
=============================================================================
PROJECT 6: SEQUENCE MODELS (RNN / LSTM / GRU)
=============================================================================
GOAL: Understand and build a SEQUENCE model that processes text in ORDER
      (one word at a time) and remembers context — unlike BoW/TF-IDF
      which throws away word order.

WHY DO WE NEED SEQUENCE MODELS?
  - "The movie was not bad"   -> POSITIVE
  - "The movie was bad, not good" -> NEGATIVE
  Both have the same words ("not", "bad", "good") — only ORDER differs.
  TF-IDF cannot tell them apart, but an RNN/LSTM can.

ARCHITECTURE LANDSCAPE:
  RNN   (Vanilla)  -> simple, struggles with long sentences (vanishing grad.)
  LSTM             -> uses gates to remember/forget, handles long sequences
  GRU              -> simpler version of LSTM, often equally good
  Bidirectional    -> reads sequence FORWARDS and BACKWARDS

WE WILL BUILD:
  An LSTM model for binary text classification on a small movie-review dataset.
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import re
import nltk
import spacy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, GRU,
                                     Bidirectional, Dense, Dropout)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# -----------------------------------------------------------------------------
# STEP 1: SAMPLE DATASET (movie reviews, positive vs negative)
# -----------------------------------------------------------------------------
# In a real project this would be IMDB (50,000 reviews) — here we use a small
# set so it trains in seconds.
texts = [
    # Positive reviews
    "I absolutely loved this movie, the acting was brilliant",
    "What a wonderful film, the storyline was amazing",
    "Best movie I have seen in years, fantastic direction",
    "A masterpiece, every scene was beautifully crafted",
    "Loved every minute of it, the cast was outstanding",
    "Heartwarming story with great performances",
    "An emotional roller coaster that I really enjoyed",
    "Brilliant cinematography and a powerful script",
    "Fun, exciting, and totally worth watching",
    "One of the best films of the decade",
    # Negative reviews
    "Terrible movie, complete waste of two hours",
    "Boring plot and the acting was awful",
    "I hated this film, very disappointing",
    "Worst movie I have seen this year, do not watch",
    "Predictable, dull, and poorly directed",
    "The script was lazy and the characters flat",
    "A complete mess from start to finish",
    "Overrated and forgettable, I want my money back",
    "Awful pacing and a confusing storyline",
    "Just bad — bad acting, bad story, bad everything",
]

labels = np.array([1] * 10 + [0] * 10)   # 1 = positive, 0 = negative


# -----------------------------------------------------------------------------
# STEP 2: TEXT CLEANING using spaCy
# -----------------------------------------------------------------------------
def clean(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc
                    if not tok.is_stop and not tok.is_space and tok.is_alpha)

cleaned_texts = [clean(t) for t in texts]
print("Example cleaning:")
print(" Original:", texts[0])
print(" Cleaned :", cleaned_texts[0])


# -----------------------------------------------------------------------------
# STEP 3: TOKENIZATION + PADDING
# -----------------------------------------------------------------------------
# An LSTM requires inputs as INTEGER SEQUENCES of equal length.
print("\n" + "=" * 70)
print("STEP 3: TOKENIZE AND PAD SEQUENCES")
print("=" * 70)

VOCAB_SIZE = 200
MAX_LEN = 10           # all sequences will be padded/truncated to length 10
EMBED_DIM = 16
LSTM_UNITS = 16

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(cleaned_texts)

sequences = tokenizer.texts_to_sequences(cleaned_texts)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Vocabulary size : {len(tokenizer.word_index)}")
print(f"Padded matrix   : {X.shape}")
print(f"First sequence  : {X[0]}")


# -----------------------------------------------------------------------------
# STEP 4: BUILD AN LSTM MODEL
# -----------------------------------------------------------------------------
# LAYER-BY-LAYER EXPLANATION:
#   1. Embedding   -> turns word IDs into dense vectors
#   2. LSTM        -> reads the sequence one step at a time, keeping a "memory"
#   3. Dropout     -> randomly disables neurons during training (reduces overfit)
#   4. Dense (1)   -> final sigmoid neuron -> 0 or 1
print("\n" + "=" * 70)
print("STEP 4: BUILD THE LSTM MODEL")
print("=" * 70)

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBED_DIM,
              input_length=MAX_LEN),
    LSTM(LSTM_UNITS, return_sequences=False),    # use last hidden state only
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# -----------------------------------------------------------------------------
# STEP 5: TRAIN
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: TRAINING THE LSTM")
print("=" * 70)
history = model.fit(X, labels, epochs=40, verbose=0)
print(f"Final train accuracy: {history.history['accuracy'][-1]:.2f}")


# -----------------------------------------------------------------------------
# STEP 6: PREDICT
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: PREDICTIONS")
print("=" * 70)

new_reviews = [
    "An incredible film with outstanding acting",
    "Boring story and terrible direction, total waste",
    "Not bad — actually enjoyed the second half",
    "I really hated this movie",
]

new_clean = [clean(r) for r in new_reviews]
new_seq = tokenizer.texts_to_sequences(new_clean)
new_X = pad_sequences(new_seq, maxlen=MAX_LEN, padding='post')
preds = model.predict(new_X, verbose=0)

for review, prob in zip(new_reviews, preds):
    sentiment = "POSITIVE" if prob[0] > 0.5 else "NEGATIVE"
    print(f"  '{review[:50]}...' -> {sentiment} (score={prob[0]:.2f})")


# =============================================================================
# BONUS: SAME MODEL WITH GRU AND BIDIRECTIONAL LSTM
# =============================================================================
print("\n" + "=" * 70)
print("BONUS: BUILDING GRU AND BIDIRECTIONAL LSTM VARIANTS")
print("=" * 70)

# --- GRU (smaller, faster cousin of LSTM) ---
gru_model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    GRU(LSTM_UNITS),
    Dense(1, activation='sigmoid'),
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Bidirectional LSTM (reads sequence both forward and backward) ---
bilstm_model = Sequential([
    Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(1, activation='sigmoid'),
])
bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nGRU model summary:")
gru_model.summary()
print("\nBidirectional LSTM summary:")
bilstm_model.summary()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
What you learned:
  - Sequence models READ TEXT IN ORDER (unlike BoW/TF-IDF)
  - LSTM/GRU use gates to remember important earlier words
  - Bidirectional LSTMs read forwards AND backwards for richer context
  - Pipeline: Tokenizer -> pad_sequences -> Embedding -> LSTM -> Dense

In the next project we'll apply this same idea to spam detection.
Real-world tip: for big tasks, replace LSTM with a Transformer (BERT, GPT).
""")
