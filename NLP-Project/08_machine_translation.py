"""
=============================================================================
PROJECT 8: MACHINE TRANSLATION (English  ->  French / Hindi / Spanish)
=============================================================================
GOAL: Build a system that translates English sentences into another language.

WE WILL EXPLORE THREE APPROACHES:
  PART A:  Quick & easy — use a PRE-TRAINED Hugging Face model
           (Helsinki-NLP MarianMT)  -> production-quality results

  PART B:  Build a SIMPLE Sequence-to-Sequence (Seq2Seq) model from scratch
           using an LSTM Encoder + LSTM Decoder. This shows HOW machine
           translation actually works under the hood.

  PART C:  A spaCy-based educational toy:  a word-by-word "translator" using a
           tiny English -> French dictionary (just to demonstrate tokenization).

NOTE: True deep-learning translation requires HUGE bilingual datasets and
      hours of GPU training. In this project we DEMONSTRATE the architecture
      with a small toy dataset.
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import nltk
import spacy

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nlp = spacy.load("en_core_web_sm")


# =============================================================================
# PART A: USE A PRE-TRAINED TRANSLATION MODEL  (Hugging Face)
# =============================================================================
# This is what most real applications do: do not train from scratch — use a
# model that already learned from millions of sentence pairs.
#
# Install once:    pip install transformers sentencepiece
print("=" * 70)
print("PART A: PRE-TRAINED HUGGING FACE TRANSLATION (recommended approach)")
print("=" * 70)

try:
    from transformers import pipeline

    # Helsinki-NLP/opus-mt-en-fr  -> English to French
    en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
    sentences = [
        "Hello, how are you?",
        "Machine translation is fascinating.",
        "I love learning natural language processing.",
    ]
    print("\nEnglish -> French:")
    for s in sentences:
        out = en_fr(s)[0]['translation_text']
        print(f"   EN: {s}")
        print(f"   FR: {out}\n")

    # You can swap models for other languages:
    #   Helsinki-NLP/opus-mt-en-hi   -> English to Hindi
    #   Helsinki-NLP/opus-mt-en-es   -> English to Spanish
    #   Helsinki-NLP/opus-mt-en-de   -> English to German
except Exception as e:
    print(f"[Skipping Part A — transformers/sentencepiece not installed]\n  {e}\n"
          "Install with:  pip install transformers sentencepiece torch\n")


# =============================================================================
# PART B: BUILD A SEQ2SEQ TRANSLATOR FROM SCRATCH (educational)
# =============================================================================
# Architecture:
#
#   ENGLISH SENTENCE  ->  [ENCODER LSTM]  -> context vector
#                                              |
#                                              v
#                          [DECODER LSTM]  ->  FRENCH SENTENCE (one word at a time)
#
# The encoder reads the whole input sentence and condenses it into a hidden
# state. The decoder uses that state to generate the target sentence.
print("=" * 70)
print("PART B: BUILDING A TINY SEQ2SEQ TRANSLATOR FROM SCRATCH")
print("=" * 70)

# --- Step 1: A TOY parallel dataset ---
# In the real world this would be millions of sentence pairs.
pairs = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("how are you", "comment ca va"),
    ("i am fine", "je vais bien"),
    ("thank you", "merci"),
    ("yes", "oui"),
    ("no", "non"),
    ("i love you", "je t aime"),
    ("see you tomorrow", "a demain"),
    ("good bye", "au revoir"),
    ("please", "s il vous plait"),
]

# Add special start <START> and end <END> tokens to the target
input_texts = [src for src, _ in pairs]
target_texts = [f"\t{tgt}\n" for _, tgt in pairs]   # \t = START, \n = END


# --- Step 2: BUILD CHARACTER-LEVEL VOCABULARIES ---
# We use characters (not words) because our toy data is tiny.
input_chars  = sorted(set("".join(input_texts)))
target_chars = sorted(set("".join(target_texts)))

input_char_index  = {c: i for i, c in enumerate(input_chars)}
target_char_index = {c: i for i, c in enumerate(target_chars)}
reverse_target_char_index = {i: c for c, i in target_char_index.items()}

n_enc_tokens = len(input_chars)
n_dec_tokens = len(target_chars)
max_enc_len = max(len(t) for t in input_texts)
max_dec_len = max(len(t) for t in target_texts)

print(f"Number of pairs        : {len(pairs)}")
print(f"Unique input chars     : {n_enc_tokens}")
print(f"Unique target chars    : {n_dec_tokens}")
print(f"Max input length       : {max_enc_len}")
print(f"Max target length      : {max_dec_len}")


# --- Step 3: ONE-HOT ENCODE the data ---
encoder_input = np.zeros((len(pairs), max_enc_len, n_enc_tokens), dtype='float32')
decoder_input = np.zeros((len(pairs), max_dec_len, n_dec_tokens), dtype='float32')
decoder_target = np.zeros((len(pairs), max_dec_len, n_dec_tokens), dtype='float32')

for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
    for t, c in enumerate(inp):
        encoder_input[i, t, input_char_index[c]] = 1.0
    for t, c in enumerate(tgt):
        decoder_input[i, t, target_char_index[c]] = 1.0
        if t > 0:
            # decoder_target is shifted by 1 step (teacher forcing)
            decoder_target[i, t - 1, target_char_index[c]] = 1.0


# --- Step 4: BUILD THE MODEL ---
# ENCODER: takes the source sentence, returns its final hidden state
LATENT_DIM = 64

encoder_inputs = Input(shape=(None, n_enc_tokens))
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]    # this is the "thought vector"

# DECODER: takes the encoder state + the previous output, predicts next char
decoder_inputs = Input(shape=(None, n_dec_tokens))
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(n_dec_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# --- Step 5: TRAIN ---
print("\nTraining the seq2seq model...")
model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=8, epochs=200, verbose=0)
print("Training done.")


# --- Step 6: BUILD INFERENCE MODELS ---
# At inference time the decoder runs ONE STEP at a time (since it doesn't know
# the target). So we split the model in two for prediction.
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs2, state_h2, state_c2])


def translate(input_text):
    """Translate one English sentence using the trained seq2seq model."""
    # Encode the input as a one-hot tensor
    x = np.zeros((1, max_enc_len, n_enc_tokens), dtype='float32')
    for t, c in enumerate(input_text):
        if c in input_char_index:
            x[0, t, input_char_index[c]] = 1.0

    # Get the encoder's "thought vector"
    states = encoder_model.predict(x, verbose=0)

    # Start the decoder with the START token \t
    target_seq = np.zeros((1, 1, n_dec_tokens))
    target_seq[0, 0, target_char_index['\t']] = 1.0

    decoded = ''
    while True:
        output, h, c = decoder_model.predict([target_seq] + states, verbose=0)
        idx = np.argmax(output[0, -1, :])
        char = reverse_target_char_index[idx]
        if char == '\n' or len(decoded) > max_dec_len:
            break
        decoded += char
        # Feed the predicted char back in as the next input
        target_seq = np.zeros((1, 1, n_dec_tokens))
        target_seq[0, 0, idx] = 1.0
        states = [h, c]
    return decoded


print("\nTranslations from our hand-built Seq2Seq:")
for sentence in ["hello", "thank you", "i love you", "good night"]:
    print(f"   EN: {sentence:<20} ->  FR: {translate(sentence)}")


# =============================================================================
# PART C: TOY DICTIONARY-BASED TRANSLATION (educational only, NOT real MT)
# =============================================================================
print("\n" + "=" * 70)
print("PART C: WORD-BY-WORD TRANSLATION (toy demo using spaCy tokenizer)")
print("=" * 70)

en_to_fr = {
    "i": "je", "you": "tu", "he": "il", "she": "elle",
    "love": "aime", "hate": "déteste", "eat": "mange",
    "apple": "pomme", "book": "livre", "cat": "chat",
    "dog": "chien", "the": "le", "a": "un", "is": "est",
    "good": "bon", "bad": "mauvais",
}

def naive_translate(sentence):
    """Tokenize with spaCy, then look each word up in the dictionary."""
    doc = nlp(sentence.lower())
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        out.append(en_to_fr.get(tok.text, tok.text))   # fallback to original word
    return " ".join(out)

for s in ["I love the cat", "She is good", "The dog eats a apple"]:
    print(f"   EN: {s}")
    print(f"   FR: {naive_translate(s)}\n")


# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
You explored 3 approaches to Machine Translation:
   A. PRE-TRAINED model (Hugging Face MarianMT)  -> use this in production
   B. SEQ2SEQ from scratch (LSTM Encoder + Decoder) -> understand the theory
   C. Dictionary-based toy translator -> just for tokenization practice

KEY INSIGHTS:
   - Real MT models use Transformer (attention) architectures, not plain LSTMs
   - They are trained on millions of (English, French) sentence pairs
   - Modern systems (Google Translate, DeepL) use these techniques
""")
