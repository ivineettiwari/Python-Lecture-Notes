"""
=============================================================================
PROJECT 5: SENTIMENT ANALYSIS ON AMAZON REVIEWS
=============================================================================
GOAL: Build a model that reads an Amazon product review and predicts whether
      the customer is HAPPY (positive) or UNHAPPY (negative).

DATASET:
  amazon_data/train.ft.txt   (training reviews)
  amazon_data/test.ft.txt    (testing reviews)
  Format: each line starts with __label__1 (negative) or __label__2 (positive)
          followed by the review text.

PIPELINE:
  1. Load the FastText-format file
  2. Clean the text (NLTK + spaCy)
  3. Vectorize with TF-IDF
  4. Train Logistic Regression
  5. Evaluate + predict on new reviews

NOTE: The full file has 3.6 million reviews — we will only load a SAMPLE
      so that this trains in seconds on a laptop.
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import re
import string
import nltk
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)

# spaCy with the parser/NER disabled for SPEED — we only need the tokenizer
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# -----------------------------------------------------------------------------
# STEP 1: LOAD THE AMAZON REVIEWS DATASET
# -----------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "amazon_data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.ft.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test.ft.txt")

# To keep things fast for learning, use only a small slice of the dataset.
N_TRAIN = 20000     # increase to 200_000+ for a real run
N_TEST  = 4000


def load_fasttext_file(path, n_rows):
    """Read the first `n_rows` lines from a FastText-format file.

    Each line looks like:
        __label__1 The product is terrible. I want my money back.
        __label__2 Excellent quality, fast shipping, will buy again!
    """
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            line = line.strip()
            if not line:
                continue
            # First token is the label, rest is the review
            label_tag, _, text = line.partition(" ")
            # __label__1 -> 0 (negative),  __label__2 -> 1 (positive)
            label = 0 if label_tag == "__label__1" else 1
            labels.append(label)
            texts.append(text)
    return texts, np.array(labels)


print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

train_texts, y_train = load_fasttext_file(TRAIN_FILE, N_TRAIN)
test_texts,  y_test  = load_fasttext_file(TEST_FILE,  N_TEST)

print(f"Loaded {len(train_texts):,} training and {len(test_texts):,} test reviews")
print(f"Class balance (train): negative={np.sum(y_train==0):,}, "
      f"positive={np.sum(y_train==1):,}")
print(f"\nExample positive review:\n  {train_texts[np.where(y_train==1)[0][0]][:200]}")
print(f"\nExample negative review:\n  {train_texts[np.where(y_train==0)[0][0]][:200]}")


# -----------------------------------------------------------------------------
# STEP 2: TEXT CLEANING (NLTK version + spaCy version)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: CLEANING THE REVIEWS")
print("=" * 70)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# NLTK-based cleaner -- fast, simple
def clean_nltk(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)        # remove URLs
    text = re.sub(r'<[^>]+>', ' ', text)                  # remove HTML
    text = re.sub(r'[^a-z\s]', ' ', text)                 # keep only letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# spaCy-based cleaner -- slower but uses smarter tokenization + lemmatization
def clean_spacy(text):
    text = re.sub(r'http\S+|<[^>]+>', ' ', text.lower())
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc
                    if not tok.is_stop and not tok.is_punct
                    and not tok.is_space and tok.is_alpha)

# Demo on the first review
print("\nOriginal :", train_texts[0][:120])
print("NLTK     :", clean_nltk(train_texts[0])[:120])
print("spaCy    :", clean_spacy(train_texts[0])[:120])

# Use the NLTK cleaner for the full dataset (much faster on 20k rows)
print("\nCleaning all reviews with NLTK (this takes a few seconds)...")
train_clean = [clean_nltk(t) for t in train_texts]
test_clean  = [clean_nltk(t) for t in test_texts]


# -----------------------------------------------------------------------------
# STEP 3: TF-IDF VECTORIZATION
# -----------------------------------------------------------------------------
# We use unigrams + bigrams ("not good" matters in sentiment!)
print("\n" + "=" * 70)
print("STEP 3: TF-IDF VECTORIZATION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=20000,    # keep only the top 20k terms
    ngram_range=(1, 2),    # use single words AND pairs of words
    min_df=2,              # ignore terms that appear in fewer than 2 docs
)
X_train = vectorizer.fit_transform(train_clean)
X_test  = vectorizer.transform(test_clean)

print(f"Train matrix shape: {X_train.shape}")
print(f"Test  matrix shape: {X_test.shape}")
print(f"Vocabulary size   : {len(vectorizer.vocabulary_)}")


# -----------------------------------------------------------------------------
# STEP 4: TRAIN LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
# Logistic Regression is a strong, fast baseline for text classification.
print("\n" + "=" * 70)
print("STEP 4: TRAINING THE MODEL")
print("=" * 70)

model = LogisticRegression(max_iter=1000, n_jobs=-1, C=1.0)
model.fit(X_train, y_train)
print("Training complete.")


# -----------------------------------------------------------------------------
# STEP 5: EVALUATE
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------------------------------------------------
# STEP 6: PREDICT NEW REVIEWS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: PREDICT ON NEW (UNSEEN) REVIEWS")
print("=" * 70)

my_reviews = [
    "Absolutely love this product. Best purchase I have made all year!",
    "Total waste of money. Broke after two days, do not buy.",
    "It works as advertised. Nothing fancy but does the job well.",
    "The quality is poor and the customer service was rude.",
    "Five stars! Fast delivery, great packaging, exactly as described.",
]

cleaned = [clean_nltk(r) for r in my_reviews]
X_new = vectorizer.transform(cleaned)
probs = model.predict_proba(X_new)
preds = model.predict(X_new)

for review, p, prob in zip(my_reviews, preds, probs):
    label = "POSITIVE" if p == 1 else "NEGATIVE"
    confidence = prob[p]
    print(f"\n  Review : {review}")
    print(f"  Pred   : {label}  (confidence = {confidence:.2%})")


# -----------------------------------------------------------------------------
# STEP 7: WHICH WORDS DRIVE THE PREDICTIONS?
# -----------------------------------------------------------------------------
# Logistic Regression coefficients tell us which words push predictions
# toward POSITIVE (positive coef) or NEGATIVE (negative coef).
print("\n" + "=" * 70)
print("STEP 7: TOP WORDS PER CLASS")
print("=" * 70)

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = model.coef_[0]

top_pos_idx = np.argsort(coefs)[-15:][::-1]
top_neg_idx = np.argsort(coefs)[:15]

print("\nTop 15 POSITIVE words/phrases:")
for i in top_pos_idx:
    print(f"  {feature_names[i]:<25} weight={coefs[i]:+.3f}")

print("\nTop 15 NEGATIVE words/phrases:")
for i in top_neg_idx:
    print(f"  {feature_names[i]:<25} weight={coefs[i]:+.3f}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
You built a complete Sentiment Analysis pipeline on real Amazon data:
   - Loaded a FastText-formatted file
   - Cleaned text with NLTK / spaCy
   - Vectorized using TF-IDF with unigrams + bigrams
   - Trained Logistic Regression
   - Evaluated on a held-out test set
   - Inspected which words push the prediction

Try increasing N_TRAIN to 200_000 for higher accuracy (~92%+).
""")
