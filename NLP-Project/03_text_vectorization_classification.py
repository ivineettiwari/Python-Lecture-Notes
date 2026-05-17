"""
=============================================================================
PROJECT 3: TEXT VECTORIZATION AND CLASSIFICATION
=============================================================================
GOAL: Convert text into numbers (vectors) and use them to train a Machine
      Learning model that can CLASSIFY new text into categories.

WHY VECTORIZE?
  Machine Learning models do not understand words — they only work with
  numbers. So we must convert "I love NLP" into something like [0, 1, 1, 0, 1].

VECTORIZATION TECHNIQUES WE WILL COVER:
  1. Bag of Words (BoW)   -> counts how many times each word appears
  2. TF-IDF               -> weights words by how RARE / IMPORTANT they are
  3. (We'll see Word Embeddings in Project 4)

CLASSIFICATION TASK:
  Build a model that decides whether a sentence is about SPORTS or TECHNOLOGY.
=============================================================================
"""

import nltk
import spacy
import re
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------------------------
# STEP 1: A TINY DATASET (5 sports + 5 tech sentences)
# -----------------------------------------------------------------------------
texts = [
    # Sports (label = 0)
    "The football match between Real Madrid and Barcelona was thrilling",
    "Cricket world cup final is being played in India this Sunday",
    "The basketball player scored 40 points in the NBA finals",
    "He won the gold medal in the Olympic swimming championship",
    "Tennis legend Roger Federer announced his retirement from the sport",
    # Tech (label = 1)
    "Apple released a new iPhone with an advanced AI chip and better camera",
    "Google launched a new large language model competing with ChatGPT",
    "Microsoft is investing billions of dollars in cloud computing and AI",
    "The latest Android update brings better security and battery life",
    "Tesla unveiled a new self-driving car powered by neural networks",
]

labels = [0, 0, 0, 0, 0,    # sports
          1, 1, 1, 1, 1]    # tech

label_names = {0: "Sports", 1: "Technology"}


# -----------------------------------------------------------------------------
# STEP 2: TEXT CLEANING (using both NLTK and spaCy)
# -----------------------------------------------------------------------------
def clean_with_nltk(text):
    """Quick NLTK cleaner: lowercase + remove punctuation + remove stop words."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop and len(t) > 1]
    return " ".join(tokens)


def clean_with_spacy(text):
    """spaCy cleaner: lemmatize + remove stop words + punctuation."""
    doc = nlp(text.lower())
    tokens = [tok.lemma_ for tok in doc
              if not tok.is_stop and not tok.is_punct and not tok.is_space]
    return " ".join(tokens)


print("=" * 70)
print("STEP 2: CLEANING THE TEXT")
print("=" * 70)
print("Original :", texts[0])
print("NLTK     :", clean_with_nltk(texts[0]))
print("spaCy    :", clean_with_spacy(texts[0]))

# Use spaCy cleaning for the model (since it lemmatizes)
clean_texts = [clean_with_spacy(t) for t in texts]


# -----------------------------------------------------------------------------
# STEP 3: VECTORIZATION  --  TECHNIQUE 1: BAG OF WORDS
# -----------------------------------------------------------------------------
# BoW: build a vocabulary of all words, then count occurrences in each doc.
# Result: each document becomes a vector of length = vocabulary size.
print("\n" + "=" * 70)
print("STEP 3a: BAG OF WORDS (CountVectorizer)")
print("=" * 70)

bow = CountVectorizer()
X_bow = bow.fit_transform(clean_texts)   # learns vocab + transforms

print(f"Vocabulary size: {len(bow.vocabulary_)}")
print(f"Shape of matrix: {X_bow.shape}  (10 docs x {X_bow.shape[1]} unique words)")
print(f"First 15 vocab words: {bow.get_feature_names_out()[:15]}")
print(f"\nDocument 0 vector (first 15 values): {X_bow.toarray()[0][:15]}")


# -----------------------------------------------------------------------------
# STEP 4: VECTORIZATION  --  TECHNIQUE 2: TF-IDF
# -----------------------------------------------------------------------------
# TF-IDF = Term Frequency * Inverse Document Frequency
#   - Words common in ONE document but rare across all documents get HIGH score
#   - Common words like "the" get LOW score
# Almost always works BETTER than plain BoW.
print("\n" + "=" * 70)
print("STEP 3b: TF-IDF VECTORIZATION")
print("=" * 70)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(clean_texts)

print(f"Shape: {X_tfidf.shape}")
print(f"Document 0 TF-IDF (non-zero values):")
row = X_tfidf.toarray()[0]
vocab = tfidf.get_feature_names_out()
for i, val in enumerate(row):
    if val > 0:
        print(f"   {vocab[i]:<15} = {val:.3f}")


# -----------------------------------------------------------------------------
# STEP 5: TRAIN A CLASSIFIER
# -----------------------------------------------------------------------------
# We will use TWO classic algorithms for text classification:
#   - Multinomial Naive Bayes (perfect for text)
#   - Logistic Regression     (also very strong baseline)
print("\n" + "=" * 70)
print("STEP 4: TRAINING THE CLASSIFIER")
print("=" * 70)

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Model 1: Naive Bayes ---
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"\n[Naive Bayes] Accuracy = {accuracy_score(y_test, y_pred_nb):.2f}")

# --- Model 2: Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"[Logistic Reg] Accuracy = {accuracy_score(y_test, y_pred_lr):.2f}")


# -----------------------------------------------------------------------------
# STEP 6: PREDICT NEW (UNSEEN) SENTENCES
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: PREDICTING NEW SENTENCES")
print("=" * 70)

new_sentences = [
    "Lionel Messi scored a hat-trick in last night's match",
    "Nvidia's new GPU is powering generative AI breakthroughs",
    "The Indian cricket team won the tournament after a tight final",
    "OpenAI released a new version of ChatGPT with improved reasoning",
]

# IMPORTANT: apply the SAME cleaning + the SAME vectorizer (already fit)
cleaned_new = [clean_with_spacy(s) for s in new_sentences]
X_new = tfidf.transform(cleaned_new)        # use .transform, NOT fit_transform!

predictions = lr.predict(X_new)

for sentence, pred in zip(new_sentences, predictions):
    print(f"  '{sentence[:60]}...' -> {label_names[pred]}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
You learned the full pipeline of TEXT CLASSIFICATION:
  1. Collect labeled text data
  2. Clean / preprocess it (NLTK or spaCy)
  3. Convert to numeric vectors (BoW or TF-IDF)
  4. Train an ML model (Naive Bayes / Logistic Regression)
  5. Use it to predict labels for NEW unseen text

Always remember: use fit_transform() ONLY on training data,
                  and transform() on test/new data.
""")
