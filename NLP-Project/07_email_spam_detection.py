"""
=============================================================================
PROJECT 7: EMAIL SPAM DETECTION
=============================================================================
GOAL: Build a classifier that automatically detects whether an email is
      SPAM or HAM (legitimate). This is one of the most classic NLP tasks.

DATASET:
  email_data/spam_ham_dataset.csv
  Columns:  label (ham/spam),  text (the email body)

PIPELINE:
  1. Load and explore the dataset
  2. Clean each email (NLTK + spaCy)
  3. Vectorize with TF-IDF
  4. Train Naive Bayes  AND  Logistic Regression
  5. Evaluate (accuracy + confusion matrix)
  6. Predict on brand-new emails
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import re
import string
import numpy as np
import pandas as pd

import nltk
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# -----------------------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "email_data",
                         "spam_ham_dataset.csv")

print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# CSV columns: Unnamed: 0, label, text, label_num
df = pd.read_csv(DATA_PATH)
print(f"Total emails        : {len(df):,}")
print(f"Columns             : {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nFirst spam example:\n{df[df['label']=='spam']['text'].iloc[0][:200]}...")


# -----------------------------------------------------------------------------
# STEP 2: TEXT CLEANING
# -----------------------------------------------------------------------------
# Emails contain lots of noise:
#   - "Subject:" headers
#   - Forwarded markers
#   - URLs, numbers, HTML
#   - Random punctuation
print("\n" + "=" * 70)
print("STEP 2: CLEANING THE EMAILS")
print("=" * 70)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_email_nltk(text):
    """Fast NLTK-based email cleaner."""
    # Cast to str — some rows may have NaN
    text = str(text).lower()
    # Remove the literal "Subject:" tag at the start
    text = re.sub(r'^subject\s*:\s*', ' ', text)
    # Remove URLs, emails, HTML tags, numbers
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Keep only letters and whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize, drop stopwords + short tokens, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def clean_email_spacy(text):
    """spaCy version — slower but smarter."""
    text = re.sub(r'http\S+|\S+@\S+|<[^>]+>|\d+', ' ', str(text).lower())
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc
                    if tok.is_alpha and not tok.is_stop and len(tok.text) > 2)

# Demo on one email
sample = df['text'].iloc[0]
print(f"\nOriginal: {sample[:120]}...")
print(f"NLTK    : {clean_email_nltk(sample)[:120]}...")
print(f"spaCy   : {clean_email_spacy(sample)[:120]}...")

# Apply NLTK cleaner (faster) to all rows
print("\nCleaning all emails with NLTK...")
df['clean'] = df['text'].apply(clean_email_nltk)
# Drop emails that became empty after cleaning
df = df[df['clean'].str.len() > 0].reset_index(drop=True)
print(f"Emails remaining after cleaning: {len(df):,}")


# -----------------------------------------------------------------------------
# STEP 3: TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: TRAIN / TEST SPLIT")
print("=" * 70)

# Map labels to 0 (ham) and 1 (spam)
y = (df['label'] == 'spam').astype(int).values
X_text = df['clean'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}    Test: {len(X_test):,}")


# -----------------------------------------------------------------------------
# STEP 4: TF-IDF VECTORIZATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: TF-IDF VECTORIZATION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),    # unigrams + bigrams: catches "click here", "free money"
    min_df=2,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"Train matrix: {X_train_vec.shape}")
print(f"Test  matrix: {X_test_vec.shape}")


# -----------------------------------------------------------------------------
# STEP 5: TRAIN TWO MODELS AND COMPARE
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: TRAIN  &  EVALUATE")
print("=" * 70)

# --- Model 1: Multinomial Naive Bayes ---
# Naive Bayes was THE classic spam-detection algorithm for decades.
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)
print(f"\n[Naive Bayes]      Accuracy = {accuracy_score(y_test, nb_pred):.4f}")
print(classification_report(y_test, nb_pred, target_names=["Ham", "Spam"]))

# --- Model 2: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, n_jobs=-1)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)
print(f"\n[LogisticRegression] Accuracy = {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred, target_names=["Ham", "Spam"]))

# Pick the better model for downstream predictions
best_model = lr if accuracy_score(y_test, lr_pred) >= accuracy_score(y_test, nb_pred) else nb
print(f"\n>> Using BEST model: {type(best_model).__name__}")


# -----------------------------------------------------------------------------
# STEP 6: CONFUSION MATRIX
# -----------------------------------------------------------------------------
# Rows = true class, Columns = predicted class
#   [[TN  FP]
#    [FN  TP]]
print("\n" + "=" * 70)
print("STEP 6: CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test, best_model.predict(X_test_vec))
print(f"               Predicted Ham   Predicted Spam")
print(f"Actual Ham      {cm[0,0]:>10}    {cm[0,1]:>10}")
print(f"Actual Spam     {cm[1,0]:>10}    {cm[1,1]:>10}")


# -----------------------------------------------------------------------------
# STEP 7: PREDICT ON BRAND-NEW EMAILS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: PREDICTING ON NEW EMAILS")
print("=" * 70)

new_emails = [
    "Congratulations! You have WON a $1000 Walmart Gift Card. Click here to claim",
    "Hi John, please find attached the project report for our 9 AM meeting",
    "URGENT: Your account has been compromised. Verify your password immediately",
    "Reminder: Lunch with Sarah tomorrow at 1pm at the Italian place near the office",
    "Get cheap meds online! 100% discreet shipping, no prescription needed",
]

new_clean = [clean_email_nltk(e) for e in new_emails]
new_vec = vectorizer.transform(new_clean)
preds = best_model.predict(new_vec)
probs = best_model.predict_proba(new_vec)

for email, p, prob in zip(new_emails, preds, probs):
    label = "SPAM" if p == 1 else "HAM"
    confidence = prob[p]
    print(f"\n  Email: {email[:65]}...")
    print(f"   -> {label}  ({confidence:.1%} confident)")


# -----------------------------------------------------------------------------
# STEP 8: TOP "SPAMMY" WORDS
# -----------------------------------------------------------------------------
# Look at the largest weights from the Logistic Regression — these are
# the words that most strongly indicate SPAM.
print("\n" + "=" * 70)
print("STEP 8: TOP SPAM-INDICATOR WORDS")
print("=" * 70)
if isinstance(best_model, LogisticRegression):
    feats = np.array(vectorizer.get_feature_names_out())
    coefs = best_model.coef_[0]
    top_spam = np.argsort(coefs)[-15:][::-1]
    top_ham  = np.argsort(coefs)[:15]
    print("\nTop SPAM signals:")
    for i in top_spam:
        print(f"   {feats[i]:<25}  weight={coefs[i]:+.3f}")
    print("\nTop HAM signals:")
    for i in top_ham:
        print(f"   {feats[i]:<25}  weight={coefs[i]:+.3f}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
You have built a real spam classifier with ~98% accuracy.
   - Cleaned text with NLTK (regex + lemmatization)
   - TF-IDF unigrams + bigrams
   - Compared Naive Bayes vs Logistic Regression
   - Inspected the actual words that signal spam

NEXT STEPS:
   - Try character-level n-grams (great for catching misspelled spam words)
   - Add metadata features: number of $ signs, count of capitalized words
   - Replace TF-IDF with a deep model (LSTM or BERT) for production use
""")
