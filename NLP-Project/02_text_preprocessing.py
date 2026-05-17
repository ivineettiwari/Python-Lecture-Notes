"""
=============================================================================
PROJECT 2: TEXT PREPROCESSING USING NLTK AND SPACY
=============================================================================
GOAL: Clean and prepare raw text so that a Machine Learning model can use it.
      Real-world text is messy: HTML, punctuation, capitalization, stop words,
      different verb forms, etc. We must standardize it first.

PIPELINE WE WILL BUILD:
  Raw text  ->  Lowercase  ->  Remove HTML/URLs  ->  Remove punctuation
            ->  Tokenize   ->  Remove Stop Words ->  Stem/Lemmatize  -> Clean text

WHY PREPROCESS?
  - Reduces vocabulary size  (-> faster training)
  - Removes noise            (-> better accuracy)
  - Standardizes the input   (-> "Running", "ran", "runs" all become "run")
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import re                       # for regular expressions (HTML, URLs, etc.)
import string                   # has a list of all punctuation characters
import nltk
import spacy

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords         # list of common words to remove ("the", "is")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer        # cuts off word endings (fast, rough)
from nltk.stem import WordNetLemmatizer    # uses dictionary (slower, accurate)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------------------------
# SAMPLE NOISY TEXT (a fake product review with all kinds of mess)
# -----------------------------------------------------------------------------
raw_text = """
<p>I LOVED this product!!! It's the BEST phone I have ever owned :-)
Visit https://example.com/review for more details. The battery is running
amazingly, and the cameras are better than my older phones. #Awesome 100%</p>
"""

print("=" * 70)
print("ORIGINAL RAW TEXT:")
print("=" * 70)
print(raw_text)


# =============================================================================
# PART A: TEXT PREPROCESSING USING NLTK
# =============================================================================
print("=" * 70)
print("PART A:  PREPROCESSING WITH N L T K")
print("=" * 70)

def preprocess_with_nltk(text):
    """Full preprocessing pipeline using NLTK."""

    # STEP 1: LOWERCASE  -> "LOVED" and "loved" become the same token
    text = text.lower()

    # STEP 2: REMOVE HTML TAGS  -> <p>, </p> etc.
    text = re.sub(r'<.*?>', ' ', text)

    # STEP 3: REMOVE URLS  -> https://...  www....
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # STEP 4: REMOVE NUMBERS (optional — sometimes numbers matter)
    text = re.sub(r'\d+', ' ', text)

    # STEP 5: REMOVE PUNCTUATION using str.translate (fast)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # STEP 6: TOKENIZE -> split into individual words
    tokens = word_tokenize(text)

    # STEP 7: REMOVE STOP WORDS  ("the", "is", "and", "a", ...)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # STEP 8a: STEMMING -> chops the end off words (fast but ugly)
    #   "running" -> "run",  "amazingly" -> "amazingli"
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in tokens]

    # STEP 8b: LEMMATIZATION -> proper dictionary form (slower but readable)
    #   "running" -> "running" (need POS for verb), "cameras" -> "camera"
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t, pos='v') for t in tokens]  # 'v' = verb

    return tokens, stemmed, lemmatized


tokens, stemmed, lemmatized = preprocess_with_nltk(raw_text)

print("\n[NLTK] Cleaned tokens (no stopwords/punct):")
print(" ", tokens)
print("\n[NLTK] After STEMMING:")
print(" ", stemmed)
print("\n[NLTK] After LEMMATIZATION:")
print(" ", lemmatized)


# =============================================================================
# PART B: TEXT PREPROCESSING USING SPACY
# =============================================================================
print("\n" + "=" * 70)
print("PART B:  PREPROCESSING WITH S P A C Y")
print("=" * 70)

def preprocess_with_spacy(text):
    """Full preprocessing pipeline using spaCy.
    spaCy is more elegant — token attributes do most of the work for us."""

    # STEP 1: lowercase + remove HTML/URLs/numbers (regex still needed)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # STEP 2: pass through spaCy pipeline (tokenize + tag + lemmatize all at once)
    doc = nlp(text)

    # STEP 3: filter tokens using built-in spaCy attributes
    cleaned_tokens = []
    for token in doc:
        # token.is_stop  -> True for "the", "is", "and"...
        # token.is_punct -> True for "!", ".", ","
        # token.is_space -> True for " ", "\n", "\t"
        # token.lemma_   -> dictionary form of the word
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if len(token.lemma_) <= 1:
            continue
        cleaned_tokens.append(token.lemma_)

    return cleaned_tokens


spacy_clean = preprocess_with_spacy(raw_text)
print("\n[spaCy] Cleaned + Lemmatized tokens:")
print(" ", spacy_clean)


# =============================================================================
# COMPARING NLTK vs SPACY
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"NLTK tokens count   : {len(lemmatized)}")
print(f"spaCy tokens count  : {len(spacy_clean)}")
print("""
Key differences:
  - NLTK gives you fine-grained CONTROL but requires more code.
  - spaCy is MORE PRODUCTION-READY: one .nlp() call gives you everything.
  - For learning, NLTK is excellent. For real apps, spaCy is preferred.
""")


# =============================================================================
# BONUS: PREPROCESSING A LIST OF DOCUMENTS  (real-world use case)
# =============================================================================
print("=" * 70)
print("BONUS: BATCH PREPROCESSING A LIST OF REVIEWS")
print("=" * 70)

reviews = [
    "I LOVE the new iPhone 15! Camera is amazing.",
    "Battery dies too fast. Not worth $999. Will return!!!",
    "Good build, decent screen, but software is buggy.",
]

print("\nOriginal reviews:")
for r in reviews:
    print(" -", r)

print("\nCleaned with spaCy:")
for r in reviews:
    print(" -", preprocess_with_spacy(r))
