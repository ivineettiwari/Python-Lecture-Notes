"""
=============================================================================
PROJECT 1: INTRODUCTION TO NATURAL LANGUAGE PROCESSING (NLP)
=============================================================================
GOAL: Understand what NLP is and explore the basic building blocks of text
      processing using NLTK and spaCy. We will analyze a sample paragraph
      and demonstrate how a computer can "understand" text.

WHAT IS NLP?
NLP is a branch of Artificial Intelligence that helps computers understand,
interpret, and generate human language. Examples: Google Translate, Siri,
ChatGPT, Spam Filters, Auto-correct, etc.

LIBRARIES WE WILL USE:
- NLTK  (Natural Language Toolkit) -> Educational, rule-based, classic NLP
- spaCy (Industrial-strength NLP)  -> Fast, production-ready, pre-trained
=============================================================================
"""

# -----------------------------------------------------------------------------
# STEP 1: IMPORT THE REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
import nltk        # NLTK for traditional NLP tasks
import spacy       # spaCy for modern, fast NLP

# Download NLTK data (only needs to run once on a new machine)
# 'punkt'           -> for sentence/word tokenization
# 'averaged_perceptron_tagger' -> for Part-of-Speech tagging
# 'maxent_ne_chunker' & 'words' -> for Named Entity Recognition
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

# Load the small English model for spaCy (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


# -----------------------------------------------------------------------------
# STEP 2: OUR SAMPLE TEXT (the "document" we want the computer to understand)
# -----------------------------------------------------------------------------
text = (
    "Natural Language Processing is an exciting field of Artificial Intelligence. "
    "Companies like Google, Microsoft, and OpenAI are investing heavily in NLP. "
    "In 2023, ChatGPT became one of the fastest-growing apps in the world."
)

print("=" * 70)
print("ORIGINAL TEXT:")
print("=" * 70)
print(text, "\n")


# =============================================================================
# PART A: USING NLTK
# =============================================================================
print("=" * 70)
print("PART A:  N L T K")
print("=" * 70)

# --- 1. SENTENCE TOKENIZATION ---
# Splits the text into a list of sentences.
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(text)
print("\n[NLTK] Sentence Tokenization:")
for i, s in enumerate(sentences, 1):
    print(f"  {i}. {s}")

# --- 2. WORD TOKENIZATION ---
# Splits a sentence into individual words / punctuation marks.
words = word_tokenize(text)
print("\n[NLTK] Word Tokenization (first 15 tokens):")
print(" ", words[:15])

# --- 3. PART-OF-SPEECH (POS) TAGGING ---
# Labels each word with its grammatical role (noun, verb, adjective, etc.)
pos_tags = nltk.pos_tag(words)
print("\n[NLTK] POS Tagging (first 10):")
print(" ", pos_tags[:10])
# NN = Noun, VB = Verb, JJ = Adjective, IN = Preposition, etc.

# --- 4. NAMED ENTITY RECOGNITION (NER) ---
# Finds real-world entities like people, organizations, dates, locations.
entities = nltk.ne_chunk(pos_tags)
print("\n[NLTK] Named Entities Found:")
for chunk in entities:
    if hasattr(chunk, 'label'):       # Only print the tagged entities
        entity_name = " ".join(c[0] for c in chunk)
        print(f"  - {entity_name}  ({chunk.label()})")


# =============================================================================
# PART B: USING SPACY
# =============================================================================
print("\n" + "=" * 70)
print("PART B:  S P A C Y")
print("=" * 70)

# spaCy processes the entire text in ONE pass and returns a "Doc" object
# that contains tokens, POS tags, entities, dependencies — all at once.
doc = nlp(text)

# --- 1. SENTENCE TOKENIZATION (spaCy) ---
print("\n[spaCy] Sentence Tokenization:")
for i, sent in enumerate(doc.sents, 1):
    print(f"  {i}. {sent.text}")

# --- 2. WORD TOKENIZATION + POS TAGGING (spaCy does both at once) ---
print("\n[spaCy] Tokens + POS (first 10):")
for token in doc[:10]:
    # token.text -> the actual word
    # token.pos_ -> coarse POS (NOUN, VERB, ADJ, ...)
    # token.tag_ -> fine-grained tag (NN, NNS, VBD, ...)
    print(f"  {token.text:<15} POS={token.pos_:<8} TAG={token.tag_}")

# --- 3. NAMED ENTITY RECOGNITION (spaCy) ---
# Much more accurate and easier to read than NLTK's NER.
print("\n[spaCy] Named Entities:")
for ent in doc.ents:
    # ent.text  -> the entity string
    # ent.label_ -> the type (ORG, PERSON, DATE, GPE, ...)
    print(f"  - {ent.text:<20} ({ent.label_})")

# --- 4. LEMMATIZATION ---
# Reduces a word to its dictionary form. e.g. "investing" -> "invest"
print("\n[spaCy] Lemmatization (first 10 tokens):")
for token in doc[:10]:
    print(f"  {token.text:<15} -> {token.lemma_}")


# =============================================================================
# SUMMARY / WHAT YOU LEARNED
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
You just performed the four most common NLP tasks:
  1. Tokenization  - breaking text into words/sentences
  2. POS Tagging   - finding the grammatical role of each word
  3. NER           - identifying real-world entities (people, places, orgs)
  4. Lemmatization - reducing words to their root form

NLTK is great for LEARNING the concepts step-by-step.
spaCy is great for BUILDING real applications (faster + more accurate).
""")
