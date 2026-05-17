"""
=============================================================================
PROJECT 9: INTRODUCTION TO LARGE LANGUAGE MODELS  (Working with OpenAI API)
=============================================================================
GOAL: Learn what an LLM is, how it differs from classical NLP, and how to
      USE the OpenAI API to build powerful applications.

WHAT IS AN LLM?
  A Large Language Model is a neural network (a Transformer) trained on
  hundreds of billions of words. It can:
      - chat / answer questions
      - summarize long documents
      - translate between languages
      - generate code
      - extract structured data from unstructured text
      - reason about problems

POPULAR LLMs:
  - OpenAI:    GPT-4o, GPT-4o-mini, GPT-3.5
  - Anthropic: Claude (Opus, Sonnet, Haiku)
  - Google:    Gemini
  - Meta:      LLaMA  (open source)

WHAT YOU WILL BUILD HERE:
   1. A simple chat completion
   2. A sentiment classifier (NO training data needed!)
   3. A text summarizer
   4. A translator
   5. A spam detector that returns JSON
   6. A small chatbot with memory

PREREQUISITES:
   pip install openai python-dotenv
   1. Sign up at https://platform.openai.com/
   2. Get your API key from "API keys" section
   3. Set it as an environment variable:
        Windows : setx OPENAI_API_KEY "sk-..."
        Mac/Linux: export OPENAI_API_KEY="sk-..."
      OR put it in a `.env` file:
        OPENAI_API_KEY=sk-...
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import json

# `python-dotenv` lets us load the API key from a `.env` file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass    # not required if env var is set another way

# Read the API key from the environment
API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("=" * 70)
    print("WARNING: OPENAI_API_KEY environment variable is not set.")
    print("This file demonstrates the code; actual API calls will fail until")
    print("you set your key.  See instructions at the top of the file.")
    print("=" * 70)


# -----------------------------------------------------------------------------
# SET UP THE OPENAI CLIENT
# -----------------------------------------------------------------------------
# Modern API uses the `openai` package (>= 1.0). Old style: openai.ChatCompletion
# New style:  client.chat.completions.create(...)
from openai import OpenAI

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"        # fast + cheap for learning;  use "gpt-4o" for best quality


# =============================================================================
# EXAMPLE 1: A SIMPLE CHAT COMPLETION
# =============================================================================
# Every call has a list of "messages" with roles:
#   system    -> instructions for the model's behavior
#   user      -> what you (the human) want
#   assistant -> the model's previous replies (used for multi-turn chat)
def simple_chat():
    """Send one user prompt and print the answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful NLP tutor."},
            {"role": "user",   "content": "Explain what tokenization is in NLP, in 2 sentences."},
        ],
        temperature=0.3,        # lower = more deterministic, higher = more creative
        max_tokens=150,
    )
    # The actual text is at choices[0].message.content
    print("\n--- EXAMPLE 1: SIMPLE CHAT ---")
    print(response.choices[0].message.content)


# =============================================================================
# EXAMPLE 2: ZERO-SHOT SENTIMENT CLASSIFIER
# =============================================================================
# No training data needed! We just ask the LLM to classify the text.
def classify_sentiment(text):
    """Use the LLM as a zero-shot sentiment classifier."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a sentiment classifier. Reply ONLY with one "
                        "word: POSITIVE, NEGATIVE, or NEUTRAL."},
            {"role": "user", "content": text},
        ],
        temperature=0,    # we want a deterministic single-word answer
        max_tokens=5,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# EXAMPLE 3: SUMMARIZATION
# =============================================================================
def summarize(text, max_words=50):
    """Summarize a long passage into <= max_words words."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": f"Summarize the user's text in at most {max_words} words. "
                        "Be concise and capture the main idea."},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# EXAMPLE 4: TRANSLATION
# =============================================================================
def translate(text, target_language="French"):
    """Translate text into the target language."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": f"Translate the user's English text into {target_language}. "
                        "Return only the translation, nothing else."},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# EXAMPLE 5: STRUCTURED OUTPUT  (return JSON)
# =============================================================================
# We can ask the LLM to return strict JSON, then parse it. Very useful for
# building APIs/pipelines on top of unstructured user text.
def spam_detector_json(email_text):
    """Classify an email and return a structured JSON result."""
    prompt = f"""
Analyze the following email and decide if it is spam.
Return a JSON object with these EXACT fields:
   - "is_spam": true or false
   - "confidence": a number between 0 and 1
   - "reason": short explanation (one sentence)

Email:
\"\"\"{email_text}\"\"\"
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a strict JSON-only response generator. "
                        "Reply with valid JSON only — no Markdown, no commentary."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},   # forces JSON output
        temperature=0,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)     # convert JSON string to a Python dict


# =============================================================================
# EXAMPLE 6: MULTI-TURN CHATBOT WITH MEMORY
# =============================================================================
# By appending each user turn AND the assistant's reply to the messages list,
# we create a chatbot that REMEMBERS the conversation history.
def chatbot_loop():
    """An interactive chatbot. Type 'quit' to exit."""
    history = [
        {"role": "system",
         "content": "You are a friendly assistant who teaches Natural Language "
                    "Processing concepts in simple words."}
    ]
    print("\n--- EXAMPLE 6: CHATBOT WITH MEMORY (type 'quit' to exit) ---")
    while True:
        user_msg = input("\nYou: ").strip()
        if user_msg.lower() in {"quit", "exit", ""}:
            print("Bye!")
            break

        history.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})

        print(f"Bot: {reply}")


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================
if __name__ == "__main__":
    if not API_KEY:
        print("\nNo API key found — exiting demo. Set OPENAI_API_KEY and try again.")
        raise SystemExit(0)

    # --- 1. Simple chat ---
    simple_chat()

    # --- 2. Sentiment classification ---
    print("\n--- EXAMPLE 2: ZERO-SHOT SENTIMENT ---")
    reviews = [
        "I absolutely love this product, it changed my life!",
        "Terrible experience, the item arrived broken.",
        "It's okay, nothing special.",
    ]
    for r in reviews:
        print(f"  '{r}' -> {classify_sentiment(r)}")

    # --- 3. Summarization ---
    print("\n--- EXAMPLE 3: SUMMARIZATION ---")
    long_text = (
        "Natural Language Processing (NLP) is a subfield of artificial "
        "intelligence focused on the interaction between computers and human "
        "language. NLP enables computers to read, understand, and generate "
        "human languages. Modern NLP relies heavily on deep learning, "
        "particularly Transformer-based architectures like BERT and GPT. "
        "Applications include search engines, chatbots, machine translation, "
        "sentiment analysis, and text summarization."
    )
    print("Summary:", summarize(long_text, max_words=30))

    # --- 4. Translation ---
    print("\n--- EXAMPLE 4: TRANSLATION ---")
    eng = "Machine learning is changing the world rapidly."
    print(f"  EN -> FR: {translate(eng, 'French')}")
    print(f"  EN -> HI: {translate(eng, 'Hindi')}")
    print(f"  EN -> ES: {translate(eng, 'Spanish')}")

    # --- 5. Structured JSON output ---
    print("\n--- EXAMPLE 5: SPAM DETECTOR (JSON output) ---")
    test_email = ("Congratulations! You have WON a free iPhone. "
                  "Click http://win-prize.com to claim now!")
    result = spam_detector_json(test_email)
    print(f"  Result: {result}")

    # --- 6. Chatbot ---  (uncomment to try interactively)
    # chatbot_loop()


# =============================================================================
# SUMMARY
# =============================================================================
print("""

============================================================
SUMMARY:  WHY LLMs ARE A GAME CHANGER
============================================================
Compared to classical NLP (TF-IDF + Logistic Regression, LSTMs, etc.):

CLASSICAL NLP                    LLM (GPT, Claude, Gemini)
--------------------------       --------------------------
Need labelled data              Zero-shot — just describe the task
One model per task              ONE model handles ALL tasks
Manual feature engineering      No features — model figures it out
Fast & cheap                    Slower & paid per token
Predictable, easy to debug      Stochastic, occasionally hallucinates

When to use which:
   - High-volume, low-cost tasks  -> classical NLP
   - Complex tasks with little data -> LLM
   - Production: combine BOTH (LLM for hard cases, classical for the rest)

NEXT STEPS:
   - Try function calling (let the LLM call your Python functions)
   - Try Retrieval-Augmented Generation (RAG) on your own documents
   - Compare OpenAI vs Anthropic Claude vs open-source LLaMA
""")
