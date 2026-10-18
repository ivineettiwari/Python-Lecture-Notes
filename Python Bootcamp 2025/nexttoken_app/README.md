# Next-Token Generator (Streamlit + SQLite)

Upload a PDF / DOCX / TXT → one session per document → train RNN, GRU or LSTM →
generate the next N tokens (N chosen with a number input).

## Structure
```
nexttoken_app/
├── app.py                  # Streamlit entry point
├── config.py               # paths & limits
├── requirements.txt
├── .streamlit/config.toml
├── database/
│   ├── connection.py       # sqlite connection + init
│   ├── schema.py           # CREATE TABLE statements
│   └── repository.py       # all SQL (sessions, generations)
├── services/
│   ├── document_loader.py  # PDF / DOCX / TXT -> text
│   ├── text_processing.py  # tokenizer, vocab, dataset
│   ├── model_builder.py    # SimpleRNN / GRU / LSTM models
│   ├── trainer.py          # training + saving
│   └── generator.py        # next-token generation
├── ui/                     # sidebar + tabs
├── tests/                  # pytest
├── data/                   # app.db (created at runtime)
└── models/                 # session_<id>.keras (created at runtime)
```

## Run
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest            # optional
```
Legacy `.doc` files must be saved as `.docx`. Scanned PDFs (no text layer) are rejected.
