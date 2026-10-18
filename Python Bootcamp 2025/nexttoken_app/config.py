"""Central configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "app.db"

MAX_VOCAB = 5000          # largest vocabulary kept (incl. pad + unk)
MAX_SAMPLES = 60000       # cap on training samples per document
MIN_TOKENS = 30           # minimum tokens a document must contain
MODEL_TYPES = ["LSTM", "GRU", "SimpleRNN"]

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
