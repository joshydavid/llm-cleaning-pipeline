import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

# ======================
# === CONFIGURATION ====
# ======================
AIB_DATASET = "aib-dataset.csv"
OUTPUT_FILENAME = "aib-dataset_cleaned.csv"
LM_STUDIO_API = os.getenv("LM_STUDIO_API")
LLM_MODEL_ = os.getenv("LLM_MODEL_", "llama-3-8b-instruct-1048k")
GOOGLE_BOOKS_API = os.getenv("GOOGLE_BOOKS_API")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MODEL_PARAMS_BILLIONS = float(os.getenv("MODEL_PARAMS_BILLIONS", "8"))
NUM_ROWS_TO_PROCESSED = int(os.getenv("NUM_ROWS_TO_PROCESSED", "15"))


# ======================
# ====== LLM Role ======
# ======================
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
