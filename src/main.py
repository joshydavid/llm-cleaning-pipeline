import pandas as pd

from constants import (
    AIB_DATASET,
    GOOGLE_BOOKS_API,
    LLM_MODEL,
    LM_STUDIO_API_BASE_URL,
)
from RAGCleaningPipeline import RAGCleaningPipeline

if __name__ == "__main__":
    try:
        print(f"Loading {AIB_DATASET}...")
        df = pd.read_csv(AIB_DATASET)

        rag_cleaning_pipeline = RAGCleaningPipeline(
            google_books_api_url=GOOGLE_BOOKS_API,
            llm_api_base_url=LM_STUDIO_API_BASE_URL,
            llm_model=LLM_MODEL,
        )
        rag_cleaning_pipeline.run_pipeline(df)

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
