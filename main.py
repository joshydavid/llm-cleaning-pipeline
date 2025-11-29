import json
import math
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =====================
# === CONFIGURATION ===
# =====================
AIB_DATASET = "aib-dataset.csv"
OUTPUT_FILENAME = "aib-dataset_cleaned.csv"
LM_STUDIO_API = os.getenv("LM_STUDIO_API")
LLM_MODEL_ = os.getenv("LLM_MODEL_", "llama-3-8b-instruct-1048k")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MODEL_PARAMS_BILLIONS = float(os.getenv("MODEL_PARAMS_BILLIONS", "8"))


# ========================
# === GOOGLE BOOKS API ===
# ========================
def _fetch_book_context(title, author_hint):
    """
    Searches Google Books API to find the 'Ground Truth' for a book.
    """

    if not isinstance(title, str):
        return "No Title Provided"

    clean_title = title.split("(")[0].strip()
    query = f"intitle:{clean_title}"
    if isinstance(author_hint, str):
        clean_author = author_hint.split(",")[0].strip()
        query += f"+inauthor:{clean_author}"

    GOOGLE_BOOKS_API_URL = (
        f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
    )

    try:
        response = requests.get(GOOGLE_BOOKS_API_URL).json()
        if "items" in response and len(response["items"]) > 0:
            info = response["items"][0]["volumeInfo"]
            return {
                "found_title": info.get("title", "N/A"),
                "found_authors": info.get("authors", ["N/A"]),
                "found_year": info.get("publishedDate", "N/A")[:4],
                "found_pages": info.get("pageCount", "N/A"),
                "found_ratings_count": info.get("ratingsCount", 0),
            }
        return "Book not found in external database."
    except Exception as e:
        return f"API Error: {str(e)}"


# =======================
# === FLOPS ESTIMATOR ===
# =======================
def estimate_flops(input_text, output_text, model_params_billions):
    # Rough token estimation (1 word ~= 1.3 tokens)
    in_tok = len(input_text.split()) * 1.3
    out_tok = len(output_text.split()) * 1.3
    return 2 * (model_params_billions * 1e9) * (in_tok + out_tok)


# =======================
# === MAIN PROCESSING ===
# =======================
def process_batch_with_rag(batch_df, client):
    batch_context = _retrieve_book_context(batch_df)
    results, duration, flops = _clean_batch_with_llm(batch_context, client)
    return results or [], duration, flops


def _retrieve_book_context(batch_df):
    """
    Step 1: Retrieve ground-truth book info from Google Books API for the batch.
    """

    batch_context = []
    print(f" > Retrieving context for {len(batch_df)} books...")
    for idx, row in batch_df.iterrows():
        external_info = _fetch_book_context(row["Title"], row["Author"])
        batch_context.append(
            {
                "id": idx,
                "original_data": row[
                    ["Title", "Author", "Year", "ISBN", "Series", "Pages"]
                ].to_dict(),
                "retrieved_context": external_info,
            }
        )
    return batch_context


def _clean_batch_with_llm(batch_context, client):
    """
    Step 2: Send RAG prompt to the local LLM and parse the cleaned JSON response.
    """

    input_json = json.dumps(batch_context, indent=2, ensure_ascii=False)
    prompt = f"""
     You are a Data Cleaning Assistant using Retrieval-Augmented Generation (RAG).
     I will provide a list of books with "original_data" (which may have typos/missing values)
     and "retrieved_context" (data found from a trusted Google Books API).

     YOUR TASKS:
     1. Correct the Author and Title using the 'retrieved_context'.
     2. FILL MISSING Year and Pages using 'retrieved_context'.
     3. Fix the ISBN if possible.
     4. Classify Sales based on 'found_ratings_count':
         - > 100 ratings = "High Sales"
         - > 10 ratings = "Medium Sales"
         - < 10 ratings = "Low Sales"

     INPUT DATA:
     {input_json}

     OUTPUT ONLY a valid JSON array with the following keys:
     Title, Author, Year, ISBN, Series, Pages, Sales
     """

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON-only API. Never add explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        content = response.choices[0].message.content.strip()
        duration = time.time() - start_time
        flops = estimate_flops(prompt, content, MODEL_PARAMS_BILLIONS)
        clean_json = content.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(clean_json)
        return data, duration, flops

    except Exception as e:
        print(f" > LLM call failed: {e}")
        return None, 0, 0


# ==========================
# === EXECUTION SKELETON ===
# ==========================
if __name__ == "__main__":
    try:
        # 1. SETUP
        client = OpenAI(base_url=LM_STUDIO_API, api_key="lm-studio")
        print(f"Loading {AIB_DATASET}...")
        df = pd.read_csv(AIB_DATASET)

        # 2. PREPROCESSING
        df = df.rename(
            columns={
                "anno pubblicazione": "Year",
                "codice": "ISBN",
                "titolo": "Title",
                "autore": "Author",
                "collana": "Series",
                "pagine": "Pages",
            }
        )

        df["Year"] = df["Year"].astype("Int64")
        df["Pages"] = df["Pages"].astype("Int64")

        print("Limiting processing to the first 10 rows for testing...")
        df = df.head(50).copy()

        all_results = []
        total_flops = 0
        total_time = 0

        # 3. BATCH PROCESSING LOOP
        total_batches = math.ceil(len(df) / BATCH_SIZE)
        print(f"Starting processing of {len(df)} rows in {total_batches} batches...")

        for i in range(total_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch = df.iloc[start_idx:end_idx]

            print(
                f"\n--- Batch {i + 1}/{total_batches} (Rows {start_idx}-{end_idx}) ---"
            )

            # Run RAG Pipeline
            results, duration, flops = process_batch_with_rag(batch, client)

            if results:
                print(
                    f"   > ✅ Success. {len(results)} items cleaned in {duration:.2f}s"
                )
                all_results.extend(results)
                total_flops += flops
                total_time += duration
            else:
                print("   > ⚠️ Batch failed. Using original data as fallback.")
                all_results.extend(batch.to_dict(orient="records"))

            time.sleep(1)

        # 4. FINAL OUTPUT & SAVING
        if all_results:
            final_df = pd.DataFrame(all_results)

            print("\n" + "=" * 40)
            print("       FINAL DATA PREVIEW")
            print("=" * 40 + "\n")
            print(final_df.head(10).to_string(index=False))
            print("\n" + "-" * 40)
            print(f"Total Processing Time: {total_time:.2f}s")
            print(f"Total Estimated FLOPs: {total_flops:.2e}")

            final_df.to_csv(OUTPUT_FILENAME, index=False)
            print(f"\n✅ Saved cleaned dataset to: {OUTPUT_FILENAME}")

        else:
            print("\n❌ No results generated.")

    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
