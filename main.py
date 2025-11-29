import json
import math
import time

import pandas as pd
import requests
from openai import OpenAI

from constants import (
    AIB_DATASET,
    BATCH_SIZE,
    GOOGLE_BOOKS_API,
    LLM_MODEL_,
    LM_STUDIO_API,
    MODEL_PARAMS_BILLIONS,
    NUM_ROWS_TO_PROCESSED,
    OUTPUT_FILENAME,
    Role,
)


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

    GOOGLE_BOOKS_API_URL = f"{GOOGLE_BOOKS_API}?q={query}&maxResults=1"

    try:
        response = requests.get(GOOGLE_BOOKS_API_URL).json()
        if "items" in response and len(response["items"]) > 0:
            info = response["items"][0]["volumeInfo"]
            identifiers = info.get("industryIdentifiers", [])
            found_isbn = next(
                (i["identifier"] for i in identifiers if i["type"] == "ISBN_13"), None
            )

            if not found_isbn:
                found_isbn = next(
                    (i["identifier"] for i in identifiers if i["type"] == "ISBN_10"),
                    "N/A",
                )

            return {
                "found_title": info.get("title", "N/A"),
                "found_subtitle": info.get("subtitle", "N/A"),
                "found_authors": info.get("authors", ["N/A"]),
                "found_year": info.get("publishedDate", "N/A")[:4],
                "found_pages": info.get("pageCount", "N/A"),
                "found_isbn": found_isbn,
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
    prompt = _generate_prompt(input_json)

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_,
            messages=[
                {
                    "role": Role.SYSTEM,
                    "content": "You are a JSON-only API. Never add explanations.",
                },
                {"role": Role.USER, "content": prompt},
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


def _generate_prompt(input_json):
    prompt = f"""
     You are a Data Cleaning Assistant. Your goal is to produce a CLEAN dataset by
     comparing "original_data" (potentially messy) with "retrieved_context"
     (Google Books API ground truth) to produce a fxinal validated record.

     ## YOUR TASKS (Follow these 7 steps strictly):
     --- FILL MISSING VALUES ---
         1. **Publication Year**: If 'Year' is missing in original_data, fill it using 'retrieved_context'.
         2. **Authors**: Fill using 'retrieved_context'. **IMPORTANT**: If the API provides a list (e.g., ["Name A", "Name B"]), join them into a single string separated by commas.
         3. **Page Count**: Fill using 'retrieved_context'. Ensure it is an Integer.

     --- CORRECT ERRORS ---
         4. **Author Correction**: Compare original vs. retrieved. If the original Author has typos or is incorrect, overwrite it with the 'retrieved_context' author.
         5. **Code (ISBN) Correction**: Check the 'ISBN'. If it looks invalid or differs from the 'found_isbn' in 'retrieved_context', use the retrieved ISBN.
         6. **Series Extraction**: Check 'found_subtitle' or 'found_title'. Only extract if it clearly looks like a series (e.g., contains "Vol", "Book 1", "Trilogy"). Ignore generic descriptions like "A Novel".

     --- CLASSIFICATION ---
        **7. Sales Classification**: Calculate "Sales" based on 'found_ratings_count' (int):
            - "High Sales": if there are **more than 100** ratings.
            - "Medium Sales": if there are **more than 10 but 100 or fewer** ratings.
            - "Low Sales": if there are **10 or fewer** ratings (or if ratings are missing).

     ## INPUT DATA:
     {input_json}

     ## OUTPUT Format:
     - Return a JSON LIST of objects.
         - Keys must be: "Title", "Author", "Year" (int), "ISBN" (string), "Series", "Pages" (int), "Sales".
         - No Markdown. No Code Blocks. Just the raw JSON string.
     """

    return prompt


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

        print(
            f"Limiting processing to the first {NUM_ROWS_TO_PROCESSED} rows for testing..."
        )
        df = df.head(NUM_ROWS_TO_PROCESSED).copy()

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

            final_df["Year"] = pd.to_numeric(final_df["Year"], errors="coerce")
            final_df["Pages"] = pd.to_numeric(final_df["Pages"], errors="coerce")

            final_df["Year"] = final_df["Year"].astype("Int64")
            final_df["Pages"] = final_df["Pages"].astype("Int64")

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
