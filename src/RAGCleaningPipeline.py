import json
import math
import time

import pandas as pd
from openai import OpenAI

from constants import (
    BATCH_SIZE,
    NUM_ROWS_TO_PROCESSED,
    OUTPUT_FILENAME,
    Role,
)
from GoogleBooksAPI import GoogleBooksAPI


# =============================
# === RAG Cleaning Pipeline ===
# =============================
class RAGCleaningPipeline:
    """
    Manages the entire RAG-based data cleaning pipeline.
    """

    def __init__(
        self, google_books_api_url: str, llm_api_base_url: str, llm_model: str
    ):
        self.api_client = GoogleBooksAPI(google_books_api_url)
        self.llm_client = OpenAI(base_url=llm_api_base_url, api_key="ollama")
        self.llm_model = llm_model
        self.total_time = 0
        self.all_results = []

    def _generate_prompt(self, input_json: str) -> str:
        """
        Generates the LLM prompt.
        """
        prompt = f"""
            You are a Data Cleaning Assistant. Your goal is to produce a CLEAN dataset by
            comparing "original_data" (potentially messy) with "retrieved_context"
            (Google Books API ground truth) to produce a final validated record.

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
                - Your response MUST be a JSON array (list) of objects.
                    - Title: str
                    - Author: str
                    - Year: int
                    - ISBN: str
                    - Series: str
                    - Pages: int
                    - Sales: str
                    - id: str
                - No Markdown. No Code Blocks. Just the raw JSON string.
            """
        return prompt

    def _retrieve_book_context(self, batch_df):
        """
        Step 1: Retrieve ground-truth book info from Google Books API for the batch.
        """

        batch_context = []
        print(f" > Retrieving context for {len(batch_df)} books...")
        for idx, row in batch_df.iterrows():
            external_info = self.api_client.fetch_book_context(
                row["Title"], row["Author"]
            )

            content = {
                "id": idx,
                "original_data": row[
                    ["Title", "Author", "Year", "ISBN", "Series", "Pages"]
                ].to_dict(),
                "retrieved_context": external_info,
            }
            batch_context.append(content)
        return batch_context

    def _clean_batch_with_llm(self, batch_context):
        """
        Step 2: Send RAG prompt to the local LLM and parse the cleaned JSON response.
        """

        input_json = json.dumps(batch_context, indent=2, ensure_ascii=False)
        prompt = self._generate_prompt(input_json)

        start_time = time.time()
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": Role.SYSTEM.value,
                        "content": "You are a JSON-only API. Never add explanations.",
                    },
                    {"role": Role.USER.value, "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2500,
            )
            content = response.choices[0].message.content.strip()
            duration = time.time() - start_time
            clean_json = content.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean_json)
            return data, duration

        except Exception as e:
            print(f" > LLM call failed: {e}")
            return [], 0

    def _calculate_accuracy(self, cleaned_data, context_data):
        """
        Step 3: Compares the LLM's cleaned output against the Ground Truth context.
        """
        context_map = {
            item["id"]: item["retrieved_context"]
            for item in context_data
            if isinstance(item["retrieved_context"], dict)
            and "error" not in item["retrieved_context"]
        }

        results_with_metrics = []

        for item in cleaned_data:
            current_id = item.get("id")
            gt = context_map.get(current_id)

            if gt and isinstance(gt, dict):
                # Author Accuracy
                llm_authors_raw = item.get("Author")
                gt_authors_raw = gt.get("found_authors")
                is_author_match = False

                if llm_authors_raw and isinstance(gt_authors_raw, list):
                    llm_authors_set = set(
                        name.strip().lower()
                        for name in str(llm_authors_raw).split(",")
                        if name.strip()
                    )

                    gt_authors_set = set(
                        name.strip().lower() for name in gt_authors_raw
                    )
                    is_author_match = llm_authors_set == gt_authors_set

                item["Accuracy_Author"] = "Match" if is_author_match else "Mismatch"

                # Year Accuracy
                llm_year = str(item.get("Year")).strip()
                gt_year = str(gt.get("found_year")).strip()
                item["Accuracy_Year"] = "Match" if llm_year == gt_year else "Mismatch"

                # ISBN Accuracy
                llm_isbn = str(item.get("ISBN")).strip().replace("-", "")
                gt_isbn = str(gt.get("found_isbn")).strip().replace("-", "")
                item["Accuracy_ISBN"] = "Match" if llm_isbn == gt_isbn else "Mismatch"

                # Pages Accuracy
                llm_pages = str(item.get("Pages")).strip()
                gt_pages = str(gt.get("found_pages")).strip()
                item["Accuracy_Pages"] = (
                    "Match" if llm_pages == gt_pages else "Mismatch"
                )
            else:
                missing_context = "N/A"
                item["Accuracy_Author"] = missing_context
                item["Accuracy_Year"] = missing_context
                item["Accuracy_ISBN"] = missing_context
                item["Accuracy_Pages"] = missing_context

            results_with_metrics.append(item)

        return results_with_metrics

    def process_batch_with_rag(self, batch_df: pd.DataFrame) -> None:
        """
        Orchestrates the RAG pipeline for a single batch.
        """
        # 1. Retrieve Context (Input for LLM)
        batch_context = self._retrieve_book_context(batch_df)

        # 2. Clean Data with LLM
        results, duration = self._clean_batch_with_llm(batch_context)
        self.total_time += duration

        # 3. Calculate Accuracy
        if results:
            results_with_accuracy = self._calculate_accuracy(results, batch_context)
            self.all_results.extend(results_with_accuracy)
            print(f"   > ✅ Success. {len(results)} items cleaned in {duration:.2f}s")
        else:
            print("   > ⚠️ Batch failed. Using original data as fallback.")
            self.all_results.extend(batch_df.to_dict(orient="records"))

    def run_pipeline(self, df: pd.DataFrame):
        """
        Main execution loop for the entire dataset.
        """

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

        total_batches = math.ceil(len(df) / BATCH_SIZE)
        print(f"Starting processing of {len(df)} rows in {total_batches} batches...")

        for i in range(total_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch = df.iloc[start_idx:end_idx]

            print(
                f"\n--- Batch {i + 1}/{total_batches} (Rows {start_idx}-{end_idx}) ---"
            )
            self.process_batch_with_rag(batch)
            time.sleep(1)

        self._finalize_and_save()

    def _finalize_and_save(self):
        """
        Handles final data cleanup, printing summary, and saving.
        """

        if not self.all_results:
            print("\n❌ No results generated.")
            return

        final_df = pd.DataFrame(self.all_results)
        final_df["Year"] = pd.to_numeric(final_df["Year"], errors="coerce").astype(
            "Int64"
        )
        final_df["Pages"] = pd.to_numeric(final_df["Pages"], errors="coerce").astype(
            "Int64"
        )
        final_df = final_df.drop(columns=["id"], errors="ignore")

        print("\n" + "=" * 40)
        print("       FINAL DATA PREVIEW")
        print("=" * 40 + "\n")
        print(final_df.head(10).to_string(index=False))

        accuracy_cols = [col for col in final_df.columns if col.startswith("Accuracy_")]
        if accuracy_cols:
            total_records = len(final_df)
            print("\n" + "=" * 40)
            print("       ACCURACY SUMMARY")
            print("=" * 40)
            for col in accuracy_cols:
                match_count = (final_df[col] == "Match").sum()
                accuracy_percent = (match_count / total_records) * 100
                metric_name = col.replace("Accuracy_", "")
                print(
                    f" {metric_name:<15}: {match_count} / {total_records} = {accuracy_percent:.2f}%"
                )
            print("-" * 40)

        print(f"Total Processing Time: {self.total_time:.2f}s")
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ Saved cleaned dataset to: {OUTPUT_FILENAME}")
