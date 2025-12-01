## 🧹 LLM-Augmented Dataset Cleaning Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline to clean, standardise, and enrich a raw book dataset using a local Large Language Model (LLM) and data retrieved from the Google Books API. The pipeline ensures high-quality, ground-truth data by first retrieving external context and then instructing the LLM to perform cleaning, correction, and classification tasks based on that context.

## ✨ Features

- RAG Integration: Augments LLM instructions with real-time data from the Google Books API.
- Data Standardization: Corrects common data issues (typos, missing values) in book titles, authors, and metadata (Year, Pages).
- Sales Classification: Classifies book sales into `Low`, `Medium` or `High` based on the `ratingsCount` field retrieved from Google Books.
- Batch Processing: Processes large datasets efficiently using configurable batch sizes.
- LLM Metrics: Includes a rough FLOPs (Floating Point Operations) estimation to track computational load.
- Local LLM Support: Designed to run with local models served via APIs like LM Studio or Ollama.

## ⚙️ Prerequisites

Before running the script, ensure you have the following installed:

- `Python 3.8+`
- `LLM Server`: A running local LLM server (e.g., LM Studio) that exposes an OpenAI-compatible API endpoint.

## 🌐 Getting Started
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 main.py
```

## 🛠️ How the Pipeline Works

The `process_batch_with_rag` function orchestrates the three main steps for each batch:

- Retrieval (Google Books API) The script queries the Google Books API using the original Title and Author as hints. It retrieves ground-truth data like the verified title, authors, published year, page count, and, crucially, the ratings count.
- Augmentation and Cleaning (LLM) The retrieved data is packaged alongside the original data into a single JSON object. This combined JSON is sent to the LLM with a detailed prompt (System Instruction and User Query).
- The LLM is instructed to:
  - Correct Title and Author using the retrieved context.
  - Fill missing Year and Pages.
  - Classify Sales based on `found_ratings_count`:
    - Less than 10 ratings = `Low Sales`
    - More than 10 ratings = `Medium Sales`
    - More than 100 ratings = `High Sales`
  - Output ONLY a clean JSON array with the specified schema `Title, Author, Year, ISBN, Series, Pages, Sales`

## 👨‍💻 Acknowledgement

Developed by [Joshua David](https://joshydavid.com)

<a href="https://joshydavid.com">
  <img src="https://github.com/user-attachments/assets/4dfe0c89-8ced-4e08-bcf3-6261bdbb956d" width="80">
</a>
