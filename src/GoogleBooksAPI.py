import requests


# ==============================
# === GOOGLE BOOKS API Class ===
# ==============================
class GoogleBooksAPI:
    """
    Encapsulates logic for Google Books API.
    """

    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url

    def fetch_book_context(self, title: str, author_hint: str) -> dict:
        """
        Searches Google Books API to find the 'Ground Truth' for a book.
        """

        if not isinstance(title, str):
            return {"error": "No Title Provided"}

        clean_title = title.split("(")[0].strip()
        query = f"intitle:{clean_title}"
        if isinstance(author_hint, str):
            clean_author = author_hint.split(",")[0].strip()
            query += f"+inauthor:{clean_author}"

        api_url = f"{self.api_base_url}?q={query}&maxResults=1"

        try:
            response = requests.get(api_url).json()
            if "items" in response and len(response["items"]) > 0:
                info = response["items"][0]["volumeInfo"]
                identifiers = info.get("industryIdentifiers", [])

                # Retrieve ISBN
                found_isbn = next(
                    (i["identifier"] for i in identifiers if i["type"] == "ISBN_13"),
                    None,
                )
                if not found_isbn:
                    found_isbn = next(
                        (
                            i["identifier"]
                            for i in identifiers
                            if i["type"] == "ISBN_10"
                        ),
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
            return {"error": "Book not found in external database."}
        except Exception as e:
            return {"error": f"API Error: {str(e)}"}
