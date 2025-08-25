from typing import Callable, Any
from langchain_core.documents import Document

from langchain_mcp.readers.scraper import WebScraper
from langchain_mcp.vector_store.vector import DocumentVectorizer


class WebContextBuilder:
    """
    Ingests web content using WebScraper, and vectorizes them using DocumentVectorizer.
    """

    def __init__(
        self,
        web_scraper: WebScraper,
        vectorizer: DocumentVectorizer,
    ):
        self.web_scraper = web_scraper
        self.vectorizer = vectorizer

    def build_context(self, url: str):
        """
        Read web content, and store them with summaries as metadata.
        """
        content = self.web_scraper.scrape(url)

        documents = []

        if not content["text"].strip():
            return  # skip empty content

        try:
            # Create metadata with file path
            metadata = {
                "url": url,
            }

            # Wrap as LangChain Document
            doc = Document(page_content=content["text"], metadata=metadata)
            documents.append(doc)

        except Exception as e:
            print(f"Skipping {url} due to error: {e}")

        # Add to vector store
        return self.vectorizer.add_documents(documents)
