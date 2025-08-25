from typing import Callable, Any
from langchain_core.documents import Document

from langchain_mcp.readers.pdfextractor import PDFTextExtractor
from langchain_mcp.vector_store.vector import DocumentVectorizer


class PDFContextBuilder:
    """
    Ingests PDF files using PDFTextExtractor, and vectorizes them using DocumentVectorizer.
    """

    def __init__(
        self,
        pdf_extractor: PDFTextExtractor,
        vectorizer: DocumentVectorizer,
    ):
        self.pdf_extractor = pdf_extractor
        self.vectorizer = vectorizer

    def build_context(self, pdf_path: str):
        """
        Read PDF, and store them with summaries as metadata.
        """
        content = self.pdf_extractor.extract_text(pdf_path)

        documents = []

        if not content.strip():
            return  # skip empty files

        try:
            # Create metadata with file path
            metadata = {
                "file_path": pdf_path,
            }

            # Wrap as LangChain Document
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        except Exception as e:
            print(f"Skipping {pdf_path} due to error: {e}")

        # Add to vector store
        return self.vectorizer.add_documents(documents)
