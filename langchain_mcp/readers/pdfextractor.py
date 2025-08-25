import os
import io
import requests
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from pypdf import PdfReader


class PDFTextExtractor:
    """
    A class that extracts and normalizes text from PDF files located locally or at a URL using pypdf.
    """

    def __init__(self, source: str):
        """
        Initialize the PDFTextExtractor.

        Args:
            source: Path to a local PDF file or URL to a PDF file.
        """
        self.source = source
        self._is_url = self._check_if_url(source)
        self._pdf_reader = None
        self._stream = None

    def _check_if_url(self, source: str) -> bool:
        parsed = urlparse(source)
        return bool(parsed.scheme and parsed.netloc)

    def _load_pdf(self) -> None:
        """
        Load the PDF from the source (local file or URL).
        """
        if self._pdf_reader is not None:
            return

        try:
            if self._is_url:
                response = requests.get(self.source, stream=True)
                response.raise_for_status()
                self._stream = io.BytesIO(response.content)
                self._pdf_reader = PdfReader(self._stream)
            else:
                self._pdf_reader = PdfReader(self.source)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading PDF from URL: {str(e)}")
        except FileNotFoundError:
            raise Exception(f"PDF file not found: {self.source}")
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")

    def _normalize_text(self, raw_text: str) -> str:
        """
        Normalize extracted text by joining broken lines into paragraphs.

        Args:
            raw_text: Raw text with excessive line breaks.

        Returns:
            Cleaned-up paragraph-style text.
        """
        lines = raw_text.splitlines()
        paragraphs = []
        current_paragraph = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(stripped)

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        return "\n\n".join(paragraphs)

    def extract_text(self) -> str:
        """
        Extract and normalize text from the entire PDF.

        Returns:
            Normalized full text as a single string.
        """
        try:
            self._load_pdf()
            text = ""

            for page in self._pdf_reader.pages:
                raw_text = page.extract_text() or ""
                normalized = self._normalize_text(raw_text)
                text += normalized + "\n\n"

            return text.strip()

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_by_page(self) -> List[str]:
        """
        Extract normalized text from each page separately.

        Returns:
            List of paragraph-formatted text from each page.
        """
        try:
            self._load_pdf()
            pages_text = []

            for page in self._pdf_reader.pages:
                raw_text = page.extract_text() or ""
                normalized = self._normalize_text(raw_text)
                pages_text.append(normalized)

            return pages_text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def get_page_count(self) -> int:
        """
        Return the total number of pages in the PDF.
        """
        self._load_pdf()
        return len(self._pdf_reader.pages)

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the PDF.

        Returns:
            Dictionary of metadata fields and values.
        """
        try:
            self._load_pdf()
            return self._pdf_reader.metadata or {}
        except Exception as e:
            raise Exception(f"Error extracting metadata from PDF: {str(e)}")

    def __del__(self):
        """
        Clean up any open streams on deletion.
        """
        if self._stream is not None:
            try:
                self._stream.close()
            except:
                pass


# Example usage
if __name__ == "__main__":
    url = "https://antislaverylaw.ac.uk/wp-content/uploads/2019/08/The-Labour-Act-2017.pdf"
    pdf_extractor = PDFTextExtractor(url)

    try:
        full_text = pdf_extractor.extract_text()
        print(f"Extracted {len(full_text)} characters of normalized text\n")
        print(full_text[:500] + "\n...\n")

        page_count = pdf_extractor.get_page_count()
        print(f"Total pages: {page_count}")

        metadata = pdf_extractor.extract_metadata()
        print(f"Metadata: {metadata}")

    except Exception as e:
        print(f"Error: {str(e)}")
