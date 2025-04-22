import os
import io
import requests
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from pypdf import PdfReader


class PDFTextExtractor:
    """
    A class that extracts text from PDF files located either locally or at a URL using pypdf.
    """

    def __init__(self, source: str):
        """
        Initialize the PDFTextExtractor.

        Args:
            source: Path to a local PDF file or URL to a PDF file
        """
        self.source = source
        self._is_url = self._check_if_url(source)
        self._pdf_reader = None
        self._stream = None

    def _check_if_url(self, source: str) -> bool:
        """
        Check if the provided source is a URL.

        Args:
            source: Path or URL to check

        Returns:
            True if source is a URL, False otherwise
        """
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
                # Download PDF from URL
                response = requests.get(self.source, stream=True)
                response.raise_for_status()  # Raise exception for HTTP errors

                # Create a file-like object from the response content
                self._stream = io.BytesIO(response.content)
                self._pdf_reader = PdfReader(self._stream)
            else:
                # Open local PDF file
                self._pdf_reader = PdfReader(self.source)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading PDF from URL: {str(e)}")
        except FileNotFoundError:
            raise Exception(f"PDF file not found: {self.source}")
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")

    def extract_text(self) -> str:
        """
        Extract all text from the PDF.

        Returns:
            String containing all text extracted from the PDF
        """
        try:
            self._load_pdf()

            # Extract text from all pages
            text = ""
            for page_num in range(len(self._pdf_reader.pages)):
                page = self._pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text + "\n"

            return text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_by_page(self) -> List[str]:
        """
        Extract text from the PDF page by page.

        Returns:
            List of strings, where each string contains text from one page
        """
        try:
            self._load_pdf()

            # Extract text from each page separately
            pages_text = []
            for page_num in range(len(self._pdf_reader.pages)):
                page = self._pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                pages_text.append(page_text)

            return pages_text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def get_page_count(self) -> int:
        """
        Get the number of pages in the PDF.

        Returns:
            Integer representing the number of pages
        """
        self._load_pdf()
        return len(self._pdf_reader.pages)

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the PDF.

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            self._load_pdf()
            return self._pdf_reader.metadata

        except Exception as e:
            raise Exception(f"Error extracting metadata from PDF: {str(e)}")

    def extract_images(self, output_dir: str = "./pdf_images") -> List[str]:
        """
        Extract images from the PDF.

        Args:
            output_dir: Directory where images will be saved

        Returns:
            List of paths to the extracted images
        """
        try:
            self._load_pdf()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            image_paths = []
            image_count = 0

            for page_num, page in enumerate(self._pdf_reader.pages):
                for image_file_object in page.images:
                    # Extract image data
                    image_name = image_file_object.name
                    image_data = image_file_object.data

                    # Determine file extension from image name or default to .png
                    file_extension = (
                        os.path.splitext(image_name)[1].lower()
                        if "." in image_name
                        else ".png"
                    )
                    if not file_extension:
                        file_extension = ".png"

                    # Create output path
                    image_path = os.path.join(
                        output_dir,
                        f"page{page_num + 1}_img{image_count}{file_extension}",
                    )

                    # Save image
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_data)

                    # Add to list of extracted images
                    image_paths.append(image_path)
                    image_count += 1

            return image_paths

        except Exception as e:
            raise Exception(f"Error extracting images: {str(e)}")

    def __del__(self):
        """
        Clean up resources when the object is deleted.
        """
        # Close the stream if it was created
        if self._stream is not None:
            try:
                self._stream.close()
            except:
                pass


# Example usage
# if __name__ == "__main__":
#     # Example with a URL
#     url = "https://www.immigration.govt.nz/documents/amendment-circulars/amendment-circular-2024-14.pdf"
#     pdf_extractor = PDFTextExtractor(url)
#
#     try:
#         # Get full text
#         text = pdf_extractor.extract_text()
#         print(f"Extracted {len(text)} characters of text")
#         print(f"First 200 characters: {text[:200]}...")
#
#         # Get page count
#         page_count = pdf_extractor.get_page_count()
#         print(f"PDF has {page_count} pages")
#
#         # Get text by page
#         pages_text = pdf_extractor.extract_text_by_page()
#         print(f"First page has {len(pages_text[0])} characters")
#
#         # Get metadata
#         metadata = pdf_extractor.extract_metadata()
#         print(f"PDF metadata: {metadata}")
#
#         # Uncomment to extract images
#         # images = pdf_extractor.extract_images()
#         # print(f"Extracted {len(images)} images")
#
#     except Exception as e:
#         print(f"Error: {str(e)}")

# Example with a local file
# local_pdf = "/path/to/local/document.pdf"
# local_extractor = PDFTextExtractor(local_pdf)
# local_text = local_extractor.extract_text()
