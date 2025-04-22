import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple


class WebScraper:
    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        timeout: int = 30,
        skip_tags: List[str] | None = None,
        min_text_length: int = 20,
    ):
        """
        Initialize the web scraper with specified parameters.

        Args:
            user_agent: User agent string to use for requests
            timeout: Request timeout in seconds
            skip_tags: HTML tags to skip when extracting text
            min_text_length: Minimum length of text blocks to keep
        """
        self.headers = {"User-Agent": user_agent}
        self.timeout = timeout
        self.skip_tags = skip_tags or [
            "script",
            "style",
            "noscript",
            "iframe",
            "head",
            "meta",
            "link",
        ]
        self.min_text_length = min_text_length

    def validate_url(self, url: str) -> bool:
        """
        Validate if the given string is a proper URL.

        Args:
            url: URL string to validate

        Returns:
            Boolean indicating if URL is valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in [
                "http",
                "https",
            ]
        except:
            return False

    def fetch_html(self, url: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Fetch HTML content from the specified URL.

        Args:
            url: URL to fetch

        Returns:
            Tuple of (HTML content, metadata) or (None, error_info) if failed
        """
        if not self.validate_url(url):
            return None, {"error": "Invalid URL format"}

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # Extract metadata
            metadata = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "title": "",  # Will be populated during parsing
            }

            return response.text, metadata

        except requests.exceptions.RequestException as e:
            return None, {"error": f"Request error: {str(e)}"}

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Replace multiple whitespace characters with a single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def extract_text_from_html(self, html: str) -> Tuple[str, str]:
        """
        Extract plain text from HTML content.

        Args:
            html: HTML content as string

        Returns:
            Tuple of (extracted text, page title)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Get page title
        title = ""
        if soup.title and soup.title.string:
            title = self.clean_text(soup.title.string)

        # Remove unwanted tags
        for tag in self.skip_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract text blocks and filter by minimum length
        text_blocks = []
        for paragraph in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            text = self.clean_text(paragraph.get_text())
            if len(text) >= self.min_text_length:
                text_blocks.append(text)

        # If we don't have enough text blocks, try getting all text
        if not text_blocks:
            body_text = self.clean_text(soup.get_text())
            # Split on multiple newlines to try to get logical blocks
            text_blocks = [
                block.strip()
                for block in re.split(r"\n\s*\n", body_text)
                if len(block.strip()) >= self.min_text_length
            ]

        return "\n\n".join(text_blocks), title

    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape and extract text content from the specified URL.

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing extracted text, metadata, and status
        """
        html, metadata = self.fetch_html(url)

        if html is None:
            return {"success": False, "text": "", "metadata": metadata}

        text, title = self.extract_text_from_html(html)

        # Update metadata with title
        if isinstance(metadata, dict):
            metadata["title"] = title

        return {"success": True, "text": text, "metadata": metadata}
