import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple

from playwright.sync_api import sync_playwright


class WebScraper:
    def __init__(
        self,
        timeout: int = 30,
        skip_tags: Optional[List[str]] = None,
        min_text_length: int = 20,
    ):
        self.timeout = timeout * 1000  # ms for Playwright
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
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in [
                "http",
                "https",
            ]
        except:
            return False

    def fetch_html(self, url: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not self.validate_url(url):
            return None, {"error": "Invalid URL format"}

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                response = page.goto(url, timeout=self.timeout)

                page.wait_for_load_state("networkidle")

                html = page.content()

                metadata = {
                    "url": url,
                    "status_code": response.status if response else 0,
                    "content_type": response.headers.get("content-type", "")
                    if response
                    else "",
                    "last_modified": response.headers.get("last-modified", "")
                    if response
                    else "",
                    "title": page.title(),
                }

                browser.close()
                return html, metadata

        except Exception as e:
            return None, {"error": str(e)}

    def clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def extract_text_from_html(self, html: str) -> Tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags
        for tag in self.skip_tags:
            for element in soup.find_all(tag):
                element.decompose()

        text_blocks = []
        for el in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            text = self.clean_text(el.get_text())
            if len(text) >= self.min_text_length:
                text_blocks.append(text)

        if not text_blocks:
            full_text = self.clean_text(soup.get_text())
            text_blocks = [
                block.strip()
                for block in re.split(r"\n\s*\n", full_text)
                if len(block.strip()) >= self.min_text_length
            ]

        title = soup.title.string.strip() if soup.title else ""
        return "\n\n".join(text_blocks), title

    def scrape(self, url: str) -> Dict[str, Any]:
        html, metadata = self.fetch_html(url)

        if html is None:
            return {"success": False, "text": "", "metadata": metadata}

        text, title = self.extract_text_from_html(html)

        if isinstance(metadata, dict):
            metadata["title"] = title

        return {"success": True, "text": text, "metadata": metadata}
