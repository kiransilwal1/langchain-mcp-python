import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


class URLExtractor:
    def __init__(self):
        self.session = requests.Session()

    def extract_urls(self, url: str) -> list[str]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        print(soup)
        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
        urls = []

        for link in soup.find_all("a", href=True):
            full_url = urljoin(base_url, link["href"])
            urls.append(full_url)

        # Optional: Remove duplicates
        return list(set(urls))
