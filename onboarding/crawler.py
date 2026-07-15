"""
Website crawler for corpus ingestion.

Uses requests + BeautifulSoup to crawl a website's pages and extract
clean text content.  crawl4ai is imported as an optional richer alternative.
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BusinessChatbotCrawler/1.0; "
        "+https://github.com/your-org/business-ai-chatbot)"
    )
}


def crawl_website(
    start_url: str,
    max_pages: int = 50,
    delay: float = 0.5,
    timeout: int = 15,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[dict]:
    """
    Crawl `start_url` up to `max_pages` pages.

    Returns a list of dicts:
      [{"url": str, "title": str, "text": str}, ...]
    """
    base = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
    visited: set[str] = set()
    queue: list[str] = [start_url]
    results: list[dict] = []

    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)

    while queue and len(results) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Crawl failed for %s: %s", url, exc)
            continue

        if "text/html" not in resp.headers.get("Content-Type", ""):
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract text
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url
        text = soup.get_text(separator="\n", strip=True)
        text = _clean_text(text)

        if len(text) > 100:  # skip near-empty pages
            results.append({"url": url, "title": title, "text": text})
            logger.info("Crawled [%d/%d]: %s", len(results), max_pages, url)

        # Enqueue same-domain links
        for link in soup.find_all("a", href=True):
            href = urljoin(base, link["href"])
            parsed = urlparse(href)
            # Same domain only, no fragments
            if parsed.netloc == urlparse(base).netloc and parsed.fragment == "":
                href_clean = href.split("#")[0].rstrip("/")
                if href_clean not in visited:
                    if _matches(href_clean, include_patterns, exclude_patterns):
                        queue.append(href_clean)

        time.sleep(delay)

    logger.info("Crawl complete: %d pages collected from %s.", len(results), start_url)
    return results


def _clean_text(text: str) -> str:
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _matches(url: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    if exclude:
        for pattern in exclude:
            if re.search(pattern, url):
                return False
    if include:
        return any(re.search(p, url) for p in include)
    return True
