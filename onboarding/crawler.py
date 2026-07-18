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


def extract_business_workflow(pages: list[dict], groq_api_key: str) -> str:
    """
    Analyzes crawled website pages to extract business workflow, contact methods, and booking procedures.
    """
    if not groq_api_key or not pages:
        return ""
        
    try:
        from chatbot.llm import chat_completion, FAST_MODEL
        
        # Combine text from up to 5 important pages to save tokens
        text_corpus = "\n\n".join(
            [f"--- Page: {p['url']} ---\n{p['text'][:2000]}" for p in pages[:5]]
        )
        
        system_prompt = (
            "You are a business process analyst extracting contact and booking workflows from website copy.\n"
            "Analyze the provided website text and extract the exact instructions a customer service representative "
            "should give a user to book an appointment or contact the business.\n"
            "Format your response as a clear, step-by-step instruction manual for an AI chatbot.\n"
            "Example:\n"
            "'To book an appointment, instruct the user to call 555-0192 or visit https://example.com/book. "
            "If they have an emergency, tell them to call immediately.'\n\n"
            "If no clear booking workflow is found, suggest directing them to the extracted phone number or email, or a general contact page."
        )
        
        response = chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Website Content:\n\n{text_corpus}"}
            ],
            model=FAST_MODEL,
            api_key=groq_api_key,
            temperature=0.1
        )
        return response.strip()
    except Exception as exc:
        logger.error("Failed to extract business workflow: %s", exc)
        return ""


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
