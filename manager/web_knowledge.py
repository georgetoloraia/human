from __future__ import annotations

from bs4 import BeautifulSoup


def extract_plain_text(html: str, max_chars: int = 4000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text[:max_chars]
