from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

ALLOWED_DOMAINS = {
    "docs.python.org",
    "realpython.com",
    "developer.mozilla.org",
}


class WebSensor:
    """
    Controlled HTTP client:
      - domain whitelist
      - simple rate limiting
      - short timeout
    """

    def __init__(self, timeout: float = 5.0, rate_limit: float = 1.0) -> None:
        self.timeout = timeout
        self.rate_limit = rate_limit
        self._last_call = 0.0

    def fetch_text(self, url: str) -> str:
        domain = urlparse(url).netloc
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain not allowed: {domain}")

        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        self._last_call = time.time()
        return resp.text
