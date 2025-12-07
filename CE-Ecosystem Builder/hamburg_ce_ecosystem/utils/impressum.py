from __future__ import annotations

from typing import Optional, Dict
from urllib.parse import urljoin

import re
import requests
from bs4 import BeautifulSoup


def _fetch(url: str, timeout: int = 25) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""


def find_impressum_url(home_url: str) -> Optional[str]:
    html = _fetch(home_url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a"):
        text = (a.get_text(strip=True) or "").lower()
        if any(k in text for k in ["impressum", "imprint"]):
            href = a.get("href")
            if href:
                return urljoin(home_url, href)

    for path in ["/impressum", "/imprint", "/de/impressum", "/en/imprint"]:
        candidate = urljoin(home_url, path)
        html2 = _fetch(candidate)
        if html2 and len(html2) > 200:
            return candidate

    return None


def analyze_impressum(home_url: str) -> Dict[str, object]:
    url = find_impressum_url(home_url)
    if not url:
        return {"found": False, "hamburg_confidence": 0.0, "evidence": ""}

    html = _fetch(url)
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    lower = text.lower()

    postal_hit = re.search(r"\b2[0-2][0-9]{3}\b", lower) is not None
    name_hit = any(n in lower for n in [
        "hamburg", "altona", "eimsbüttel", "wandsbek", "bergedorf", "harburg",
    ])
    phone_hit = "+49 40" in lower or re.search(r"\b040\b", lower) is not None

    conf = min(1.0, (postal_hit + name_hit + phone_hit) / 3.0)
    parts = []
    if postal_hit:
        parts.append("postal")
    if name_hit:
        parts.append("name")
    if phone_hit:
        parts.append("phone")

    return {
        "found": True,
        "hamburg_confidence": conf,
        "evidence": f"Impressum[{', '.join(parts)}] {url}",
    }
