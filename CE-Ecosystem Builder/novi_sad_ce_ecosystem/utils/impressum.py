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
        # Serbian and English contact page keywords
        if any(k in text for k in ["impressum", "imprint", "kontakt", "contact", "o nama", "about"]):
            href = a.get("href")
            if href:
                return urljoin(home_url, href)

    # Try common contact/about page paths
    for path in ["/impressum", "/imprint", "/kontakt", "/contact", "/o-nama", "/about", "/en/contact", "/sr/kontakt"]:
        candidate = urljoin(home_url, path)
        html2 = _fetch(candidate)
        if html2 and len(html2) > 200:
            return candidate

    return None


def analyze_impressum(home_url: str) -> Dict[str, object]:
    url = find_impressum_url(home_url)
    if not url:
        return {"found": False, "novi_sad_confidence": 0.0, "evidence": ""}

    html = _fetch(url)
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(" ", strip=True)
    lower = text.lower()

    # Novi Sad postal codes: 21000-21999
    postal_hit = re.search(r"\b21[0-9]{3}\b", lower) is not None
    # Novi Sad and its districts/municipalities
    name_hit = any(n in lower for n in [
        "novi sad", "нови сад", "petrovaradin", "петроварадин",
        "futog", "футог", "veternik", "ветерник", "sremska kamenica",
        "сремска каменица", "begeč", "бегеч", "serbia", "србија", "srbija"
    ])
    # Serbian country code +381 and Novi Sad area code 21 (or 021)
    phone_hit = "+381 21" in lower or re.search(r"\b021\b", lower) is not None or "+381 (0)21" in lower

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
        "novi_sad_confidence": conf,
        "evidence": f"Impressum[{', '.join(parts)}] {url}",
    }
