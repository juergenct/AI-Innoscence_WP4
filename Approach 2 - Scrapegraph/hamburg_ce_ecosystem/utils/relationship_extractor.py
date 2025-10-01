from __future__ import annotations

from bs4 import BeautifulSoup
from typing import List
import re


def extract_partner_names(html: str) -> List[str]:
    names: List[str] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Heuristic: look for sections mentioning 'Partner' and collect nearby link texts
        partner_sections = soup.find_all(string=re.compile(r"partner", re.I))
        for text_node in partner_sections:
            section = text_node.parent
            if not section:
                continue
            for anchor in section.find_all_next("a", limit=30):
                text = (anchor.get_text(strip=True) or "").strip()
                if 2 <= len(text) <= 120 and not re.search(r"@|http", text):
                    names.append(text)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for n in names:
            if n not in seen:
                unique.append(n)
                seen.add(n)
        return unique[:50]
    except Exception:
        return []
