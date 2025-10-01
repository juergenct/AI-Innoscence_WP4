"""
Name-to-Domain Resolver

Reads a CSV (single column: name/company_name) and resolves official websites
via web search + light verification. Outputs:
 - seeds.csv (column: website)
 - mapping.csv (company_name, domain, confidence, method, evidence_url, notes)

Backends:
 - DuckDuckGo HTML (default, no API key). Lightly parses search result links.
 - Bing Web Search (optional, requires BING_API_KEY env var).

Quality signals:
 - Name match (token overlap / substring)
 - Hamburg signals (word 'Hamburg', postal codes 20xxx–22xxx) in site/impressum
 - Presence of Impressum/Kontakt/Über uns pages
 - .de TLD preference

Input CSV columns supported:
 - name/company_name/company/firma/unternehmensname (required if website missing)
 - website/url/entity_url/domain (optional; if present for a row, search is skipped)

Usage:
  python -m circular_scraper.utils.name_to_domain_resolver \
      --input companies.csv \
      --out-seeds seeds.csv \
      --out-mapping mapping.csv \
      --city Hamburg --country DE \
      --max 50000 --concurrency 8 --min-confidence 0.70
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "de-DE,de;q=0.9,en;q=0.8"}


@dataclass
class ResolveResult:
    company_name: str
    domain: str
    confidence: float
    method: str
    evidence_url: str
    notes: str


DIRECTORY_HINTS = (
    "branchenbuch", "gelbeseiten", "11880", "dastelefonbuch",
    "wer-zu-wem", "yelp.", "tripadvisor.", "booking.", "kununu.",
    "stepstone.", "indeed.", "facebook.", "instagram.", "linkedin.", "xing."
)

HAMBURG_POSTAL_RE = re.compile(r"\b2[0-2]\d{3}\b")


def normalize_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\s+", " ", n)
    return n


def is_probably_directory(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in DIRECTORY_HINTS)


def is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return p.netloc.lower()
    except Exception:
        return ""


def ddg_search(query: str, max_results: int = 5, timeout: int = 10) -> List[str]:
    """DuckDuckGo HTML search (no API). Returns result URLs.
    We respect basic etiquette: one request per query.
    """
    params = {"q": query}
    try:
        resp = requests.get("https://duckduckgo.com/html/", params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        urls: List[str] = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if href and is_http_url(href):
                urls.append(href)
            if len(urls) >= max_results:
                break
        # Fallback selector if class changes
        if not urls:
            for a in soup.select("a"):
                href = a.get("href")
                if href and is_http_url(href) and "duckduckgo" not in href:
                    urls.append(href)
                if len(urls) >= max_results:
                    break
        return urls
    except Exception:
        return []


def fetch_text(url: str, timeout: int = 10) -> Tuple[str, str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.url, r.text
    except Exception:
        return url, ""


def find_impressum_or_kontakt(home_html: str, base_url: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(home_html, "lxml")
        candidates = []
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            text = (a.get_text() or "").lower()
            if any(key in text for key in ["impressum", "kontakt", "über uns", "ueber uns", "about"]):
                candidates.append(href)
        # Build absolute URLs
        from urllib.parse import urljoin
        for href in candidates:
            absu = urljoin(base_url, href)
            if is_http_url(absu):
                return absu
    except Exception:
        pass
    return None


def name_similarity_score(company: str, html: str, title: str = "") -> float:
    c = normalize_name(company)
    if not html and not title:
        return 0.0
    text = (title or "") + "\n" + (html or "")
    text_l = text.lower()
    # direct substring
    if c and c in text_l:
        return 0.9
    # token overlap (very simple)
    tokens_c = set(re.findall(r"[a-z0-9äöüß]+", c))
    tokens_t = set(re.findall(r"[a-z0-9äöüß]+", text_l))
    if not tokens_c:
        return 0.0
    overlap = len(tokens_c & tokens_t) / max(1, len(tokens_c))
    return min(0.85, overlap)


def hamburg_signal_score(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    if "hamburg" in t:
        score += 0.3
    if HAMBURG_POSTAL_RE.search(t):
        score += 0.3
    return min(score, 0.6)


def tld_score(domain: str) -> float:
    return 0.1 if domain.endswith(".de") else 0.0


def score_candidate(company: str, home_url: str, home_html: str, impressum_html: str) -> float:
    # Use <title>
    title = ""
    try:
        soup = BeautifulSoup(home_html or "", "lxml")
        title = (soup.title.string or "").strip() if soup.title else ""
    except Exception:
        title = ""
    s_name = name_similarity_score(company, home_html, title)
    s_hh = max(hamburg_signal_score(home_html), hamburg_signal_score(impressum_html))
    s_tld = tld_score(extract_domain(home_url))
    # Weighted
    score = 0.6 * s_name + 0.3 * s_hh + 0.1 * s_tld
    return round(score, 3)


def resolve_company(company: str, city: str, country: str, min_conf: float, timeout: int = 10) -> ResolveResult:
    q = f"{company} {city} offizielle webseite"
    urls = ddg_search(q, max_results=5, timeout=timeout)
    urls = [u for u in urls if is_http_url(u) and not is_probably_directory(u)]
    best: Tuple[str, float, str] = ("", 0.0, "")  # url, score, method
    for u in urls[:5]:
        home_url, home_html = fetch_text(u, timeout=timeout)
        if not home_html:
            continue
        imp_url = find_impressum_or_kontakt(home_html, home_url)
        imp_html = ""
        if imp_url:
            _, imp_html = fetch_text(imp_url, timeout=timeout)
        score = score_candidate(company, home_url, home_html, imp_html)
        if score > best[1]:
            best = (home_url, score, "ddg+verify")
        # small pause to be polite
        time.sleep(0.1)
    if best[1] >= min_conf:
        return ResolveResult(company, best[0], best[1], best[2], best[0], "")
    return ResolveResult(company, "", 0.0, "none", "", "low confidence")


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser(description="Resolve company names to official domains")
    p.add_argument("--input", required=True, help="CSV file with a single column: name/company_name")
    p.add_argument("--out-seeds", default="data/iter_seeds/seeds_from_names.csv", help="Output seeds CSV (website)")
    p.add_argument("--out-mapping", default="data/iter_seeds/name_to_domain_mapping.csv", help="Output mapping CSV")
    p.add_argument("--city", default="Hamburg")
    p.add_argument("--country", default="DE")
    p.add_argument("--min-confidence", type=float, default=0.70)
    p.add_argument("--max", type=int, default=50000)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return

    # Read rows (name + optional website)
    rows_in: List[Dict[str, str]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        # Try csv with headers; else plain lines
        try:
            snif = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
        except Exception:
            snif = False
        if snif:
            r = csv.DictReader(f)
            name_col = None
            site_col = None
            fns = list(r.fieldnames or [])
            for c in fns:
                c_l = (c or "").strip().lower()
                if not name_col and c_l in {"name", "company_name", "company", "firma", "unternehmensname"}:
                    name_col = c
                if not site_col and c_l in {"website", "url", "entity_url", "domain"}:
                    site_col = c
            if not name_col and fns:
                name_col = fns[0]
            for row in r:
                nm = (row.get(name_col) or "").strip() if name_col else ""
                site = (row.get(site_col) or "").strip() if site_col else ""
                rows_in.append({"name": nm, "website": site})
        else:
            f.seek(0)
            for line in f:
                nm = (line or "").strip()
                rows_in.append({"name": nm, "website": ""})

    rows_in = rows_in[: args.max]
    print(f"Loaded {len(rows_in)} rows")

    seeds_rows: List[Dict[str, str]] = []
    mapping_rows: List[Dict[str, str]] = []

    def task(rec: Dict[str, str]) -> ResolveResult:
        nm = rec.get("name", "").strip()
        site = rec.get("website", "").strip()
        # If website provided, validate lightly and accept with high confidence
        if site:
            if not is_http_url(site):
                site = "https://" + site
            # Fetch home to follow redirects and ensure it’s live
            final_url, html = fetch_text(site, timeout=8)
            if html:
                return ResolveResult(nm or final_url, final_url, 0.95, "provided", final_url, "")
            # if site not reachable, fall back to search if name exists
            if nm:
                return resolve_company(nm, args.city, args.country, args.min_confidence)
            return ResolveResult(nm or site, "", 0.0, "none", "", "provided unreachable")
        # Otherwise, resolve via search
        return resolve_company(nm, args.city, args.country, args.min_confidence)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(task, rec): rec for rec in rows_in}
        for fut in as_completed(futures):
            res: ResolveResult = fut.result()
            if res.domain:
                seeds_rows.append({"website": res.domain})
            mapping_rows.append({
                "company_name": res.company_name,
                "domain": res.domain,
                "confidence": f"{res.confidence:.3f}",
                "method": res.method,
                "evidence_url": res.evidence_url,
                "notes": res.notes,
            })

    # Deduplicate seeds by domain
    seen = set()
    unique_seeds: List[Dict[str, str]] = []
    for r in seeds_rows:
        d = r.get("website")
        if d and d not in seen:
            seen.add(d)
            unique_seeds.append(r)

    write_csv(Path(args.out_seeds), unique_seeds, ["website"])
    write_csv(Path(args.out_mapping), mapping_rows, ["company_name", "domain", "confidence", "method", "evidence_url", "notes"])
    print(f"Seeds written: {len(unique_seeds)} → {args.out_seeds}")
    print(f"Mapping written: {len(mapping_rows)} → {args.out_mapping}")


if __name__ == "__main__":
    main()


