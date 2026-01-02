#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract websites for Public Service Agency data (Moldova) for entities located in Cahul.

This script reads three Excel workbooks (companies, NGOs, public institutions),
filters rows to entities located in Cahul, and attempts to discover the official
website for each entity via a conservative search-and-verify pipeline.

First run supports sampling the first N entities (default: 10) to validate the approach.

Outputs (written next to this script by default):
- psa_websites_resolved_sample.csv
- psa_websites_unresolved_sample.csv
- psa_websites_cache.json (search and verify cache)

Requirements in environment: pandas, requests, beautifulsoup4, duckduckgo_search
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
	# Preferred import per package rename
	from ddgs import DDGS  # type: ignore
except Exception:
	try:
		# Backward compatibility
		from duckduckgo_search import DDGS  # type: ignore
	except Exception:  # pragma: no cover
		DDGS = None  # will handle gracefully

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT

EXCEL_PATHS = [
	INPUT_DIR / "Public Service Agency Raw Data asp.gov.md" / "2025.01.11-apc-apl.xlsx",
	INPUT_DIR / "Public Service Agency Raw Data asp.gov.md" / "company.xlsx",
	INPUT_DIR / "Public Service Agency Raw Data asp.gov.md" / "RSON.xlsx",
]

RESOLVED_OUT = INPUT_DIR / "psa_websites_resolved.csv"
UNRESOLVED_OUT = INPUT_DIR / "psa_websites_unresolved.csv"
CACHE_PATH = INPUT_DIR / "psa_websites_cache.json"

HEADERS = {
	"User-Agent": (
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
		"(KHTML, like Gecko) Chrome/121 Safari/537.36"
	)
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10

SOCIAL_DOMAINS = (
	"facebook.com",
	"instagram.com",
	"linkedin.com",
	"twitter.com",
	"x.com",
	"youtube.com",
	"youtu.be",
	"t.me",
	"tiktok.com",
	"vk.com",
	"ok.ru",
)

FREE_MAIL_DOMAINS = (
	"gmail.com",
	"yahoo.com",
	"yandex.ru",
	"mail.ru",
	"outlook.com",
	"icloud.com",
	"hotmail.com",
)

# obvious non-official domains to ignore in search results
BLOCKLIST_DOMAINS = (
	"stackoverflow.com",
	"stackexchange.com",
	"canva.com",
	"play.google.com",
	"apps.apple.com",
	"support.google.com",
	"microsoft.com",
)

# Business directories - accept as fallback (two-tier strategy)
DIRECTORY_DOMAINS = (
	"infobiz.md",
	"idno.md",
	"bizzer.md",
	"yellowpages.md",
	"bizvizor.md",
	"bizgu.md",
	"companii.md",
	"bizfinder.md",
)

LEGAL_SUFFIXES = [
	# Romanian/Moldovan legal forms and common descriptors
	r"\bSRL\b",
	r"\bS\.R\.L\b",
	r"\bSA\b",
	r"\bS\.A\b",
	r"\bÎI\b",
	r"\bII\b",
	r"\bÎ\.I\b",
	r"\bAO\b",
	r"\bONG\b",
	r"\bIM\b",
	r"\bÎ\.M\b",
	r"\bIP\b",
	r"\bInstitut(ul|ia)?\b",
	r"\bAgenția\b",
	r"\bCompania\b",
	r"\bSocietatea\b",
	r"\bGrupul\b",
]
LEGAL_SUFFIX_RE = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Common multi-word legal/entity forms to strip from names before searching
# Ordered from longest to shortest for proper matching
LEGAL_PHRASES = [
	# Longest phrases first
	"societatea cu răspundere limitată",
	"societatea cu raspundere limitata",
	"societate cu răspundere limitată",
	"societate cu raspundere limitata",
	"cu răspundere limitată",
	"cu raspundere limitata",
	"întreprinderea municipală",
	"intreprinderea municipala",
	"întreprinderea individuală",
	"intreprinderea individuala",
	"întreprinzător individual",
	"intreprinzator individual",
	"asociația proprietarilor",
	"asociatia proprietarilor",
	"societatea comercială",
	"societatea comerciala",
	"societatea pe acțiuni",
	"societatea pe actiuni",
	"cooperativa de consum",
	"gospodăria țărănească",
	"gospodaria taraneasca",
	# Shorter phrases
	"întreprinderea",
	"intreprinderea",
	"pe acțiuni",
	"pe actiuni",
	"comercială",
	"comerciala",
	"societatea",
	"societate",
	"compania",
	"î.i.",
	"i.i.",
]

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def strip_accents(text: str) -> str:
	if not isinstance(text, str):
		return ""
	return "".join(
		c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
	)


def normalize_name(name: str) -> str:
	name = strip_accents(name or "").lower()
	# remove common multi-word legal phrases first
	for phrase in LEGAL_PHRASES:
		if phrase in name:
			name = name.replace(phrase, " ")
	# remove legal suffixes
	name = LEGAL_SUFFIX_RE.sub(" ", name)
	# remove quotes and punctuation
	name = re.sub(r"[\"'`“”„«»]+", " ", name)
	name = re.sub(r"[^\w\s-]", " ", name)
	name = re.sub(r"\s+", " ", name).strip()
	return name


def tokenize(name: str) -> List[str]:
	toks = [t for t in re.split(r"\W+", name) if len(t) >= 3]
	# remove very common words
	stop = {"compania", "grupul", "agentia", "institutul", "national", "centrul", "unitatea"}
	return [t for t in toks if t not in stop]


def is_social(url: str) -> bool:
	try:
		host = urlparse(url).netloc.lower()
	except Exception:
		return False
	return any(dom in host for dom in SOCIAL_DOMAINS)


def is_directory(url: str) -> bool:
	"""Check if URL is a business directory (two-tier strategy fallback)"""
	try:
		host = urlparse(url).netloc.lower()
	except Exception:
		return False
	return any(dom in host for dom in DIRECTORY_DOMAINS)


def is_external(url: str) -> bool:
	try:
		host = urlparse(url).netloc
		return bool(host)
	except Exception:
		return False


def canonical_home(url: str) -> str:
	url = url.strip()
	if not url.lower().startswith(("http://", "https://")):
		url = ("https://" + url) if url.lower().startswith("www.") else ("http://" + url)
	return url


def fetch_html(url: str) -> Tuple[int, str]:
	try:
		r = SESSION.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True)
		return r.status_code, r.text if isinstance(r.text, str) else ""
	except Exception:
		return 0, ""


def html_signals_score(html: str, name_tokens: List[str]) -> int:
	if not html:
		return 0
	try:
		soup = BeautifulSoup(html, "lxml")
	except Exception:
		return 0
	title = (soup.title.text if soup.title else "").lower()
	h1 = (soup.find("h1").get_text(" ") if soup.find("h1") else "").lower()
	body = soup.get_text(" ", strip=True).lower()

	score = 0
	# name overlap
	overlap = sum(1 for t in name_tokens if t in title or t in h1)
	score += 20 * min(overlap, 2)

	# Cahul mention
	if "cahul" in body:
		score += 15

	# penalize extremely short titles
	if len(title) < 4:
		score -= 5

	return score


# ----------------------------------------------------------------------------
# Search + scoring
# ----------------------------------------------------------------------------


def ddg_search_candidates(queries: List[str], k: int = 8) -> List[Dict]:
	if DDGS is None:
		return []
	results: List[Dict] = []

	# Exponential backoff for rate limiting (Fix #4)
	max_retries = 3
	base_delay = 5.0  # Start with 5 seconds

	with DDGS() as ddgs:  # type: ignore
		for q in queries:
			retry_count = 0
			while retry_count <= max_retries:
				try:
					for item in ddgs.text(q, max_results=k, safesearch="off", region="wt-wt"):
						# item: {title, href, body}
						results.append(item)
					# Increased delay to avoid rate limiting (Fix #1)
					time.sleep(2.0 + random.random() * 3.0)  # 2-5 seconds
					break  # Success - move to next query
				except Exception as e:
					error_str = str(e).lower()
					# Check for 429 rate limit error
					if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
						retry_count += 1
						if retry_count <= max_retries:
							# Exponential backoff: 5s, 10s, 20s
							delay = base_delay * (2 ** (retry_count - 1))
							logging.warning(f"Rate limited (429), retrying in {delay}s (attempt {retry_count}/{max_retries})")
							time.sleep(delay)
						else:
							logging.error(f"Rate limited after {max_retries} retries, skipping query: {q}")
							break
					else:
						# Other error - skip this query
						break
	return results


def score_candidate(candidate: Dict, norm_name: str, name_tokens: List[str]) -> int:
	url = candidate.get("href") or ""
	title = strip_accents((candidate.get("title") or "").lower())
	body = strip_accents((candidate.get("body") or "").lower())

	if not url or not is_external(url) or is_social(url):
		return -100

	host = urlparse(url).netloc.lower()
	if any(b in host for b in BLOCKLIST_DOMAINS):
		return -100
	score = 0

	# Prefer .md
	if host.endswith(".md"):
		score += 20

	# Name token overlap
	overlap = sum(1 for t in name_tokens if t in title or t in body)
	score += 15 * min(overlap, 2)

	# Host token overlap with normalized name tokens
	host_base = host.split(":")[0]
	if any(t in host_base for t in name_tokens):
		score += 25

	# Snippet mentions Cahul
	if "cahul" in body:
		score += 10

	# Penalize directories/news
	if any(x in url.lower() for x in ["facebook.com", "/news/", "/stiri/", "/articles/", "/item/"]):
		score -= 15

	return score


# ----------------------------------------------------------------------------
# Loading Excel and filtering
# ----------------------------------------------------------------------------


def load_all_workbooks(paths: List[Path]) -> Dict[Tuple[str, str], pd.DataFrame]:
	data: Dict[Tuple[str, str], pd.DataFrame] = {}
	for p in paths:
		if not p.exists():
			logging.warning("Missing workbook: %s", p)
			continue
		try:
			sheets = pd.read_excel(p, sheet_name=None, dtype=str)
		except Exception as e:  # pragma: no cover
			logging.error("Failed reading %s: %s", p, e)
			continue
		for sheet, df in sheets.items():
			# Attempt to detect and fix mis-parsed headers (when the real header is in the first row)
			fixed_df = coerce_header_row(df)
			data[(str(p), str(sheet))] = fixed_df
	return data


def coerce_header_row(df: pd.DataFrame) -> pd.DataFrame:
	if df is None or df.empty:
		return df
	cols = list(df.columns)
	unnamed_frac = sum(1 for c in cols if isinstance(c, str) and c.lower().startswith("unnamed")) / max(len(cols), 1)
	first = df.iloc[0].astype(str).fillna("")
	header_tokens = ["denumirea", "idno", "cod fiscal", "adresa", "cuatm", "licentiate", "nelicentiate"]
	header_like = sum(1 for v in first.values if any(tok in v.lower() for tok in header_tokens))
	if unnamed_frac >= 0.5 and header_like >= 2:
		new_cols = [str(v).strip() if str(v).strip() else str(c) for v, c in zip(first.values, cols)]
		new_df = df.iloc[1:].copy()
		new_df.columns = new_cols
		return new_df
	return df


def pick_name_column(df: pd.DataFrame) -> Optional[str]:
	# 1) direct exact matches
	exact = [
		"denumire", "denumirea", "nume", "name", "titlu", "entitate", "institutie",
		"denumire juridica", "denumire completa", "denumire_completa",
		"denumirea companiei", "denumirea entitatii", "denumirea organizației", "denumirea organizatiei",
	]
	lower_cols = {c.lower(): c for c in df.columns}
	for key in exact:
		if key in lower_cols:
			return lower_cols[key]

	# 2) fuzzy substring matches
	substrs = ["denumir", "nume", "titlu", "entitat", "institut", "organiza", "compani", "firma"]
	for c in df.columns:
		lc = c.lower()
		if "unnamed" in lc:
			continue
		if any(s in lc for s in substrs):
			return c

	# 3) heuristic: choose the column with the best "name-like" score
	def col_score(series: pd.Series) -> float:
		if series.dtype != object:
			return -1
		s = series.dropna().astype(str)
		if s.empty:
			return -1
		# exclude numeric-only columns
		letters_frac = (s.str.contains(r"[A-Za-zĂÂÎȘŢăâîșțşţ]", regex=True)).mean()
		digits_frac = (s.str.match(r"^[0-9\-\.\s/]+$")).mean()
		avg_len = s.str.len().mean()
		avg_tokens = s.str.split().map(len).mean()
		return letters_frac * 2 + (avg_tokens > 1) + (avg_len > 8) - digits_frac

	# filter out unnamed and obviously id-like columns
	eligible_cols = []
	for c in df.columns:
		lc = c.lower()
		if lc.startswith("unnamed"):
			continue
		if df[c].dtype != object:
			continue
		s = df[c].dropna().astype(str)
		if not s.empty:
			digit_ratio = (s.str.match(r"^[0-9\-\.\s/]+$")).mean()
			if digit_ratio >= 0.5:
				continue
		eligible_cols.append(c)

	scored = [(c, col_score(df[c])) for c in eligible_cols]
	scored.sort(key=lambda x: x[1], reverse=True)
	if scored and scored[0][1] > 0:
		return scored[0][0]

	# 4) fallback: first string column with any letters
	for c in df.columns:
		if df[c].dtype == object:
			s = df[c].dropna().astype(str)
			if not s.empty and (s.str.contains(r"[A-Za-z]", regex=True)).any():
				return c
	return None


def is_cahul_row(row: pd.Series) -> bool:
	# true if any string cell contains 'cahul'
	for v in row.values.tolist():
		if not isinstance(v, str):
			continue
		txt = strip_accents(v).lower()
		if "cahul" in txt:
			return True
	return False


def should_skip_entity_type(row: pd.Series) -> bool:
	"""
	Returns True if this entity is a small business type unlikely to have a website.
	Skips: Întreprindere individuală, Gospodărie țărănească, Asociația Proprietarilor
	"""
	# Check all string values in the row for entity type indicators
	row_text = " ".join([str(v) for v in row.values.tolist() if isinstance(v, str)]).lower()
	row_text = strip_accents(row_text)

	# Small business types to skip (normalized, accents removed)
	skip_patterns = [
		"intreprindere individuala",
		"intreprinderea individuala",  # With definite article
		"intreprinzator individual",
		"intreprinzatorul individual",  # With definite article
		"gospodaria taraneasca",
		"gospodarie",  # Shorter form
		"asociatia proprietarilor",
		"asociatie proprietari",  # Without "lor"
	]

	for pattern in skip_patterns:
		if pattern in row_text:
			return True

	return False


def iter_cahul_entities(
	data: Dict[Tuple[str, str], pd.DataFrame],
	only_files: Optional[set[str]] = None,
	only_sheets: Optional[set[str]] = None,
	exclude_phrases: Optional[List[str]] = None,
):
	exclude_norm = []
	if exclude_phrases:
		exclude_norm = [strip_accents(p).lower().strip() for p in exclude_phrases if p]
	for (path, sheet), df in data.items():
		base = os.path.basename(path)
		if only_files and base not in only_files:
			continue
		if only_sheets and sheet not in only_sheets:
			continue
		if df is None or df.empty:
			continue
		name_col = pick_name_column(df)
		if not name_col:
			continue
		# ensure string
		sdf = df.copy()
		for c in sdf.columns:
			if sdf[c].dtype != object:
				sdf[c] = sdf[c].astype(str)
		# filter rows to Cahul
		mask = sdf.apply(is_cahul_row, axis=1)
		sdf = sdf[mask]
		logging.info("Sheet '%s' from '%s': using name column '%s' with %d Cahul rows", sheet, os.path.basename(path), name_col, len(sdf))

		# Filter out small business types
		pre_filter_count = len(sdf)
		entity_type_mask = ~sdf.apply(should_skip_entity_type, axis=1)
		sdf = sdf[entity_type_mask]
		skipped_count = pre_filter_count - len(sdf)
		if skipped_count > 0:
			logging.info("Skipped %d small business entities (Întreprindere individuală, Gospodărie, etc.)", skipped_count)

		for _, row in sdf.iterrows():
			raw_name = (row.get(name_col) or "").strip()
			if not raw_name:
				continue
			if exclude_norm:
				raw_norm = strip_accents(raw_name).lower()
				if any(p in raw_norm for p in exclude_norm):
					continue
			address_raw = " ".join([str(x) for x in row.values.tolist() if isinstance(x, str)])[:500]
			yield {
				"source_file": path,
				"sheet_name": sheet,
				"raw_name": raw_name,
				"address_raw": address_raw,
			}


# ----------------------------------------------------------------------------
# Resolver with caching
# ----------------------------------------------------------------------------


class WebsiteResolver:
	def __init__(self, cache_path: Path = CACHE_PATH):
		self.cache_path = cache_path
		self.cache: Dict[str, Dict] = {}
		if cache_path.exists():
			try:
				self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
			except Exception:
				self.cache = {}

	def save(self):
		try:
			self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception:  # pragma: no cover
			pass

	def key(self, name: str) -> str:
		return normalize_name(name)

	def resolve(self, name: str) -> Dict:
		k = self.key(name)
		if k in self.cache:
			return self.cache[k]

		norm = normalize_name(name)
		tokens = tokenize(norm)
		if not tokens:
			result = {"website": "", "domain": "", "confidence": 0, "decision": "not-found", "method": "search", "notes": "no tokens"}
			self.cache[k] = result
			return result

		queries = [
			f"{' '.join(tokens)} Cahul site:.md",
			f"{' '.join(tokens)} Cahul Moldova",
			f"{' '.join(tokens)} municipiul Cahul Moldova",
		]
		candidates = ddg_search_candidates(queries, k=8)

		scored: List[Tuple[int, Dict]] = []
		for c in candidates:
			s = score_candidate(c, norm, tokens)
			if s > -50:
				scored.append((s, c))
		scored.sort(reverse=True, key=lambda x: x[0])

		best = None
		best_score = -999
		verify_bonus = 0
		for s, c in scored[:5]:
			url = c.get("href") or ""
			if not url:
				continue
			status, html = fetch_html(url)
			if status >= 200:
				verify_bonus = html_signals_score(html, tokens)
			total = s + verify_bonus
			if total > best_score:
				best = c
				best_score = total
			# early stop if strong
			if total >= 70:
				break

		if not best:
			result = {"website": "", "domain": "", "confidence": 0, "decision": "not-found", "method": "search", "notes": "no suitable candidates"}
			self.cache[k] = result
			return result

		url = best.get("href") or ""
		if url and not url.lower().startswith(("http://", "https://")):
			url = canonical_home(url)
		domain = urlparse(url).netloc.lower() if url else ""

		# Two-tier resolution strategy (Fix #3, #6)
		# Tier 1: Official websites (high confidence)
		# Tier 2: Business directories (fallback)
		is_dir = is_directory(url)

		if is_dir:
			# Directory entry - accept as fallback if score is reasonable
			if best_score >= 30:  # Lowered threshold for directories (Fix #6)
				decision = "directory"  # Mark as directory for manual review
			else:
				decision = "not-found"
		else:
			# Official website - higher threshold
			if best_score >= 70:
				decision = "accepted"
			elif best_score >= 40:  # Lowered threshold (Fix #6)
				decision = "review-needed"
			else:
				decision = "not-found"

		result = {
			"website": url,
			"domain": domain,
			"confidence": int(best_score),
			"decision": decision,
			"method": "search",
			"notes": f"title={best.get('title','')} score={best_score} {'[directory]' if is_dir else ''}",
		}
		self.cache[k] = result
		return result


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def run(
	limit: Optional[int] = None,
	only_files: Optional[List[str]] = None,
	only_sheets: Optional[List[str]] = None,
	exclude_phrases: Optional[List[str]] = None,
):
	logging.info("Loading workbooks ...")
	data = load_all_workbooks(EXCEL_PATHS)

	resolver = WebsiteResolver(CACHE_PATH)

	resolved_rows: List[Dict] = []
	unresolved_rows: List[Dict] = []

	count = 0
	only_files_set = set(only_files) if only_files else None
	only_sheets_set = set(only_sheets) if only_sheets else None
	for ent in iter_cahul_entities(
		data,
		only_files=only_files_set,
		only_sheets=only_sheets_set,
		exclude_phrases=exclude_phrases,
	):
		if limit is not None and count >= limit:
			break
		count += 1
		raw_name = ent["raw_name"]
		res = resolver.resolve(raw_name)
		row = {
			"source_file": ent["source_file"],
			"sheet_name": ent["sheet_name"],
			"raw_name": raw_name,
			"normalized_name": normalize_name(raw_name),
			"address_raw": ent["address_raw"],
			"website": res.get("website", ""),
			"domain": res.get("domain", ""),
			"confidence": res.get("confidence", 0),
			"decision": res.get("decision", "not-found"),
			"method": res.get("method", ""),
			"notes": res.get("notes", ""),
		}
		if row["decision"] == "accepted":
			resolved_rows.append(row)
		else:
			unresolved_rows.append(row)
		# Increased delay to avoid rate limiting (Fix #1)
		time.sleep(2.0 + random.random() * 3.0)  # 2-5 seconds

	# Write outputs
	if resolved_rows:
		pd.DataFrame(resolved_rows).to_csv(RESOLVED_OUT, index=False)
	if unresolved_rows:
		pd.DataFrame(unresolved_rows).to_csv(UNRESOLVED_OUT, index=False)

	resolver.save()

	print(f"Processed entities: {count}")
	print(f"Resolved: {len(resolved_rows)} | Unresolved: {len(unresolved_rows)}")
	print(f"Resolved CSV: {RESOLVED_OUT if resolved_rows else 'n/a'}")
	print(f"Unresolved CSV: {UNRESOLVED_OUT if unresolved_rows else 'n/a'}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Resolve websites for PSA Cahul entities")
	parser.add_argument("--limit", type=int, default=None, help="Number of entities to process (default: all)")
	parser.add_argument("--only-file", action="append", dest="only_files", help="Only process this workbook file (basename). Can be provided multiple times.")
	parser.add_argument("--only-sheet", action="append", dest="only_sheets", help="Only process this sheet name. Can be provided multiple times.")
	parser.add_argument(
		"--exclude-phrase",
		action="append",
		dest="exclude_phrases",
		help="Skip entities whose name contains this phrase (case-insensitive, accents ignored). Can be repeated.",
	)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	run(
		limit=args.limit,
		only_files=args.only_files,
		only_sheets=args.only_sheets,
		exclude_phrases=args.exclude_phrases,
	)

