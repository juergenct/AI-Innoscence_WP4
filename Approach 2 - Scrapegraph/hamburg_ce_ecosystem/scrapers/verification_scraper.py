from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import re
import requests
import yaml
from bs4 import BeautifulSoup
from scrapegraphai.graphs import SmartScraperGraph

from hamburg_ce_ecosystem.config.prompts import VERIFICATION_PROMPT
from hamburg_ce_ecosystem.models.entity import EntityVerification
from hamburg_ce_ecosystem.utils.cache import FileCache
from hamburg_ce_ecosystem.utils.logging_setup import setup_logging
from hamburg_ce_ecosystem.utils.impressum import analyze_impressum

VERIFICATION_SCHEMA = {
    "is_hamburg_based": "boolean",
    "hamburg_confidence": "number",
    "hamburg_evidence": "string",
    "is_ce_related": "boolean",
    "ce_confidence": "number",
    "ce_evidence": "string"
}


class VerificationScraper:
    def __init__(self, config_path: str | Path, cache_dir: str | Path | None = None, logger: logging.Logger | None = None):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.cache = FileCache(cache_dir or (self.config_path.parent.parent / 'data' / '.cache' / 'verification'))
        self.logger = logger or setup_logging(self.config_path.parent.parent / 'logs' / 'scraping_errors.log', console_level=logging.ERROR)
        self.max_retries: int = int(self.config.get('scraper', {}).get('max_retries', 3))

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_verification_prompt() -> str:
        return VERIFICATION_PROMPT

    @staticmethod
    def _coerce_result(raw: Any) -> Dict[str, Any]:
        """Extract JSON from response, handling various formats."""
        if isinstance(raw, dict):
            if 'content' in raw:
                inner = raw['content']
                if isinstance(inner, dict):
                    return inner
                if isinstance(inner, str):
                    raw = inner
            else:
                return raw
        
        if isinstance(raw, str):
            s = raw.strip()
            
            # Remove <think> tags
            s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
            
            # Remove code fences
            if s.startswith('```'):
                s = re.sub(r'^```[a-z]*\n?', '', s)
                s = re.sub(r'\n?```$', '', s)
            
            # Find first { and last }
            start = s.find('{')
            end = s.rfind('}')
            if start >= 0 and end > start:
                s = s[start:end+1]
            
            try:
                return json.loads(s)
            except Exception:
                # Try to find JSON in the text
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', s)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        pass
                return {}
        
        return {}

    @staticmethod
    def _heuristic_verify(url: str) -> EntityVerification:
        ua = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }
        try:
            resp = requests.get(url, headers=ua, timeout=25)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            html = ""
        soup = BeautifulSoup(html or "", "html.parser")
        text = soup.get_text(" ", strip=True).lower()

        postal_hit = bool(re.search(r"\b2[0-2][0-9]{3}\b", text))
        name_hit = any(n in text for n in [
            "hamburg", "altona", "eimsbüttel", "wandsbek", "bergedorf", "harburg",
        ])
        phone_hit = "+49 40" in text or re.search(r"\b040\b", text) is not None
        is_hamburg = postal_hit or name_hit or phone_hit
        hamburg_conf = min(1.0, (postal_hit + name_hit + phone_hit) / 3.0)

        ce_terms = [
            "kreislaufwirtschaft", "circular economy", "recycling", "zero waste",
            "nachhaltigkeit", "ressourceneffizienz", "wiederverwendung", "reuse",
            "waste management", "sustainability", "repair", "remanufacturing"
        ]
        ce_hits = sum(1 for t in ce_terms if t in text)
        is_ce = ce_hits > 0
        ce_conf = 0.2 if ce_hits == 1 else (0.5 if ce_hits == 2 else (0.8 if ce_hits >= 3 else 0.0))

        evidence_hh = "; ".join([
            "postal" if postal_hit else "",
            "name" if name_hit else "",
            "phone" if phone_hit else "",
        ]).strip("; ,")
        evidence_ce = ", ".join([t for t in ce_terms if t in text][:6])

        return EntityVerification(
            url=url,
            is_hamburg_based=bool(is_hamburg),
            hamburg_confidence=float(hamburg_conf),
            is_ce_related=bool(is_ce),
            ce_confidence=float(ce_conf),
            verification_reasoning=f"Hamburg: {evidence_hh} | CE: {evidence_ce}",
            should_extract=bool(is_hamburg and is_ce),
        )

    def verify_entity(self, url: str) -> EntityVerification:
        cache_key = f"verification::{url}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return EntityVerification(
                    url=url,
                    is_hamburg_based=bool(cached.get('is_hamburg_based', False)),
                    hamburg_confidence=float(cached.get('hamburg_confidence', 0.0)),
                    is_ce_related=bool(cached.get('is_ce_related', False)),
                    ce_confidence=float(cached.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Hamburg: {cached.get('hamburg_evidence','')} | CE: {cached.get('ce_evidence','')}",
                    should_extract=bool(cached.get('is_hamburg_based') and cached.get('is_ce_related')),
                )
            except Exception:
                pass

        graph = SmartScraperGraph(
            prompt=self.create_verification_prompt(),
            source=url,
            config=self.config,
            schema=VERIFICATION_SCHEMA
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = graph.run()
                result = self._coerce_result(raw)
                
                # Validate required keys exist
                required_keys = ['is_hamburg_based', 'hamburg_confidence', 'is_ce_related', 'ce_confidence']
                if not result or not all(k in result for k in required_keys):
                    raise ValueError(f'Missing required keys in result: {result}')

                # If Hamburg uncertain, try Impressum to boost
                hh_conf = float(result.get('hamburg_confidence', 0.0) or 0.0)
                hh_based = bool(result.get('is_hamburg_based', False))
                if hh_conf < 0.6 or not hh_based:
                    imp = analyze_impressum(url)
                    if imp.get('found'):
                        boost = max(hh_conf, float(imp.get('hamburg_confidence') or 0.0))
                        result['hamburg_confidence'] = boost
                        if boost >= 0.6:
                            result['is_hamburg_based'] = True
                            # add evidence
                            ev = result.get('hamburg_evidence', '')
                            result['hamburg_evidence'] = (ev + ' | ' if ev else '') + str(imp.get('evidence'))

                self.cache.set(cache_key, result)
                verification = EntityVerification(
                    url=url,
                    is_hamburg_based=bool(result.get('is_hamburg_based', False)),
                    hamburg_confidence=float(result.get('hamburg_confidence', 0.0)),
                    is_ce_related=bool(result.get('is_ce_related', False)),
                    ce_confidence=float(result.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Hamburg: {result.get('hamburg_evidence','')} | CE: {result.get('ce_evidence','')}",
                    should_extract=bool(result.get('is_hamburg_based') and result.get('is_ce_related')),
                )
                return verification
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Verification attempt {attempt}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)
        
        # NO FALLBACK - raise exception to force LLM to work
        error_msg = f"LLM verification failed after {self.max_retries} attempts for {url}: {last_exc}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
