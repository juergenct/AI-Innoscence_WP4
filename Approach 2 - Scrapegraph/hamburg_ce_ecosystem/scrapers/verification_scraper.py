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
from hamburg_ce_ecosystem.utils.ollama_structured import call_ollama_chat, VerificationResult
from hamburg_ce_ecosystem.utils.web_fetcher import fetch_website_content

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_hamburg_based": {"type": "boolean"},
        "hamburg_confidence": {"type": "number"},
        "hamburg_evidence": {"type": "string"},
        "is_ce_related": {"type": "boolean"},
        "ce_confidence": {"type": "number"},
        "ce_evidence": {"type": "string"}
    },
    "required": ["is_hamburg_based", "hamburg_confidence", "hamburg_evidence", "is_ce_related", "ce_confidence", "ce_evidence"]
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

        # TWO-STAGE LLM APPROACH: ScrapegraphAI extraction + Ollama structuring
        
        # Stage 1: ScrapegraphAI intelligently extracts relevant information (no strict JSON required)
        scrapegraph_prompt = """Extract ONLY information that is actually on this website. Do not generate or assume information.

LOCATION (check homepage, contact, impressum):
- Full address (street, postal code, city) - extract exact text
- Phone numbers (copy exactly as shown)
- City and district names (Hamburg, Altona, Eimsbüttel, Wandsbek, Bergedorf, Harburg)

CIRCULAR ECONOMY (check homepage, about, services, sustainability):
- Copy exact keywords found: Kreislaufwirtschaft, Recycling, Nachhaltigkeit, Zero Waste, Ressourceneffizienz
- Copy descriptions of: circular economy, waste management, reuse, repair, remanufacturing activities
- Copy any CE certifications or memberships mentioned
- Copy sustainability project names

ORGANIZATION:
- Official name (exact text)
- Organization type indicators (company, university, NGO, etc.)

Visit multiple pages to find this information. Return ONLY what is actually on the website."""
        
        graph = SmartScraperGraph(
            prompt=scrapegraph_prompt,
            source=url,
            config=self.config
        )
        
        try:
            scrapegraph_output = graph.run()
            # Convert to string - we don't care about format here
            if isinstance(scrapegraph_output, dict):
                extracted_text = json.dumps(scrapegraph_output, ensure_ascii=False, indent=2)
            else:
                extracted_text = str(scrapegraph_output)
        except Exception as e:
            # ERROR RECOVERY: Extract useful content from ScrapegraphAI's error message
            # (Following pattern from GitHub issues #809, #324, #257)
            error_msg = str(e)
            
            # Suppress the error from appearing in logs by using INFO level
            # ScrapegraphAI often puts the actual LLM output in the error message
            if "Invalid json output:" in error_msg:
                # Extract the content after "Invalid json output:"
                parts = error_msg.split("Invalid json output:", 1)
                if len(parts) > 1:
                    extracted_text = parts[1].strip()
                    # Remove "For troubleshooting..." footer
                    if "For troubleshooting" in extracted_text:
                        extracted_text = extracted_text.split("For troubleshooting")[0].strip()
                    # Silent recovery - logged at DEBUG level only
                    self.logger.debug(f"Recovered content from ScrapegraphAI error for {url}")
                else:
                    # No recoverable content, use Playwright fallback
                    extracted_text = fetch_website_content(url, max_pages=3, timeout=30000)
                    self.logger.debug(f"Using Playwright fallback for {url}")
            else:
                # Different error, use Playwright fallback
                extracted_text = fetch_website_content(url, max_pages=3, timeout=30000)
                self.logger.debug(f"Using Playwright fallback for {url}")
        
        # Stage 2: Ollama + Pydantic structures the extracted information
        model = self.config.get('llm', {}).get('model', 'llama3.1:8b').replace('ollama/', '')
        base_url = self.config.get('llm', {}).get('base_url', 'http://localhost:11434')
        
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Ollama + Pydantic: structure the extracted text into perfect JSON
                pydantic_result = call_ollama_chat(
                    prompt=self.create_verification_prompt(),
                    text_content=extracted_text,  # ← From ScrapegraphAI
                    response_model=VerificationResult,
                    model=model,
                    base_url=base_url,
                    temperature=0.0
                )
                
                # Pydantic guarantees perfect structure!
                result = pydantic_result.model_dump()

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
