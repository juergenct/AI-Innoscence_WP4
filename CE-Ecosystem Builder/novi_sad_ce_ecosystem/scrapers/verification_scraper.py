from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from scrapegraphai.graphs import SmartScraperGraph

from novi_sad_ce_ecosystem.config.verification_prompts import VERIFICATION_PROMPT
from novi_sad_ce_ecosystem.models.entity import EntityVerification
from novi_sad_ce_ecosystem.utils.cache import FileCache
from novi_sad_ce_ecosystem.utils.logging_setup import setup_logging
from novi_sad_ce_ecosystem.utils.impressum import analyze_impressum
from novi_sad_ce_ecosystem.utils.ollama_structured import call_ollama_chat, VerificationResult
from novi_sad_ce_ecosystem.utils.web_fetcher import fetch_website_content

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_novi_sad_based": {"type": "boolean"},
        "novi_sad_confidence": {"type": "number"},
        "novi_sad_evidence": {"type": "string"},
        "is_ce_related": {"type": "boolean"},
        "ce_confidence": {"type": "number"},
        "ce_evidence": {"type": "string"}
    },
    "required": ["is_novi_sad_based", "novi_sad_confidence", "novi_sad_evidence", "is_ce_related", "ce_confidence", "ce_evidence"]
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

    def verify_entity(self, url: str) -> EntityVerification:
        cache_key = f"verification::{url}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return EntityVerification(
                    url=url,
                    is_novi_sad_based=bool(cached.get('is_novi_sad_based', False)),
                    novi_sad_confidence=float(cached.get('novi_sad_confidence', 0.0)),
                    is_ce_related=bool(cached.get('is_ce_related', False)),
                    ce_confidence=float(cached.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Novi Sad: {cached.get('novi_sad_evidence','')} | CE: {cached.get('ce_evidence','')}",
                    should_extract=bool(cached.get('is_novi_sad_based') and cached.get('is_ce_related')),
                )
            except Exception:
                pass

        # TWO-STAGE LLM APPROACH: ScrapegraphAI extraction + Ollama structuring
        
        # Stage 1: ScrapegraphAI intelligently extracts relevant information (no strict JSON required)
        scrapegraph_prompt = """Extract ONLY information that is actually on this website. Do not generate or assume information.
Be CONCISE - extract key facts only, not full paragraphs.

LOCATION (check homepage, contact, kontakt, o nama):
- Full address (street, postal code, city) - extract exact text
- Phone numbers (copy exactly as shown)
- City and district names (Novi Sad/Нови Сад, Petrovaradin/Петроварадин, Futog/Футог, Veternik/Ветерник, Sremska Kamenica/Сремска Каменица)

CIRCULAR ECONOMY (check homepage, about, services, sustainability, održivost, usluge, reciklaža):
- List exact CE keywords found: kružna ekonomija, cirkularна ekonomija, reciklaža, održivost, otpad, upravljanje otpadom, recycling, circular economy, sustainability, zero waste, resource efficiency
- Brief description of CE activities (1-2 sentences max)
- CE certifications or memberships (names only)

ORGANIZATION:
- Official name (exact text)
- Type (company/university/NGO/etc.)

Visit multiple pages to find this information. Return ONLY what is actually on the website. Keep responses SHORT."""
        
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
        # Use faster 7b model for simple verification task (3x speedup)
        model = self.config.get('verification', {}).get('model', self.config.get('llm', {}).get('model', 'llama3.1:8b')).replace('ollama/', '')
        base_url = self.config.get('llm', {}).get('base_url', 'http://localhost:11434')
        
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Ollama + Pydantic: structure the extracted text into perfect JSON
                # Use max_tokens=3000 to prevent truncation (increased from 2500)
                pydantic_result = call_ollama_chat(
                    prompt=self.create_verification_prompt(),
                    text_content=extracted_text,  # ← From ScrapegraphAI
                    response_model=VerificationResult,
                    model=model,
                    base_url=base_url,
                    temperature=0.0,
                    max_tokens=3000
                )
                
                # Pydantic guarantees perfect structure!
                result = pydantic_result.model_dump()

                # If Novi Sad uncertain, try Impressum to boost
                ns_conf = float(result.get('novi_sad_confidence', 0.0) or 0.0)
                ns_based = bool(result.get('is_novi_sad_based', False))
                if ns_conf < 0.6 or not ns_based:
                    imp = analyze_impressum(url)
                    if imp.get('found'):
                        boost = max(ns_conf, float(imp.get('novi_sad_confidence') or 0.0))
                        result['novi_sad_confidence'] = boost
                        if boost >= 0.6:
                            result['is_novi_sad_based'] = True
                            # add evidence
                            ev = result.get('novi_sad_evidence', '')
                            result['novi_sad_evidence'] = (ev + ' | ' if ev else '') + str(imp.get('evidence'))

                self.cache.set(cache_key, result)
                verification = EntityVerification(
                    url=url,
                    is_novi_sad_based=bool(result.get('is_novi_sad_based', False)),
                    novi_sad_confidence=float(result.get('novi_sad_confidence', 0.0)),
                    is_ce_related=bool(result.get('is_ce_related', False)),
                    ce_confidence=float(result.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Novi Sad: {result.get('novi_sad_evidence','')} | CE: {result.get('ce_evidence','')}",
                    should_extract=bool(result.get('is_novi_sad_based') and result.get('is_ce_related')),
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
