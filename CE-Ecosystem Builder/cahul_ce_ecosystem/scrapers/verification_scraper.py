from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import yaml
from scrapegraphai.graphs import SmartScraperGraph

from cahul_ce_ecosystem.config.verification_prompts import VERIFICATION_PROMPT
from cahul_ce_ecosystem.models.entity import EntityVerification
from cahul_ce_ecosystem.utils.cache import FileCache
from cahul_ce_ecosystem.utils.logging_setup import setup_logging
from cahul_ce_ecosystem.utils.impressum import analyze_impressum
from cahul_ce_ecosystem.utils.ollama_structured import call_ollama_chat, VerificationResult
from cahul_ce_ecosystem.utils.web_fetcher import fetch_website_content

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_cahul_based": {"type": "boolean"},
        "cahul_confidence": {"type": "number"},
        "cahul_evidence": {"type": "string"},
        "is_ce_related": {"type": "boolean"},
        "ce_confidence": {"type": "number"},
        "ce_evidence": {"type": "string"}
    },
    "required": ["is_cahul_based", "cahul_confidence", "cahul_evidence", "is_ce_related", "ce_confidence", "ce_evidence"]
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
                    is_cahul_based=bool(cached.get('is_cahul_based', False)),
                    cahul_confidence=float(cached.get('cahul_confidence', 0.0)),
                    is_ce_related=bool(cached.get('is_ce_related', False)),
                    ce_confidence=float(cached.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Cahul: {cached.get('cahul_evidence','')} | CE: {cached.get('ce_evidence','')}",
                    should_extract=bool(cached.get('is_cahul_based') and cached.get('is_ce_related')),
                )
            except Exception:
                pass

        # TWO-STAGE LLM APPROACH: ScrapegraphAI extraction + Ollama structuring
        
        # Stage 1: ScrapegraphAI intelligently extracts relevant information (no strict JSON required)
        scrapegraph_prompt = """Extract ONLY information that is actually on this website. Do not generate or assume information.

LOCATION (check homepage, contact, legal info/informații juridice/юридическая информация):
- Full address (street, postal code, city) - extract exact text
- Phone numbers (copy exactly as shown)
- City and region names (Cahul, Congaz, Giurgiulești, Raionul Cahul, Moldova)

CIRCULAR ECONOMY (check homepage, about, services, sustainability/sustenabilitate/устойчивость):
- Copy exact keywords found: economie circulară, циркулярная экономика, circular economy, reciclare, переработка, recycling, sustenabilitate, устойчивость, sustainability, eficiența resurselor, ресурсоэффективность
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
        # Use faster 7b model for simple verification task (3x speedup)
        model = self.config.get('verification', {}).get('model', self.config.get('llm', {}).get('model', 'llama3.1:8b')).replace('ollama/', '')
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

                # If Cahul uncertain, try Impressum to boost
                hh_conf = float(result.get('cahul_confidence', 0.0) or 0.0)
                hh_based = bool(result.get('is_cahul_based', False))
                if hh_conf < 0.6 or not hh_based:
                    imp = analyze_impressum(url)
                    if imp.get('found'):
                        boost = max(hh_conf, float(imp.get('cahul_confidence') or 0.0))
                        result['cahul_confidence'] = boost
                        if boost >= 0.6:
                            result['is_cahul_based'] = True
                            # add evidence
                            ev = result.get('cahul_evidence', '')
                            result['cahul_evidence'] = (ev + ' | ' if ev else '') + str(imp.get('evidence'))

                self.cache.set(cache_key, result)
                verification = EntityVerification(
                    url=url,
                    is_cahul_based=bool(result.get('is_cahul_based', False)),
                    cahul_confidence=float(result.get('cahul_confidence', 0.0)),
                    is_ce_related=bool(result.get('is_ce_related', False)),
                    ce_confidence=float(result.get('ce_confidence', 0.0)),
                    verification_reasoning=f"Cahul: {result.get('cahul_evidence','')} | CE: {result.get('ce_evidence','')}",
                    should_extract=bool(result.get('is_cahul_based') and result.get('is_ce_related')),
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
