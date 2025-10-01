from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import time
import re
import requests
import yaml
from bs4 import BeautifulSoup
from scrapegraphai.graphs import SmartScraperGraph

from hamburg_ce_ecosystem.config.prompts import EXTRACTION_PROMPT
from hamburg_ce_ecosystem.models.entity import EcosystemRole, EntityProfile
from hamburg_ce_ecosystem.utils.cache import FileCache
from hamburg_ce_ecosystem.utils.logging_setup import setup_logging

EXTRACTION_SCHEMA: Dict[str, str] = {
    "entity_name": "string",
    "ecosystem_role": "string",
    "contact_persons": "list[string]",
    "emails": "list[string]",
    "phone_numbers": "list[string]",
    "brief_description": "string",
    "ce_relation": "string",
    "ce_activities": "list[string]",
    "partners": "list[string]",
    "partner_urls": "list[string]",
    "address": "string",
}


class ExtractionScraper:
    def __init__(self, config_path: str | Path, cache_dir: str | Path | None = None, logger: logging.Logger | None = None):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.cache = FileCache(cache_dir or (self.config_path.parent.parent / 'data' / '.cache' / 'extraction'))
        self.logger = logger or setup_logging(self.config_path.parent.parent / 'logs' / 'scraping_errors.log', console_level=logging.ERROR)
        self.max_retries: int = int(self.config.get('scraper', {}).get('max_retries', 3))

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def create_extraction_prompt() -> str:
        return EXTRACTION_PROMPT

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
    def _determine_role(input_role: Any) -> EcosystemRole:
        if isinstance(input_role, str):
            s = input_role.strip().lower()
        else:
            s = ""
        mapping = [
            ("student", EcosystemRole.STUDENTS),
            ("researcher", EcosystemRole.RESEARCHERS),
            ("higher education", EcosystemRole.HIGHER_EDUCATION),
            ("university", EcosystemRole.HIGHER_EDUCATION),
            ("college", EcosystemRole.HIGHER_EDUCATION),
            ("research institute", EcosystemRole.RESEARCH_INSTITUTES),
            ("institute", EcosystemRole.RESEARCH_INSTITUTES),
            ("ngo", EcosystemRole.NGOS),
            ("e.v", EcosystemRole.NGOS),
            ("company", EcosystemRole.INDUSTRY),
            ("gmbh", EcosystemRole.INDUSTRY),
            ("ag", EcosystemRole.INDUSTRY),
            ("manufacturer", EcosystemRole.INDUSTRY),
            ("consult", EcosystemRole.INDUSTRY),
            ("startup", EcosystemRole.STARTUPS),
            ("entrepreneur", EcosystemRole.STARTUPS),
            ("government", EcosystemRole.PUBLIC_AUTHORITIES),
            ("behörde", EcosystemRole.PUBLIC_AUTHORITIES),
            ("senat", EcosystemRole.PUBLIC_AUTHORITIES),
            ("municipal", EcosystemRole.PUBLIC_AUTHORITIES),
            ("policy", EcosystemRole.POLICY_MAKERS),
            ("consumer", EcosystemRole.END_USERS),
            ("end-user", EcosystemRole.END_USERS),
            ("citizen", EcosystemRole.CITIZEN_ASSOCIATIONS),
            ("media", EcosystemRole.MEDIA),
            ("press", EcosystemRole.MEDIA),
            ("fund", EcosystemRole.FUNDING),
            ("investment", EcosystemRole.FUNDING),
            ("innovation", EcosystemRole.KNOWLEDGE_COMMUNITIES),
            ("cluster", EcosystemRole.KNOWLEDGE_COMMUNITIES),
            ("network", EcosystemRole.KNOWLEDGE_COMMUNITIES),
        ]
        for key, role in mapping:
            if key in s:
                return role
        return EcosystemRole.INDUSTRY

    @staticmethod
    def _heuristic_extract(url: str) -> EntityProfile:
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
        title = (soup.title.get_text(strip=True) if soup.title else "").strip()
        text = soup.get_text(" ", strip=True)
        emails = list({m.group(0) for m in re.finditer(r"[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}", text)})[:5]
        phones = list({m.group(0) for m in re.finditer(r"\+?\d[\d\s().\-]{6,}\d", text)})[:5]
        address = None
        m = re.search(r"\b2[0-2][0-9]{3}\b.*hamburg", text, flags=re.I)
        if m:
            address = m.group(0)
        role = EcosystemRole.INDUSTRY
        if re.search(r"universit|hochschule|college|tuhh|haw", text, re.I):
            role = EcosystemRole.HIGHER_EDUCATION
        elif re.search(r"research|institut|labor", text, re.I):
            role = EcosystemRole.RESEARCH_INSTITUTES
        elif re.search(r"ngo|e\.v\.|non\-government|verband|verein", text, re.I):
            role = EcosystemRole.NGOS
        elif re.search(r"behörde|senat|ministerium|amt|stadt hamburg|freie und hansestadt", text, re.I):
            role = EcosystemRole.PUBLIC_AUTHORITIES
        return EntityProfile(
            url=url,
            entity_name=title or url,
            ecosystem_role=role,
            contact_persons=[],
            emails=emails,
            phone_numbers=phones,
            brief_description=text[:480],
            ce_relation="",
            ce_activities=[],
            partners=[],
            partner_urls=[],
            address=address,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_confidence=0.3,
        )

    def extract_entity_info(self, url: str) -> EntityProfile:
        cache_key = f"extraction::{url}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                role = self._determine_role(cached.get('ecosystem_role'))
                return EntityProfile(
                    url=url,
                    entity_name=cached.get('entity_name', 'Unknown'),
                    ecosystem_role=role,
                    contact_persons=cached.get('contact_persons', []),
                    emails=cached.get('emails', []),
                    phone_numbers=cached.get('phone_numbers', []),
                    brief_description=cached.get('brief_description', ''),
                    ce_relation=cached.get('ce_relation', ''),
                    ce_activities=cached.get('ce_activities', []),
                    partners=cached.get('partners', []),
                    partner_urls=cached.get('partner_urls', []),
                    address=cached.get('address'),
                    extraction_timestamp=datetime.now().isoformat(),
                    extraction_confidence=float(cached.get('extraction_confidence', 0.8)),
                )
            except Exception:
                pass

        graph = SmartScraperGraph(
            prompt=self.create_extraction_prompt(),
            source=url,
            config=self.config,
            schema=EXTRACTION_SCHEMA
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = graph.run()
                result = self._coerce_result(raw)
                
                # Validate required keys exist
                if not result or 'entity_name' not in result:
                    raise ValueError(f'Missing entity_name in result: {result}')
                self.cache.set(cache_key, result)
                role = self._determine_role(result.get('ecosystem_role'))
                profile = EntityProfile(
                    url=url,
                    entity_name=result.get('entity_name', 'Unknown'),
                    ecosystem_role=role,
                    contact_persons=result.get('contact_persons', []),
                    emails=result.get('emails', []),
                    phone_numbers=result.get('phone_numbers', []),
                    brief_description=result.get('brief_description', ''),
                    ce_relation=result.get('ce_relation', ''),
                    ce_activities=result.get('ce_activities', []),
                    partners=result.get('partners', []),
                    partner_urls=result.get('partner_urls', []),
                    address=result.get('address'),
                    extraction_timestamp=datetime.now().isoformat(),
                    extraction_confidence=0.8,
                )
                return profile
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Extraction attempt {attempt}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)
        
        # NO FALLBACK - raise exception to force LLM to work
        error_msg = f"LLM extraction failed after {self.max_retries} attempts for {url}: {last_exc}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
