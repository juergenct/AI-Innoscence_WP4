from __future__ import annotations

import asyncio
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import time
import yaml
from scrapegraphai.graphs import SmartScraperGraph

from hamburg_ce_ecosystem.models.entity import EcosystemRole, EntityProfile
from hamburg_ce_ecosystem.utils.cache import FileCache
from hamburg_ce_ecosystem.utils.instructor_extraction import InstructorExtractor
from hamburg_ce_ecosystem.utils.logging_setup import setup_logging
from hamburg_ce_ecosystem.utils.ollama_structured import call_ollama_chat, ExtractionResult
from hamburg_ce_ecosystem.utils.web_fetcher import fetch_website_content
from hamburg_ce_ecosystem.config.ce_activities_taxonomy import (
    CE_ACTIVITIES_TAXONOMY,
    ACTIVITY_KEYWORDS,
    match_activity_to_taxonomy,
    find_activity_category,
)


class ExtractionScraper:
    def __init__(self, config_path: str | Path, cache_dir: str | Path | None = None, logger: logging.Logger | None = None):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.cache = FileCache(cache_dir or (self.config_path.parent.parent / 'data' / '.cache' / 'extraction'))
        self.logger = logger or setup_logging(self.config_path.parent.parent / 'logs' / 'scraping_errors.log', console_level=logging.ERROR)
        self.max_retries: int = int(self.config.get('scraper', {}).get('max_retries', 3))

        # HTML storage directory
        self.html_storage_dir = self.config_path.parent.parent / 'data' / 'raw_html'
        self.html_storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Instructor-enhanced extraction
        model = self.config.get('extraction', {}).get('model', self.config.get('llm', {}).get('model', 'qwen2.5:32b-instruct-q4_K_M'))
        base_url = self.config.get('llm', {}).get('base_url', 'http://localhost:11434')
        self.instructor_extractor = InstructorExtractor(
            base_url=base_url,
            model=model,
            logger=self.logger
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _save_content(self, url: str, extracted_text: str = None, original_html: str = None) -> tuple[str, str]:
        """
        Save both extracted text and original HTML to disk.

        Args:
            url: The URL being scraped
            extracted_text: ScrapegraphAI extracted/cleaned text content
            original_html: Original HTML from Playwright

        Returns:
            Tuple of (extracted_text_path, original_html_path) - either can be None if not saved
        """
        # Create entity-specific directory using URL hash
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        entity_dir = self.html_storage_dir / url_hash
        entity_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_path = None
        original_path = None

        try:
            # Save extracted text from ScrapegraphAI
            if extracted_text:
                extracted_filename = f"{timestamp}_extracted.txt"
                extracted_file_path = entity_dir / extracted_filename
                with open(extracted_file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                extracted_path = str(extracted_file_path.relative_to(self.config_path.parent.parent / 'data'))

            # Save original HTML from Playwright
            if original_html:
                original_filename = f"{timestamp}_raw.html"
                original_file_path = entity_dir / original_filename
                with open(original_file_path, 'w', encoding='utf-8') as f:
                    f.write(original_html)
                original_path = str(original_file_path.relative_to(self.config_path.parent.parent / 'data'))

            # Save metadata
            metadata = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "extracted_file": extracted_filename if extracted_text else None,
                "original_html_file": original_filename if original_html else None,
                "extracted_length": len(extracted_text) if extracted_text else 0,
                "original_length": len(original_html) if original_html else 0
            }
            metadata_path = entity_dir / f"{timestamp}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return extracted_path, original_path
        except Exception as e:
            self.logger.warning(f"Failed to save content for {url}: {e}")
            return None, None

    def _validate_discovered_entities(self, discovered_entities: list, url: str) -> list:
        """Validate and filter discovered_entities to prevent validation errors.

        Args:
            discovered_entities: List of discovered entities (Pydantic models or dicts)
            url: Entity URL (for logging)

        Returns:
            Filtered list with only valid discovered entities (as models or dicts)
        """
        if not discovered_entities:
            return []

        valid_entities = []
        filtered_entities = []

        for idx, entity in enumerate(discovered_entities):
            # Convert Pydantic model to dict for validation
            if hasattr(entity, 'model_dump'):
                entity_dict = entity.model_dump()
            elif isinstance(entity, dict):
                entity_dict = entity
            else:
                filtered_entities.append({
                    'index': idx,
                    'reason': 'Not a dictionary or Pydantic model',
                    'value': str(entity)
                })
                continue

            # Required fields
            required_fields = ['name', 'url', 'brief_description', 'context']
            missing_fields = [field for field in required_fields if field not in entity_dict]

            if missing_fields:
                filtered_entities.append({
                    'index': idx,
                    'reason': f'Missing required fields: {", ".join(missing_fields)}',
                    'entity': entity_dict
                })
                continue

            # Check for None/null values
            none_fields = [field for field in required_fields if entity_dict.get(field) is None]
            if none_fields:
                filtered_entities.append({
                    'index': idx,
                    'reason': f'None/null values in fields: {", ".join(none_fields)}',
                    'entity': entity_dict
                })
                continue

            # Check for empty strings
            empty_fields = [field for field in required_fields if not str(entity_dict.get(field, '')).strip()]
            if empty_fields:
                filtered_entities.append({
                    'index': idx,
                    'reason': f'Empty string in fields: {", ".join(empty_fields)}',
                    'entity': entity_dict
                })
                continue

            # Entity is valid - return original entity (Pydantic model or dict)
            valid_entities.append(entity)

        # Log filtered entities
        if filtered_entities:
            self.logger.warning(
                f"Filtered {len(filtered_entities)} invalid discovered_entities for {url}"
            )
            for filtered in filtered_entities:
                self.logger.debug(f"  Filtered entity {filtered['index']}: {filtered['reason']}")

            # Save filtered entities to recovery directory for later review
            try:
                recovery_dir = self.config_path.parent.parent / 'data' / 'recovery'
                recovery_dir.mkdir(parents=True, exist_ok=True)

                filtered_file = recovery_dir / 'filtered_discovered_entities.jsonl'
                with open(filtered_file, 'a', encoding='utf-8') as f:
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'url': url,
                        'filtered_count': len(filtered_entities),
                        'valid_count': len(valid_entities),
                        'filtered_entities': filtered_entities
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to save filtered entities: {e}")

        self.logger.info(
            f"Discovered entities validation for {url}: {len(valid_entities)} valid, "
            f"{len(filtered_entities)} filtered out"
        )

        return valid_entities

    def _create_minimal_profile(self, url: str, error: Exception, extracted_path: str | None = None, original_html_path: str | None = None) -> EntityProfile:
        """Create minimal entity profile when extraction fails.

        Args:
            url: Entity URL
            error: The exception that caused extraction failure
            extracted_path: Path to extracted text file
            original_html_path: Path to original HTML file

        Returns:
            Minimal EntityProfile with empty fields
        """
        # Try to derive entity name from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or parsed_url.path
        entity_name = domain.replace('www.', '').replace('.de', '').replace('.com', '').replace('.', ' ').title()

        self.logger.warning(f"Creating minimal profile for {url} due to extraction failure: {error}")

        # Create minimal profile with empty fields
        minimal_profile = EntityProfile(
            url=url,
            entity_name=entity_name,
            ecosystem_role=EcosystemRole.INDUSTRY,  # Default role
            contact_persons=[],
            emails=[],
            phone_numbers=[],
            brief_description='',
            ce_relation='',
            ce_activities=[],
            capabilities_offered=[],
            needs_requirements=[],
            capability_categories=[],
            partners=[],
            partner_urls=[],
            ce_activities_structured=[],
            ce_capabilities_offered=[],
            ce_needs_requirements=[],
            mentioned_partners=[],
            discovered_entities=[],
            address=None,
            extraction_timestamp=datetime.now().isoformat(),
            extraction_confidence=0.0,  # Zero confidence for failed extraction
            raw_html_path=extracted_path,
            raw_html_original_path=original_html_path,
        )

        # Log to partial extractions file
        try:
            recovery_dir = self.config_path.parent.parent / 'data' / 'recovery'
            recovery_dir.mkdir(parents=True, exist_ok=True)

            partial_file = recovery_dir / 'partial_extractions.jsonl'
            with open(partial_file, 'a', encoding='utf-8') as f:
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'url': url,
                    'entity_name': entity_name,
                    'error': str(error),
                    'error_type': type(error).__name__,
                    'raw_html_path': extracted_path,
                    'status': 'minimal_profile_created'
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save partial extraction record: {e}")

        return minimal_profile

    @staticmethod
    def _parse_role(input_role: Any) -> EcosystemRole:
        """Parse LLM-returned role string into EcosystemRole enum."""
        if isinstance(input_role, EcosystemRole):
            return input_role

        if not isinstance(input_role, str):
            return EcosystemRole.INDUSTRY

        role_str = input_role.strip()

        # Try exact match first (case-insensitive)
        for role in EcosystemRole:
            if role.value.lower() == role_str.lower():
                return role

        # Fuzzy matching as fallback
        role_lower = role_str.lower()
        if "student" in role_lower:
            return EcosystemRole.STUDENTS
        elif "researcher" in role_lower:
            return EcosystemRole.RESEARCHERS
        elif "higher education" in role_lower or "universit" in role_lower:
            return EcosystemRole.HIGHER_EDUCATION
        elif "research institute" in role_lower:
            return EcosystemRole.RESEARCH_INSTITUTES
        elif "non-governmental" in role_lower or "ngo" in role_lower:
            return EcosystemRole.NGOS
        elif "industry" in role_lower or "partner" in role_lower:
            return EcosystemRole.INDUSTRY
        elif "startup" in role_lower or "entrepreneur" in role_lower:
            return EcosystemRole.STARTUPS
        elif "public" in role_lower or "authorit" in role_lower:
            return EcosystemRole.PUBLIC_AUTHORITIES
        elif "policy" in role_lower:
            return EcosystemRole.POLICY_MAKERS
        elif "end-user" in role_lower:
            return EcosystemRole.END_USERS
        elif "citizen" in role_lower:
            return EcosystemRole.CITIZEN_ASSOCIATIONS
        elif "media" in role_lower:
            return EcosystemRole.MEDIA
        elif "funding" in role_lower:
            return EcosystemRole.FUNDING
        elif "knowledge" in role_lower or "innovation" in role_lower:
            return EcosystemRole.KNOWLEDGE_COMMUNITIES

        # Default fallback
        return EcosystemRole.INDUSTRY

    def _apply_ce_fallback_logic(self, entity_name: str, brief_description: str, url: str,
                                  ce_activities_structured: list, ce_capabilities_offered: list,
                                  ce_needs_requirements: list) -> tuple[list, list, list]:
        """
        Apply taxonomy-based fallback logic if CE fields are empty but CE keywords are detected.

        Uses the predefined CE Activities Taxonomy to ensure consistent, English-only activities.

        Returns:
            Tuple of (ce_activities_structured, ce_capabilities_offered, ce_needs_requirements)
        """
        # Check if we need to apply fallback (all three lists are empty)
        if ce_activities_structured or ce_capabilities_offered or ce_needs_requirements:
            # At least one field is populated, no fallback needed
            return ce_activities_structured, ce_capabilities_offered, ce_needs_requirements

        # Combine text to search for keywords
        search_text = f"{entity_name.lower()} {brief_description.lower()} {url.lower()}"

        # Use taxonomy-based keyword matching
        matched_activities = match_activity_to_taxonomy(search_text)

        # If no activities matched, return original empty lists
        if not matched_activities:
            return ce_activities_structured, ce_capabilities_offered, ce_needs_requirements

        # Apply fallback: populate fields based on matched taxonomy activities
        self.logger.info(f"CE Fallback activated for {entity_name}: matched {len(matched_activities)} taxonomy activities")

        new_activities = []
        new_capabilities = []

        for activity_name in matched_activities:
            category = find_activity_category(activity_name)

            # Add activity from taxonomy
            new_activities.append({
                'activity_name': activity_name,
                'description': f"Entity performs {activity_name.lower()} activities",
                'category': category
            })

            # Add corresponding capability (inferred from activity)
            new_capabilities.append({
                'capability_name': f"{category} Services",
                'description': f"Provides services related to {activity_name.lower()}",
                'category': category
            })

        # Remove duplicates
        seen_activities = set()
        unique_activities = []
        for act in new_activities:
            if act['activity_name'] not in seen_activities:
                seen_activities.add(act['activity_name'])
                unique_activities.append(act)

        seen_capabilities = set()
        unique_capabilities = []
        for cap in new_capabilities:
            if cap['capability_name'] not in seen_capabilities:
                seen_capabilities.add(cap['capability_name'])
                unique_capabilities.append(cap)

        return unique_activities, unique_capabilities, ce_needs_requirements

    async def extract_entity_info(self, url: str) -> EntityProfile:
        cache_key = f"extraction::{url}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                role = self._parse_role(cached.get('ecosystem_role'))
                return EntityProfile(
                    url=url,
                    entity_name=cached.get('entity_name', 'Unknown'),
                    ecosystem_role=role,
                    contact_persons=cached.get('contact_persons', []),
                    emails=cached.get('emails', []),
                    phone_numbers=cached.get('phone_numbers', []),
                    brief_description=cached.get('brief_description', ''),
                    ce_relation=cached.get('ce_relation', ''),
                    # Legacy fields
                    ce_activities=cached.get('ce_activities', []),
                    capabilities_offered=cached.get('capabilities_offered', []),
                    needs_requirements=cached.get('needs_requirements', []),
                    capability_categories=cached.get('capability_categories', []),
                    partners=cached.get('partners', []),
                    partner_urls=cached.get('partner_urls', []),
                    # New CE-focused structured fields
                    ce_activities_structured=cached.get('ce_activities_structured', []),
                    ce_capabilities_offered=cached.get('ce_capabilities_offered', []),
                    ce_needs_requirements=cached.get('ce_needs_requirements', []),
                    mentioned_partners=cached.get('mentioned_partners', []),
                    discovered_entities=cached.get('discovered_entities', []),
                    address=cached.get('address'),
                    extraction_timestamp=datetime.now().isoformat(),
                    extraction_confidence=float(cached.get('extraction_confidence', 0.8)),
                    # HTML storage paths
                    raw_html_path=cached.get('raw_html_path'),
                    raw_html_original_path=cached.get('raw_html_original_path'),
                )
            except Exception:
                pass

        # TWO-STAGE LLM APPROACH: ScrapegraphAI extraction + Ollama structuring
        
        # Stage 1: ScrapegraphAI intelligently extracts information (no strict JSON required)
        scrapegraph_prompt = """Extract ONLY information that is actually on this website. Do not generate or assume anything.

BASIC INFO (check homepage, about, header, footer, impressum):
- Official organization/company name (check: page title, logo alt text, header text, footer, impressum section "Name der Firma")
- If official name not found on page, extract from domain name in URL
- Organization type indicators: university/Hochschule, research institute/Forschungsinstitut, GmbH/AG/company, startup, e.V./NGO, government/Behörde, media, funding body, network
- Description of what they do (copy exact text, 2-3 sentences)

CONTACTS (check contact, team, about, impressum pages):
- Names of key people with exact titles (copy as shown)
- All email addresses (copy exactly - prioritize organizational emails like info@, kontakt@, contact@, hello@, hallo@, mail@, office@)
- All phone numbers (copy exactly)

LOCATION (check contact, impressum, footer):
- Complete physical address (copy exact text)
- Street name and number
- Postal code (must be 20000-22999 for Hamburg)

CIRCULAR ECONOMY (check services, products, about, sustainability, projects):
- Copy exact descriptions of CE contribution
- Copy specific CE activities/services/products mentioned
- Copy waste management, recycling, reuse, repair services
- Copy sustainability certifications mentioned
- Copy CE project names

PARTNERSHIPS (check partners, about, collaboration, network pages):
- Partner organization names (copy exactly)
- Partner website URLs (copy full URLs!)
- Collaboration project names
- Network memberships

ECOSYSTEM CAPABILITIES (check services, products, what-we-do, portfolio, expertise pages):
- What can this organization provide to others? (e.g., research, consulting, recycling services, technology, funding, training)
- Specific services or products offered
- Areas of expertise mentioned

ECOSYSTEM NEEDS (check about, contact, looking-for, challenges, goals pages):
- What is this organization looking for? (e.g., partners, funding, expertise, clients, materials, technology)
- Challenges or gaps mentioned
- "We are seeking..." or "We need..." statements

Visit all relevant pages (homepage, /about, /team, /contact, /partners, /sustainability, /impressum, /services, /products). Return ONLY what you actually find on the pages."""
        
        graph = SmartScraperGraph(
            prompt=scrapegraph_prompt,
            source=url,
            config=self.config
        )
        
        # Fetch original HTML using Playwright for complete preservation
        try:
            original_html = fetch_website_content(url, max_pages=4, timeout=30000)
            self.logger.debug(f"Fetched original HTML for {url} ({len(original_html)} chars)")
        except Exception as e:
            self.logger.warning(f"Failed to fetch original HTML for {url}: {e}")
            original_html = None

        try:
            # Run sync Playwright code in thread pool to avoid asyncio conflicts
            scrapegraph_output = await asyncio.to_thread(graph.run)
            # Convert to string - format doesn't matter
            if isinstance(scrapegraph_output, dict):
                extracted_text = json.dumps(scrapegraph_output, ensure_ascii=False, indent=2)
            else:
                extracted_text = str(scrapegraph_output)

            # Save both extracted text and original HTML
            extracted_path, original_html_path = self._save_content(url, extracted_text, original_html)
        except Exception as e:
            # ERROR RECOVERY: Extract useful content from ScrapegraphAI's error message
            # (Following pattern from GitHub issues #809, #324, #257)
            error_msg = str(e)

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
                    # No recoverable content, use original HTML as fallback
                    extracted_text = original_html if original_html else fetch_website_content(url, max_pages=4, timeout=30000)
                    self.logger.debug(f"Using Playwright fallback for {url}")
            else:
                # Different error, use original HTML as fallback
                extracted_text = original_html if original_html else fetch_website_content(url, max_pages=4, timeout=30000)
                self.logger.debug(f"Using Playwright fallback for {url}")

            # Save both extracted text and original HTML
            extracted_path, original_html_path = self._save_content(url, extracted_text, original_html)

        # Stage 2: Instructor-based parallel extraction (6 focused calls)
        # DEBUG: Log extracted text snippet to verify ScrapegraphAI output
        self.logger.debug(f"ScrapegraphAI extracted text length: {len(extracted_text)} chars")
        self.logger.debug(f"ScrapegraphAI output preview (first 500 chars): {extracted_text[:500]}")

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Instructor + Parallel extraction: 6 focused calls for better reliability
                self.logger.info(f"Starting parallel focused extraction for {url} (attempt {attempt}/{self.max_retries})")
                result = await self.instructor_extractor.extract_all_parallel(
                    text=extracted_text,  # ← From ScrapegraphAI
                    url=url
                )

                # DEBUG: Log structured field extraction results
                self.logger.debug(f"Parallel extraction results for {url}:")
                self.logger.debug(f"  - entity_name: {result.get('entity_name')}")
                self.logger.debug(f"  - emails: {len(result.get('emails', []))} items")
                self.logger.debug(f"  - phone_numbers: {len(result.get('phone_numbers', []))} items")
                self.logger.debug(f"  - ce_activities_structured: {len(result.get('ce_activities_structured', []))} items")
                self.logger.debug(f"  - ce_capabilities_offered: {len(result.get('ce_capabilities_offered', []))} items")
                self.logger.debug(f"  - ce_needs_requirements: {len(result.get('ce_needs_requirements', []))} items")
                if result.get('ce_activities_structured'):
                    self.logger.debug(f"  - Sample ce_activities_structured: {result['ce_activities_structured'][:2]}")
                if result.get('ce_capabilities_offered'):
                    self.logger.debug(f"  - Sample ce_capabilities_offered: {result['ce_capabilities_offered'][:2]}")

                # Add HTML storage paths to cache for persistence
                result['raw_html_path'] = extracted_path
                result['raw_html_original_path'] = original_html_path
                self.cache.set(cache_key, result)

                # Validate and filter discovered_entities BEFORE EntityProfile creation
                raw_discovered_entities = result.get('discovered_entities', [])
                validated_discovered_entities = self._validate_discovered_entities(
                    raw_discovered_entities, url
                )

                role = self._parse_role(result.get('ecosystem_role'))
                profile = EntityProfile(
                    url=url,
                    entity_name=result.get('entity_name', 'Unknown'),
                    ecosystem_role=role,
                    contact_persons=result.get('contact_persons', []),
                    emails=result.get('emails', []),
                    phone_numbers=result.get('phone_numbers', []),
                    brief_description=result.get('brief_description') or '',
                    ce_relation=result.get('ce_relation') or '',
                    # Legacy fields
                    ce_activities=result.get('ce_activities', []),
                    capabilities_offered=result.get('capabilities_offered', []),
                    needs_requirements=result.get('needs_requirements', []),
                    capability_categories=result.get('capability_categories', []),
                    partners=result.get('partners', []),
                    partner_urls=result.get('partner_urls', []),
                    # New CE-focused structured fields
                    ce_activities_structured=result.get('ce_activities_structured', []),
                    ce_capabilities_offered=result.get('ce_capabilities_offered', []),
                    ce_needs_requirements=result.get('ce_needs_requirements', []),
                    mentioned_partners=result.get('mentioned_partners', []),
                    discovered_entities=validated_discovered_entities,  # Use validated list
                    address=result.get('address'),
                    extraction_timestamp=datetime.now().isoformat(),
                    extraction_confidence=0.8,
                    raw_html_path=extracted_path,  # ScrapegraphAI extracted text
                    raw_html_original_path=original_html_path,  # Original HTML from Playwright
                )

                # Apply CE fallback logic if needed
                profile.ce_activities_structured, profile.ce_capabilities_offered, profile.ce_needs_requirements = \
                    self._apply_ce_fallback_logic(
                        entity_name=profile.entity_name,
                        brief_description=profile.brief_description,
                        url=str(profile.url),
                        ce_activities_structured=profile.ce_activities_structured,
                        ce_capabilities_offered=profile.ce_capabilities_offered,
                        ce_needs_requirements=profile.ce_needs_requirements
                    )

                return profile
            except Exception as e:
                last_exc = e
                self.logger.warning(f"Extraction attempt {attempt}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)

        # GRACEFUL DEGRADATION: Create minimal profile with empty fields
        # This ensures the entity exists in the ecosystem even if extraction failed
        self.logger.error(f"LLM extraction failed after {self.max_retries} attempts for {url}: {last_exc}")
        self.logger.info(f"Creating minimal profile for {url} to preserve entity in ecosystem")

        # Create minimal profile - entity will have empty fields but can be retried later
        minimal_profile = self._create_minimal_profile(
            url=url,
            error=last_exc,
            extracted_path=extracted_path,
            original_html_path=original_html_path
        )

        return minimal_profile
