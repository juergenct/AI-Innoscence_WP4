"""Stage 2.5: Process discovered entities for iterative entity discovery."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict

import yaml

from novi_sad_ce_ecosystem.config.relationship_prompts import ENTITY_DEDUPLICATION_PROMPT
from novi_sad_ce_ecosystem.models.entity import DiscoveredEntityRecord
from novi_sad_ce_ecosystem.models.schemas import EntityProfileDict, DiscoveredEntityDict, DiscoveredEntityRecordDict
from novi_sad_ce_ecosystem.utils.logging_setup import setup_logging
from novi_sad_ce_ecosystem.utils.ollama_structured import call_ollama_chat
from pydantic import BaseModel, Field


class DuplicateGroup(BaseModel):
    """Pydantic model for a group of duplicate entities."""
    entity_urls: List[str] = Field(description="List of URLs that refer to the same entity")
    canonical_name: str = Field(description="The canonical/official name to use")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that these are duplicates")


class DeduplicationResult(BaseModel):
    """Pydantic model for entity deduplication results."""
    duplicate_groups: List[DuplicateGroup] = Field(
        default_factory=list,
        description="Groups of entities that are duplicates"
    )


class DiscoveredEntityProcessor:
    """Processes discovered entities to enable iterative entity discovery."""

    def __init__(
        self,
        config_path: str | Path,
        data_dir: str | Path | None = None,
        logger: logging.Logger | None = None
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.data_dir = Path(data_dir) if data_dir else self.config_path.parent.parent / 'data'
        self.logger = logger or setup_logging(
            self.config_path.parent.parent / 'logs' / 'discovered_entities.log',
            console_level=logging.INFO
        )

        # Discovery settings
        discovery_config = self.config.get('discovered_entity_processing', {})
        self.max_discovery_depth: int = discovery_config.get('max_discovery_depth', 3)
        self.batch_size: int = discovery_config.get('batch_size', 30)
        self.min_confidence: float = discovery_config.get('min_deduplication_confidence', 0.7)

        # LLM settings
        self.model = self.config.get('llm', {}).get('model', 'qwen2.5:32b-instruct-q4_K_M').replace('ollama/', '')
        self.base_url = self.config.get('llm', {}).get('base_url', 'http://localhost:11434')

        # Storage
        self.discovered_entities_file = self.data_dir / 'discovered_entities' / 'discovery_records.json'
        self.discovered_entities_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def aggregate_discovered_entities(
        self,
        entity_profiles: List[EntityProfileDict]
    ) -> List[DiscoveredEntityDict]:
        """
        Aggregate all discovered entities from all entity profiles.

        Args:
            entity_profiles: List of entity profiles from extraction stage

        Returns:
            List of unique discovered entities
        """
        self.logger.info(f"Aggregating discovered entities from {len(entity_profiles)} profiles...")

        discovered: List[DiscoveredEntityDict] = []
        seen_urls: Set[str] = set()

        for profile in entity_profiles:
            profile_discovered = profile.get('discovered_entities', [])
            for entity in profile_discovered:
                url = entity.get('url', '')
                if url and url not in seen_urls:
                    discovered.append(entity)
                    seen_urls.add(url)

        self.logger.info(f"Found {len(discovered)} unique discovered entities")
        return discovered

    def deduplicate_entities(
        self,
        discovered_entities: List[DiscoveredEntityDict]
    ) -> Dict[str, str]:
        """
        Use LLM to deduplicate discovered entities (semantic matching).

        Args:
            discovered_entities: List of discovered entities

        Returns:
            Mapping from entity URL to canonical URL (duplicates point to canonical)
        """
        if len(discovered_entities) == 0:
            return {}

        self.logger.info(f"Deduplicating {len(discovered_entities)} entities using LLM...")

        # Process in batches
        duplicate_map: Dict[str, str] = {}  # url -> canonical_url
        total_batches = (len(discovered_entities) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(discovered_entities), self.batch_size):
            batch = discovered_entities[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1

            self.logger.info(f"Processing deduplication batch {batch_num}/{total_batches}...")

            # Format entities for LLM
            entities_text = "\n\n".join([
                f"Entity {i+1}:\n"
                f"  Name: {e.get('name', 'Unknown')}\n"
                f"  URL: {e.get('url', 'N/A')}\n"
                f"  Description: {e.get('brief_description', 'N/A')}\n"
                f"  Context: {e.get('context', 'N/A')}"
                for i, e in enumerate(batch)
            ])

            try:
                # Call LLM for deduplication
                result = call_ollama_chat(
                    prompt=ENTITY_DEDUPLICATION_PROMPT.format(entities_batch=entities_text),
                    text_content="",  # Prompt contains everything
                    response_model=DeduplicationResult,
                    model=self.model,
                    base_url=self.base_url,
                    temperature=0.0
                )

                # Process duplicate groups
                for group in result.duplicate_groups:
                    if group.confidence >= self.min_confidence and len(group.entity_urls) > 1:
                        # First URL in group is canonical
                        canonical_url = group.entity_urls[0]
                        for url in group.entity_urls:
                            duplicate_map[url] = canonical_url

                        self.logger.info(
                            f"Identified duplicate group ({group.confidence:.2f}): "
                            f"{len(group.entity_urls)} entities -> {group.canonical_name}"
                        )

            except Exception as e:
                self.logger.warning(f"Deduplication batch {batch_num} failed: {e}")
                continue

        self.logger.info(f"Deduplication complete. Found {len(duplicate_map)} duplicate mappings.")
        return duplicate_map

    def filter_new_entities(
        self,
        discovered_entities: List[DiscoveredEntityDict],
        existing_profiles: List[EntityProfileDict],
        duplicate_map: Dict[str, str]
    ) -> List[DiscoveredEntityDict]:
        """
        Filter out entities that already exist in the database.

        Args:
            discovered_entities: Discovered entities
            existing_profiles: Already processed entity profiles
            duplicate_map: URL -> canonical URL mapping from deduplication

        Returns:
            List of truly new entities to process
        """
        # Get all existing URLs
        existing_urls = {str(profile.get('url', '')) for profile in existing_profiles}

        # Apply deduplication map
        new_entities: List[DiscoveredEntityDict] = []
        for entity in discovered_entities:
            url = entity.get('url', '')
            canonical_url = duplicate_map.get(url, url)

            # Check if canonical URL is new
            if canonical_url and canonical_url not in existing_urls:
                # Use canonical entity if it's different
                if canonical_url != url:
                    # Find the canonical entity
                    canonical_entity = next(
                        (e for e in discovered_entities if e.get('url') == canonical_url),
                        entity
                    )
                    new_entities.append(canonical_entity)
                else:
                    new_entities.append(entity)
                existing_urls.add(canonical_url)  # Mark as seen

        self.logger.info(f"Filtered to {len(new_entities)} new entities (removed {len(discovered_entities) - len(new_entities)} existing/duplicates)")
        return new_entities

    def create_discovery_records(
        self,
        new_entities: List[DiscoveredEntityDict],
        entity_profiles: List[EntityProfileDict],
        discovery_depth: int,
        parent_chain: str = ""
    ) -> List[DiscoveredEntityRecordDict]:
        """
        Create discovery records tracking which entity discovered which.

        Args:
            new_entities: New entities to track
            entity_profiles: Profiles that discovered these entities
            discovery_depth: Current depth in discovery chain
            parent_chain: Discovery chain of parent entity

        Returns:
            List of discovery records
        """
        records: List[DiscoveredEntityRecordDict] = []

        # Create lookup: discovered URL -> discovering profile
        for profile in entity_profiles:
            profile_url = str(profile.get('url', ''))
            profile_name = profile.get('entity_name', 'Unknown')
            profile_discovered = profile.get('discovered_entities', [])

            for discovered in profile_discovered:
                discovered_url = discovered.get('url', '')
                discovered_name = discovered.get('name', 'Unknown')

                # Check if this discovered entity is in our new_entities list
                if any(e.get('url') == discovered_url for e in new_entities):
                    # Build discovery chain
                    if parent_chain:
                        chain = f"{parent_chain} -> {discovered_name}"
                    else:
                        chain = f"{profile_name} -> {discovered_name}"

                    record: DiscoveredEntityRecordDict = {
                        'discovered_entity_url': discovered_url,
                        'discovered_entity_name': discovered_name,
                        'discovered_by_entity': profile_name,
                        'discovered_by_url': profile_url,
                        'discovery_depth': discovery_depth,
                        'discovery_chain': chain,
                        'timestamp': datetime.now().isoformat(),
                        'processed': False
                    }
                    records.append(record)

        self.logger.info(f"Created {len(records)} discovery records at depth {discovery_depth}")
        return records

    def save_discovery_records(self, records: List[DiscoveredEntityRecordDict]) -> None:
        """Save discovery records to disk."""
        # Load existing records
        existing_records = []
        if self.discovered_entities_file.exists():
            try:
                with open(self.discovered_entities_file, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing discovery records: {e}")

        # Append new records
        existing_records.extend(records)

        # Save
        try:
            with open(self.discovered_entities_file, 'w', encoding='utf-8') as f:
                json.dump(existing_records, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(records)} new discovery records")
        except Exception as e:
            self.logger.error(f"Failed to save discovery records: {e}")

    def load_discovery_records(self) -> List[DiscoveredEntityRecordDict]:
        """Load discovery records from disk."""
        if not self.discovered_entities_file.exists():
            return []

        try:
            with open(self.discovered_entities_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load discovery records: {e}")
            return []

    def process_discovered_entities(
        self,
        entity_profiles: List[EntityProfileDict],
        discovery_depth: int = 1,
        parent_chain: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Main processing pipeline for discovered entities.

        Args:
            entity_profiles: Entity profiles from extraction stage
            discovery_depth: Current depth in discovery chain
            parent_chain: Discovery chain from parent

        Returns:
            List of new entities to add to processing queue (with URL and name)
        """
        if discovery_depth > self.max_discovery_depth:
            self.logger.info(f"Reached max discovery depth ({self.max_discovery_depth}), stopping.")
            return []

        # Step 1: Aggregate all discovered entities
        discovered = self.aggregate_discovered_entities(entity_profiles)
        if not discovered:
            self.logger.info("No discovered entities to process.")
            return []

        # Step 2: Deduplicate using LLM
        duplicate_map = self.deduplicate_entities(discovered)

        # Step 3: Filter to only new entities
        # Note: We need existing profiles from database - for now, assume entity_profiles are all existing
        new_entities = self.filter_new_entities(discovered, entity_profiles, duplicate_map)

        if not new_entities:
            self.logger.info("No new entities after filtering.")
            return []

        # Step 4: Create discovery records
        discovery_records = self.create_discovery_records(
            new_entities,
            entity_profiles,
            discovery_depth,
            parent_chain
        )

        # Step 5: Save discovery records
        self.save_discovery_records(discovery_records)

        # Step 6: Format for pipeline processing
        entities_to_process = [
            {
                'url': entity.get('url'),
                'name': entity.get('name'),
                'description': entity.get('brief_description'),
                'discovered_by': next(
                    (r['discovered_by_entity'] for r in discovery_records
                     if r['discovered_entity_url'] == entity.get('url')),
                    'Unknown'
                )
            }
            for entity in new_entities
        ]

        self.logger.info(f"Queued {len(entities_to_process)} new entities for processing at depth {discovery_depth}")
        return entities_to_process
