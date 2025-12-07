"""Instructor-enhanced extraction for reliable field-by-field extraction.

Uses the Instructor library to extract entity information in parallel focused calls,
each with automatic retries and better prompting than the single-pass approach.

IMPORTANT: All output must be in English. CE activities must match the predefined taxonomy.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Dict

import instructor
from openai import AsyncOpenAI
from pydantic import ValidationError

from hamburg_ce_ecosystem.config.extraction_prompts import (
    BASIC_INFO_PROMPT,
    CE_ACTIVITIES_PROMPT,
    CE_CAPABILITIES_PROMPT,
    CE_NEEDS_PROMPT,
    CONTACT_INFO_PROMPT,
    PARTNERSHIPS_PROMPT,
)
from hamburg_ce_ecosystem.config.ce_activities_taxonomy import (
    CE_ACTIVITIES_TAXONOMY,
    match_activity_to_taxonomy,
    get_all_activities,
    find_activity_category,
)
from hamburg_ce_ecosystem.models.extraction_models import (
    BasicInfo,
    CEActivities,
    CECapabilities,
    CENeeds,
    ContactInfo,
    Partnerships,
)


class InstructorExtractor:
    """Handles all field-by-field extractions using Instructor for reliability."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:32b-instruct-q4_K_M",
        logger: logging.Logger | None = None,
    ):
        """Initialize the Instructor-patched OpenAI client.

        Args:
            base_url: Ollama API base URL
            model: Model name to use for extraction
            logger: Optional logger for debugging
        """
        # Create AsyncOpenAI client and patch with Instructor
        # Use JSON mode with text truncation to prevent context overflow
        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama",  # Ollama doesn't need real API key
            ),
            mode=instructor.Mode.JSON,
        )
        self.model = model.replace("ollama/", "")  # Remove ollama/ prefix if present
        self.logger = logger or logging.getLogger(__name__)

    async def extract_basic_info(self, text: str, url: str) -> BasicInfo:
        """Extract basic identification information.

        Args:
            text: Extracted website content
            url: Entity URL (for deriving name if needed)

        Returns:
            BasicInfo model with entity_name, ecosystem_role, description, address
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=BasicInfo,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant. Extract information from the provided text and structure it according to the schema. Only extract explicitly stated information.",
                    },
                    {
                        "role": "user",
                        "content": f"{BASIC_INFO_PROMPT}\n\nURL: {url}\n\nExtracted Text:\n{text}",
                    },
                ],
                temperature=0.1,  # Slightly higher for better entity name inference
                max_tokens=1500,
            )
        except Exception as e:
            self.logger.warning(f"Basic info extraction failed: {e}")
            # Return minimal valid object
            from urllib.parse import urlparse

            domain = urlparse(url).netloc.replace("www.", "").split(".")[0]
            return BasicInfo(
                entity_name=domain.capitalize(),
                ecosystem_role="Industry Partners",
                brief_description="",
                address="",
            )

    async def extract_contacts(self, text: str) -> ContactInfo:
        """Extract contact information.

        Args:
            text: Extracted website content

        Returns:
            ContactInfo model with emails, phones, contact_persons
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=ContactInfo,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant. Extract contact information from the provided text. Only extract explicitly visible contacts.",
                    },
                    {
                        "role": "user",
                        "content": f"{CONTACT_INFO_PROMPT}\n\nExtracted Text:\n{text}",
                    },
                ],
                temperature=0.0,  # Very strict for contacts
                max_tokens=1000,
            )
        except Exception as e:
            self.logger.warning(f"Contact info extraction failed: {e}")
            return ContactInfo(contact_persons=[], emails=[], phone_numbers=[])

    async def extract_ce_capabilities(self, text: str, url: str) -> CECapabilities:
        """Extract CE capabilities with inference rules.

        Args:
            text: Extracted website content
            url: Entity URL (for context)

        Returns:
            CECapabilities model with ce_capabilities_offered
        """
        # Truncate text to prevent context overflow (keep first 8000 chars)
        # This ensures the schema instructions remain prominent
        truncated_text = text[:8000] if len(text) > 8000 else text

        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=CECapabilities,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract circular economy capabilities that this entity can provide to others.

Focus on:
- What services, products, or infrastructure they offer related to circular economy
- If you see recycling, waste management, repair, reuse, refurbishment, PET, Kreislaufwirtschaft → extract these
- Only extract capabilities explicitly stated or strongly implied

Valid categories: Material Recovery, Recycling Infrastructure, Product Life Extension, Repair Services, Sustainable Materials, Waste Management, CE Consulting""",
                    },
                    {
                        "role": "user",
                        "content": f"{CE_CAPABILITIES_PROMPT}\n\nURL: {url}\n\nExtracted Text:\n{truncated_text}",
                    },
                ],
                temperature=0.1,  # Low temperature for deterministic extraction
                max_tokens=2500,
            )
        except Exception as e:
            self.logger.warning(f"CE capabilities extraction failed: {e}")
            return CECapabilities(ce_capabilities_offered=[])

    async def extract_ce_activities(self, text: str) -> CEActivities:
        """Extract CE activities and relation.

        Args:
            text: Extracted website content

        Returns:
            CEActivities model with ce_relation, ce_activities_structured, ce_activities
        """
        # Truncate text to prevent context overflow
        truncated_text = text[:8000] if len(text) > 8000 else text

        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=CEActivities,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract circular economy activities performed by this entity and describe their CE relation.

Focus on:
- How the entity contributes to circular economy (ce_relation)
- Specific CE activities they perform
- If text mentions recycling, waste management, repair, PET, Kreislaufwirtschaft → extract these activities
- Only extract activities explicitly stated or strongly implied

Valid categories: Material Recovery, Chemical Recycling, Waste Processing, Product Design, Repair & Maintenance""",
                    },
                    {
                        "role": "user",
                        "content": f"{CE_ACTIVITIES_PROMPT}\n\nExtracted Text:\n{truncated_text}",
                    },
                ],
                temperature=0.1,  # Low temperature for deterministic extraction
                max_tokens=2500,
            )
        except Exception as e:
            self.logger.warning(f"CE activities extraction failed: {e}")
            return CEActivities(
                ce_relation="", ce_activities_structured=[], ce_activities=[]
            )

    async def extract_ce_needs(self, text: str) -> CENeeds:
        """Extract CE needs and requirements.

        Args:
            text: Extracted website content

        Returns:
            CENeeds model with ce_needs_requirements, needs_requirements
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=CENeeds,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction assistant identifying organizational needs. Extract CE-related needs and requirements mentioned or implied.",
                    },
                    {
                        "role": "user",
                        "content": f"{CE_NEEDS_PROMPT}\n\nExtracted Text:\n{text}",
                    },
                ],
                temperature=0.1,
                max_tokens=1500,
            )
        except Exception as e:
            self.logger.warning(f"CE needs extraction failed: {e}")
            return CENeeds(
                ce_needs_requirements=[], needs_requirements=[], capability_categories=[]
            )

    async def extract_partnerships(self, text: str) -> Partnerships:
        """Extract partnership and network information.

        Args:
            text: Extracted website content

        Returns:
            Partnerships model with mentioned_partners, discovered_entities
        """
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                response_model=Partnerships,
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract partnerships and network connections mentioned on the website.

Focus on:
- Partners or collaborators explicitly mentioned
- Other CE-related organizations discovered
- If you cannot find all required information for an entity, skip it
- Quality over quantity: empty list is better than incomplete data""",
                    },
                    {
                        "role": "user",
                        "content": f"{PARTNERSHIPS_PROMPT}\n\nExtracted Text:\n{text}",
                    },
                ],
                temperature=0.0,  # Strict for partnerships
                max_tokens=2000,
            )
        except Exception as e:
            self.logger.warning(f"Partnerships extraction failed: {e}")
            return Partnerships(
                mentioned_partners=[],
                discovered_entities=[],
                partners=[],
                partner_urls=[],
                capabilities_offered=[],
            )

    async def extract_all_parallel(self, text: str, url: str) -> dict[str, Any]:
        """Run all 6 extractions in parallel for maximum speed and reliability.

        Args:
            text: Extracted website content from ScrapegraphAI
            url: Entity URL

        Returns:
            Merged dictionary with all extracted fields
        """
        self.logger.info(f"Starting parallel extraction for {url}")

        # Launch all 6 extractions concurrently
        results = await asyncio.gather(
            self.extract_basic_info(text, url),
            self.extract_contacts(text),
            self.extract_ce_capabilities(text, url),
            self.extract_ce_activities(text),
            self.extract_ce_needs(text),
            self.extract_partnerships(text),
            return_exceptions=True,  # Don't fail if one extraction fails
        )

        # Unpack results
        (
            basic_info,
            contact_info,
            ce_capabilities,
            ce_activities,
            ce_needs,
            partnerships,
        ) = results

        # Handle any exceptions
        if isinstance(basic_info, Exception):
            self.logger.error(f"Basic info extraction exception: {basic_info}")
            basic_info = BasicInfo(
                entity_name="Unknown",
                ecosystem_role="Industry Partners",
                brief_description="",
                address="",
            )

        if isinstance(contact_info, Exception):
            self.logger.error(f"Contact info extraction exception: {contact_info}")
            contact_info = ContactInfo(contact_persons=[], emails=[], phone_numbers=[])

        if isinstance(ce_capabilities, Exception):
            self.logger.error(
                f"CE capabilities extraction exception: {ce_capabilities}"
            )
            ce_capabilities = CECapabilities(ce_capabilities_offered=[])

        if isinstance(ce_activities, Exception):
            self.logger.error(f"CE activities extraction exception: {ce_activities}")
            ce_activities = CEActivities(
                ce_relation="", ce_activities_structured=[], ce_activities=[]
            )

        if isinstance(ce_needs, Exception):
            self.logger.error(f"CE needs extraction exception: {ce_needs}")
            ce_needs = CENeeds(
                ce_needs_requirements=[], needs_requirements=[], capability_categories=[]
            )

        if isinstance(partnerships, Exception):
            self.logger.error(f"Partnerships extraction exception: {partnerships}")
            partnerships = Partnerships(
                mentioned_partners=[],
                discovered_entities=[],
                partners=[],
                partner_urls=[],
                capabilities_offered=[],
            )

        # Merge all results into single dictionary
        merged = {
            # From BasicInfo
            "entity_name": basic_info.entity_name,
            "ecosystem_role": basic_info.ecosystem_role,
            "brief_description": basic_info.brief_description,
            "address": basic_info.address,
            # From ContactInfo
            "contact_persons": contact_info.contact_persons,
            "emails": contact_info.emails,
            "phone_numbers": contact_info.phone_numbers,
            # From CECapabilities
            "ce_capabilities_offered": ce_capabilities.ce_capabilities_offered,
            # From CEActivities
            "ce_relation": ce_activities.ce_relation,
            "ce_activities_structured": ce_activities.ce_activities_structured,
            "ce_activities": ce_activities.ce_activities,
            # From CENeeds
            "ce_needs_requirements": ce_needs.ce_needs_requirements,
            "needs_requirements": ce_needs.needs_requirements,
            "capability_categories": ce_needs.capability_categories,
            # From Partnerships
            "mentioned_partners": partnerships.mentioned_partners,
            "discovered_entities": partnerships.discovered_entities,
            "partners": partnerships.partners,
            "partner_urls": partnerships.partner_urls,
            "capabilities_offered": partnerships.capabilities_offered,
        }

        self.logger.info(
            f"Parallel extraction complete for {url}: "
            f"emails={len(merged['emails'])}, "
            f"phones={len(merged['phone_numbers'])}, "
            f"capabilities={len(merged['ce_capabilities_offered'])}"
        )

        # Post-process: Validate and map activities to taxonomy
        merged['ce_activities_structured'] = self._validate_and_map_activities(
            merged.get('ce_activities_structured', [])
        )

        return merged

    def _validate_and_map_activities(
        self, activities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and map extracted activities to the predefined taxonomy.

        Args:
            activities: List of activity dicts with activity_name, description, category

        Returns:
            List of validated activities mapped to taxonomy
        """
        if not activities:
            return []

        validated_activities = []
        all_taxonomy_activities = set(get_all_activities())

        for activity in activities:
            if not isinstance(activity, dict):
                # Handle Pydantic models
                if hasattr(activity, 'model_dump'):
                    activity = activity.model_dump()
                else:
                    continue

            activity_name = activity.get('activity_name', '')
            description = activity.get('description', '')

            # Check if activity is already in taxonomy (exact match)
            if activity_name in all_taxonomy_activities:
                validated_activities.append({
                    'activity_name': activity_name,
                    'description': description,
                    'category': find_activity_category(activity_name)
                })
                continue

            # Try to match the activity to taxonomy using keywords
            matched_activities = match_activity_to_taxonomy(
                f"{activity_name} {description}"
            )

            if matched_activities:
                # Use the first matched taxonomy activity
                matched_name = matched_activities[0]
                validated_activities.append({
                    'activity_name': matched_name,
                    'description': description,
                    'category': find_activity_category(matched_name)
                })
                self.logger.debug(
                    f"Mapped activity '{activity_name}' to taxonomy: '{matched_name}'"
                )
            else:
                # Activity doesn't match taxonomy - log and skip
                self.logger.debug(
                    f"Activity '{activity_name}' does not match any taxonomy entry, skipping"
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_activities = []
        for activity in validated_activities:
            name = activity['activity_name']
            if name not in seen:
                seen.add(name)
                unique_activities.append(activity)

        return unique_activities
