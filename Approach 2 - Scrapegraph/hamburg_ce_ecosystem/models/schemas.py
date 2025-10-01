from __future__ import annotations

from typing import TypedDict, List, Optional


class VerificationResultDict(TypedDict, total=False):
    url: str
    is_hamburg_based: bool
    hamburg_confidence: float
    hamburg_evidence: str
    is_ce_related: bool
    ce_confidence: float
    ce_evidence: str
    should_extract: bool
    input_category: str


class EntityProfileDict(TypedDict, total=False):
    url: str
    entity_name: str
    ecosystem_role: str
    contact_persons: List[str]
    emails: List[str]
    phone_numbers: List[str]
    brief_description: str
    ce_relation: str
    ce_activities: List[str]
    partners: List[str]
    partner_urls: List[str]
    address: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    extraction_timestamp: str
    extraction_confidence: float
