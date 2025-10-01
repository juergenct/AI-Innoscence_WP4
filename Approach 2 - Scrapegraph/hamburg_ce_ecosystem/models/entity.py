from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class EcosystemRole(str, Enum):
    STUDENTS = "Students"
    RESEARCHERS = "Researchers"
    HIGHER_EDUCATION = "Higher Education Institutions"
    RESEARCH_INSTITUTES = "Research Institutes"
    NGOS = "Non-Governmental Organizations"
    INDUSTRY = "Industry Partners"
    STARTUPS = "Startups and Entrepreneurs"
    PUBLIC_AUTHORITIES = "Public Authorities"
    POLICY_MAKERS = "Policy Makers"
    END_USERS = "End-Users"
    CITIZEN_ASSOCIATIONS = "Citizen Associations"
    MEDIA = "Media and Communication Partners"
    FUNDING = "Funding Bodies"
    KNOWLEDGE_COMMUNITIES = "Knowledge and Innovation Communities"


class EntityVerification(BaseModel):
    url: HttpUrl
    is_hamburg_based: bool
    hamburg_confidence: float = Field(ge=0, le=1)
    is_ce_related: bool
    ce_confidence: float = Field(ge=0, le=1)
    verification_reasoning: str
    should_extract: bool


class EntityProfile(BaseModel):
    # Basic Information
    url: HttpUrl
    entity_name: str
    ecosystem_role: EcosystemRole

    # Contact Information
    contact_persons: List[str] = []
    emails: List[str] = []
    phone_numbers: List[str] = []

    # CE Information
    brief_description: str = Field(max_length=500)
    ce_relation: str = Field(description="How entity relates to Circular Economy")
    ce_activities: List[str] = []

    # Network Information
    partners: List[str] = []
    partner_urls: List[HttpUrl] = []

    # Location
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Metadata
    extraction_timestamp: str
    extraction_confidence: float = Field(ge=0, le=1)
