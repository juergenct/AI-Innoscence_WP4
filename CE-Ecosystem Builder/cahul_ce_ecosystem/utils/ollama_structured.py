"""Direct Ollama structured output integration using Pydantic (best practice)."""
from __future__ import annotations

import requests
import httpx
from typing import Dict, Any, TypeVar
from pydantic import BaseModel, Field, field_validator

T = TypeVar('T', bound=BaseModel)


class VerificationResult(BaseModel):
    """Pydantic model for Cahul/CE verification."""
    is_cahul_based: bool = Field(description="Is entity based in Cahul, Moldova?")
    cahul_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1 scale)")
    cahul_evidence: str = Field(description="Evidence found for Cahul location")
    is_ce_related: bool = Field(description="Is entity related to Circular Economy?")
    ce_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1 scale)")
    ce_evidence: str = Field(description="Evidence found for CE relation")

    @field_validator('cahul_confidence', 'ce_confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, v):
        """Auto-convert percentage (0-100) to decimal (0-1) if needed."""
        if isinstance(v, (int, float)) and v > 1.0:
            return v / 100.0
        return v


# SIMPLIFIED: Use dicts instead of nested Pydantic models for better LLM compatibility
# The LLM can easily generate simple dicts, and we preserve all the same information


class ExtractionResult(BaseModel):
    """Pydantic model for entity extraction with 14 ecosystem roles and CE-focused fields."""
    entity_name: str = Field(description="Official organization/company name")
    ecosystem_role: str = Field(
        description="Classify based on PRIMARY ACTIVITY and PURPOSE. MUST be EXACTLY one of these values: 'Students', 'Researchers', 'Higher Education Institutions', 'Research Institutes', 'Non-Governmental Organizations', 'Industry Partners', 'Startups and Entrepreneurs', 'Public Authorities', 'Policy Makers', 'End-Users', 'Citizen Associations', 'Media and Communication Partners', 'Funding Bodies', 'Knowledge and Innovation Communities'. Analyze what the entity DOES, not just their legal form."
    )
    contact_persons: list[str] = Field(default_factory=list, description="Names of key people with their titles")
    emails: list[str] = Field(default_factory=list, description="All email addresses, prioritizing organizational emails (info@, kontakt@, contact@, etc)")
    phone_numbers: list[str] = Field(default_factory=list, description="All phone numbers")
    brief_description: str = Field(default="", description="2-3 sentence description of what entity does")
    ce_relation: str = Field(default="", description="How entity contributes to Circular Economy")

    # Legacy fields
    ce_activities: list[str] = Field(default_factory=list, description="Specific CE activities as simple strings")
    capabilities_offered: list[str] = Field(default_factory=list, description="Legacy capabilities list")
    needs_requirements: list[str] = Field(default_factory=list, description="Legacy needs list")
    capability_categories: list[str] = Field(default_factory=list, description="Legacy capability categories")
    partners: list[str] = Field(default_factory=list, description="Legacy partner names list")
    partner_urls: list[str] = Field(default_factory=list, description="Legacy partner URLs list")

    # SIMPLIFIED CE-focused fields using dicts instead of nested models
    # Format: [{"activity_name": "...", "description": "...", "category": "..."}]
    ce_activities_structured: list[dict] = Field(
        default_factory=list,
        description='CE activities as list of dicts with keys: activity_name, description, category. Example: [{"activity_name": "Metal Recycling", "description": "Recycling scrap metals", "category": "Material Recovery"}]'
    )
    ce_capabilities_offered: list[dict] = Field(
        default_factory=list,
        description='CE capabilities as list of dicts with keys: capability_name, description, category. Categories: Recycling Infrastructure, Material Recovery, Product Life Extension, Repair Services, Sustainable Materials, Waste Management, CE Consulting. Example: [{"capability_name": "Scrap Metal Processing", "description": "Process all types of metals", "category": "Material Recovery"}]'
    )
    ce_needs_requirements: list[dict] = Field(
        default_factory=list,
        description='CE needs as list of dicts with keys: need_name, description, category. Categories: Sustainable Material Sourcing, CE Partnerships, Recycling Infrastructure Access, CE Funding, Circular Design Expertise. Example: [{"need_name": "Recycled Materials", "description": "Need access to recycled plastics", "category": "Sustainable Material Sourcing"}]'
    )
    mentioned_partners: list[dict] = Field(
        default_factory=list,
        description='Partners as list of dicts with keys: name, url (optional), context. Example: [{"name": "Cahul Recycling", "url": "https://...", "context": "Partnership for waste processing"}]'
    )
    discovered_entities: list[dict] = Field(
        default_factory=list,
        description='Other CE entities found as list of dicts with keys: name, url, brief_description, context. Example: [{"name": "EcoTech GmbH", "url": "https://...", "brief_description": "Recycling company", "context": "Mentioned as partner"}]'
    )

    address: str = Field(default="", description="Complete physical address in Cahul with postal code")


class RelationshipCandidate(BaseModel):
    """Pydantic model for a single relationship candidate (kept as model for relationship analysis)."""
    target_entity: str = Field(description="Name of the target entity in this relationship")
    relationship_type: str = Field(
        description="Type of relationship. Must be one of: 'partnership', 'knowledge_transfer', 'potential_synergy'"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for this relationship (0.0-1.0)")
    evidence: str = Field(description="Specific evidence or reasoning for why this relationship exists or could exist")
    bidirectional: bool = Field(description="True if the relationship goes both ways, False if it's directional")


class KnowledgeTransferAnalysisResult(BaseModel):
    """Pydantic model for analyzing knowledge transfer potential between two entities."""
    has_potential: bool = Field(description="Whether there is knowledge transfer potential between these entities")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the knowledge transfer potential")
    transfer_type: str = Field(
        default="",
        description="Type of knowledge transfer: 'research_to_industry', 'industry_to_research', 'peer_collaboration', or empty if no potential"
    )
    reasoning: str = Field(description="Detailed reasoning for the assessment")
    suggested_collaboration_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas where collaboration could happen"
    )


class BatchedKTResult(BaseModel):
    """Pydantic model for batched knowledge transfer analysis results."""
    results: list[KnowledgeTransferAnalysisResult] = Field(
        description="List of knowledge transfer analysis results, one for each consumer entity in the batch"
    )


class SynergyCandidate(BaseModel):
    """Pydantic model for a potential synergy between entities."""
    entity_names: list[str] = Field(description="Names of entities involved in this synergy (2 or more)")
    synergy_type: str = Field(
        description="Type of synergy: 'complementary_activities', 'value_chain', 'geographic_cluster', 'shared_resources'"
    )
    description: str = Field(description="Description of the synergy and how entities could benefit")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for this synergy opportunity")
    potential_impact: str = Field(description="Expected impact: 'high', 'medium', 'low'")


class EcosystemGapAnalysisResult(BaseModel):
    """Pydantic model for ecosystem gap analysis."""
    identified_gaps: list[str] = Field(
        default_factory=list,
        description="List of gaps in the ecosystem (missing stakeholders, underrepresented sectors, broken value chains)"
    )
    underrepresented_roles: list[str] = Field(
        default_factory=list,
        description="Ecosystem roles that are underrepresented or missing"
    )
    missing_connections: list[str] = Field(
        default_factory=list,
        description="Connections or relationships that should exist but are missing"
    )
    geographic_gaps: list[str] = Field(
        default_factory=list,
        description="Geographic areas within Cahul region that lack CE actors"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations to address the identified gaps"
    )


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a simple heuristic: 1 token ≈ 4 characters for English text.
    This is a conservative estimate suitable for prompt size validation.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def call_ollama_chat(
    prompt: str,
    text_content: str,
    response_model: type[T],
    model: str = "qwen2.5:32b-instruct-q4_K_M",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = None,
    num_ctx: int = None
) -> T:
    """
    Call Ollama with structured output using Pydantic model (synchronous).

    Following best practices from https://ollama.com/blog/structured-outputs

    Args:
        prompt: The instruction prompt
        text_content: The content to analyze
        response_model: Pydantic model class for response validation
        model: Ollama model name
        base_url: Ollama API base URL
        temperature: Temperature for generation (0.0 = deterministic)
        max_tokens: Maximum tokens to generate (default: 2000, use higher for large responses)
        num_ctx: Context window size for input prompt (default: None = Ollama default ~4096)
                 Set to 32768 for large prompts (clustering, etc.)

    Returns:
        Instance of response_model with validated data
    """
    url = f"{base_url}/api/chat"

    # Auto-detect if this is entity matching or clustering (needs more tokens for output)
    model_name = response_model.__name__
    is_clustering_or_matching = 'Matching' in model_name or 'Clustering' in model_name

    if max_tokens is None:
        if is_clustering_or_matching:
            max_tokens = 8000  # Large responses need more tokens for OUTPUT
        else:
            max_tokens = 2000  # Default for normal extraction

    # Build options dict
    options = {
        "temperature": temperature,
        "top_p": 0.1,  # Reduce randomness
        "repeat_penalty": 1.1,  # Discourage repetition/hallucination
        "num_predict": max_tokens
    }

    # Add num_ctx if explicitly specified (optional, for advanced use cases)
    # Note: We keep 4096 default to avoid VRAM issues; use smaller batches instead
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a data extraction and inference assistant specializing in circular economy analysis. CRITICAL RULES: (1) If you see recycling, waste management, repair, reuse, refurbishment, remanufacturing, or sustainable materials in the text - these ARE circular economy activities and you MUST extract them. (2) You MUST populate ce_capabilities_offered, ce_activities_structured, and ce_needs_requirements fields whenever ANY circular economy keywords appear in the description, entity name, or URL. (3) Follow the concrete examples in the user prompt exactly - if the example shows how to populate fields for 'recycling services', apply the same pattern. (4) Be proactive: infer CE relevance even from indirect indicators. (5) Empty lists should NEVER occur if the entity name contains CE keywords (e.g., 'recycling', 'waste') or the description mentions sustainability/materials/waste topics."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nExtracted Information:\n{text_content}"
            }
        ],
        "format": response_model.model_json_schema(),  # Pydantic schema
        "stream": False,  # Critical: disable streaming for JSON
        "options": options
    }

    try:
        # Increased timeout to 180 seconds for large models
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        # Extract and validate with Pydantic
        content = result.get('message', {}).get('content', '{}')
        return response_model.model_validate_json(content)

    except Exception as e:
        raise RuntimeError(f"Ollama API call failed: {e}")


async def call_ollama_chat_async(
    prompt: str,
    text_content: str,
    response_model: type[T],
    client: httpx.AsyncClient,
    model: str = "qwen2.5:32b-instruct-q4_K_M",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = None,
    num_ctx: int = None
) -> T:
    """
    Call Ollama with structured output using Pydantic model (asynchronous).

    This async version allows concurrent LLM calls for 3-5x speedup on I/O-bound operations.

    Args:
        prompt: The instruction prompt
        text_content: The content to analyze
        response_model: Pydantic model class for response validation
        client: httpx.AsyncClient instance (reuse for connection pooling)
        model: Ollama model name
        base_url: Ollama API base URL
        temperature: Temperature for generation (0.0 = deterministic)
        max_tokens: Maximum tokens to generate (default: 2000, use higher for large responses)
        num_ctx: Context window size for input prompt (default: None = Ollama default ~4096)
                 Set to 32768 for large prompts (clustering, etc.)

    Returns:
        Instance of response_model with validated data
    """
    url = f"{base_url}/api/chat"

    # Auto-detect if this is entity matching or clustering (needs more tokens for output)
    model_name = response_model.__name__
    is_clustering_or_matching = 'Matching' in model_name or 'Clustering' in model_name

    if max_tokens is None:
        if is_clustering_or_matching:
            max_tokens = 8000  # Large responses need more tokens for OUTPUT
        else:
            max_tokens = 2000  # Default for normal extraction

    # Build options dict
    options = {
        "temperature": temperature,
        "top_p": 0.1,  # Reduce randomness
        "repeat_penalty": 1.1,  # Discourage repetition/hallucination
        "num_predict": max_tokens
    }

    # Add num_ctx if explicitly specified (optional, for advanced use cases)
    # Note: We keep 4096 default to avoid VRAM issues; use smaller batches instead
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a data extraction and inference assistant specializing in circular economy analysis. CRITICAL RULES: (1) If you see recycling, waste management, repair, reuse, refurbishment, remanufacturing, or sustainable materials in the text - these ARE circular economy activities and you MUST extract them. (2) You MUST populate ce_capabilities_offered, ce_activities_structured, and ce_needs_requirements fields whenever ANY circular economy keywords appear in the description, entity name, or URL. (3) Follow the concrete examples in the user prompt exactly - if the example shows how to populate fields for 'recycling services', apply the same pattern. (4) Be proactive: infer CE relevance even from indirect indicators. (5) Empty lists should NEVER occur if the entity name contains CE keywords (e.g., 'recycling', 'waste') or the description mentions sustainability/materials/waste topics."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nExtracted Information:\n{text_content}"
            }
        ],
        "format": response_model.model_json_schema(),  # Pydantic schema
        "stream": False,  # Critical: disable streaming for JSON
        "options": options
    }

    try:
        # Increased timeout to 180 seconds for large models
        response = await client.post(url, json=payload, timeout=180.0)
        response.raise_for_status()
        result = response.json()

        # Extract and validate with Pydantic
        content = result.get('message', {}).get('content', '{}')
        return response_model.model_validate_json(content)

    except Exception as e:
        raise RuntimeError(f"Ollama API call failed: {e}")

