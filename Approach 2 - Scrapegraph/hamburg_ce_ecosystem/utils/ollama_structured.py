"""Direct Ollama structured output integration using Pydantic (best practice)."""
from __future__ import annotations

import requests
from typing import Dict, Any, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)


class VerificationResult(BaseModel):
    """Pydantic model for Hamburg/CE verification."""
    is_hamburg_based: bool = Field(description="Is entity based in Hamburg, Germany?")
    hamburg_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for Hamburg location")
    hamburg_evidence: str = Field(description="Evidence found for Hamburg location")
    is_ce_related: bool = Field(description="Is entity related to Circular Economy?")
    ce_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for CE relation")
    ce_evidence: str = Field(description="Evidence found for CE relation")


class ExtractionResult(BaseModel):
    """Pydantic model for entity extraction with 14 ecosystem roles."""
    entity_name: str = Field(description="Official organization/company name")
    ecosystem_role: str = Field(
        description="MUST be one of: Students, Researchers, Higher Education Institutions, Research Institutes, Non-Governmental Organizations, Industry Partners, Startups and Entrepreneurs, Public Authorities, Policy Makers, End-Users, Citizen Associations, Media and Communication Partners, Funding Bodies, Knowledge and Innovation Communities"
    )
    contact_persons: list[str] = Field(default_factory=list, description="Names of key people with their titles")
    emails: list[str] = Field(default_factory=list, description="All email addresses")
    phone_numbers: list[str] = Field(default_factory=list, description="All phone numbers")
    brief_description: str = Field(default="", description="2-3 sentence description of what entity does")
    ce_relation: str = Field(default="", description="How entity contributes to Circular Economy")
    ce_activities: list[str] = Field(default_factory=list, description="Specific CE activities, services, or products")
    partners: list[str] = Field(default_factory=list, description="Names of partner organizations")
    partner_urls: list[str] = Field(default_factory=list, description="Full website URLs of partners")
    address: str = Field(default="", description="Complete physical address in Hamburg with postal code")


def call_ollama_chat(
    prompt: str,
    text_content: str,
    response_model: type[T],
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0
) -> T:
    """
    Call Ollama with structured output using Pydantic model.
    
    Following best practices from https://ollama.com/blog/structured-outputs
    
    Args:
        prompt: The instruction prompt
        text_content: The content to analyze  
        response_model: Pydantic model class for response validation
        model: Ollama model name
        base_url: Ollama API base URL
        temperature: Temperature for generation (0.0 = deterministic)
    
    Returns:
        Instance of response_model with validated data
    """
    url = f"{base_url}/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise data extraction assistant. ONLY extract information that is explicitly stated in the provided text. If information is not found, use empty strings or empty lists. DO NOT make assumptions or generate information that is not in the source text. DO NOT hallucinate."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nExtracted Information:\n{text_content}"
            }
        ],
        "format": response_model.model_json_schema(),  # Pydantic schema
        "stream": False,  # Critical: disable streaming for JSON
        "options": {
            "temperature": temperature,
            "top_p": 0.1,  # Reduce randomness
            "repeat_penalty": 1.1,  # Discourage repetition/hallucination
            "num_predict": 2000
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        
        # Extract and validate with Pydantic
        content = result.get('message', {}).get('content', '{}')
        return response_model.model_validate_json(content)
    
    except Exception as e:
        raise RuntimeError(f"Ollama API call failed: {e}")

