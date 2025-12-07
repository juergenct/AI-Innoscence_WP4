"""Prompts for Hamburg and Circular Economy verification.

This module contains the prompt template used to verify whether entities
are based in Hamburg and related to Circular Economy activities.
"""
from __future__ import annotations

VERIFICATION_PROMPT = r'''
Based ONLY on the extracted information below, determine:

1. Is the entity based in Hamburg, Germany?
   Check for: postal codes 20000-22999, phone numbers +49 40, Hamburg district names (Altona, Eimsbüttel, Wandsbek, Bergedorf, Harburg), "Hamburg" in addresses.
   If found: confidence 0.6-1.0 (based on strength of evidence)
   If not found: confidence 0.0, is_hamburg_based = false
   Provide evidence string with EXACT text found (or empty string if not found).

2. Is the entity related to Circular Economy?
   Check for: Kreislaufwirtschaft, Recycling, Nachhaltigkeit, Zero Waste, Ressourceneffizienz, circular economy, waste management, reuse, repair, remanufacturing.
   If found: confidence based on number and relevance of keywords
   If not found: confidence 0.0, is_ce_related = false
   Provide evidence string with EXACT keywords found (or empty string if not found).

CRITICAL: Only use information from the provided text. Do not make assumptions. If information is missing, set to false/0.0/empty string.
'''

__all__ = ["VERIFICATION_PROMPT"]
