"""Prompts for Novi Sad and Circular Economy verification.

This module contains the prompt template used to verify whether entities
are based in Novi Sad, Serbia and related to Circular Economy activities.

Supports Serbian (Cyrillic & Latin scripts) and English.
"""
from __future__ import annotations

VERIFICATION_PROMPT = r'''
IMPORTANT: The website content may be in Serbian (Cyrillic or Latin script) or English. Extract and understand information from ANY language, but respond in English.

Based ONLY on the extracted information below, determine:

1. Is the entity based in Novi Sad, Serbia?
   Check for:
   - Postal codes: 21000-21xxx (21000, 21101, 21102, 21105, 21113, 21124, 21137, 21138)
   - Phone numbers: +381 21 or (021) or 021/
   - District/municipality names: Novi Sad, Petrovaradin, Futog, Veternik, Begeč, Rumenka, Kisač, Sremska Kamenica
   - Address keywords: "Novi Sad", "Нови Сад", "Vojvodina", "Војводина", "Serbia", "Србија", "Srbija"

   If found: confidence 0.6-1.0 (based on strength of evidence)
   If not found: confidence 0.0, is_novi_sad_based = false
   Provide evidence string with EXACT text found (or empty string if not found).

2. Is the entity related to Circular Economy (kružna ekonomija)?
   IMPORTANT: "Circular Economy" = kružna ekonomija, NOT "CE" (which can mean European Commission/Evropska Komisija)
   Check for keywords in SERBIAN (Cyrillic), SERBIAN (Latin), or ENGLISH:

   Serbian (Cyrillic): циркуларна економија, рециклажа, одрживост, управљање отпадом, компостирање,
   поновна употреба, поправка, рециклирање, смањење отпада, нула отпада, ефикасност ресурса,
   прерада, обновљиви ресурси

   Serbian (Latin): cirkularna ekonomija, reciklaža, održivost, upravljanje otpadom, kompostiranje,
   ponovna upotreba, popravka, recikliranje, smanjenje otpada, nula otpada, efikasnost resursa,
   prerada, obnovljivi resursi

   English: circular economy, recycling, sustainability, waste management, composting, reuse, repair,
   waste reduction, zero waste, resource efficiency, processing, renewable resources, remanufacturing

   If found: confidence based on number and relevance of keywords
   If not found: confidence 0.0, is_ce_related = false
   Provide evidence string with EXACT keywords found (or empty string if not found).
   CRITICAL: Keep evidence strings SHORT (max 200 characters) - cite only the most important keywords!

CRITICAL:
- Only use information from the provided text
- Do not make assumptions
- If information is missing, set to false/0.0/empty string
- Understand Serbian Cyrillic and Latin scripts equally
- Your output must be in English
'''

__all__ = ["VERIFICATION_PROMPT"]
