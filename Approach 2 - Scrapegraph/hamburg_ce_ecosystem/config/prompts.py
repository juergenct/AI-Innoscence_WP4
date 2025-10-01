from __future__ import annotations

VERIFICATION_PROMPT = r'''
Analyze this website and determine:

1. Is the entity based in Hamburg, Germany?
   Look for: postal codes 20000-22999, phone numbers starting with +49 40, mentions of Hamburg districts (Altona, Eimsbüttel, Wandsbek, Bergedorf, Harburg), or "Hamburg" in contact/address sections.

2. Is the entity related to Circular Economy?
   Look for keywords: Kreislaufwirtschaft, Recycling, Nachhaltigkeit, Zero Waste, Ressourceneffizienz, circular economy, waste management, reuse, repair, remanufacturing.

Provide confidence scores between 0.0 and 1.0, and evidence strings explaining what you found.
'''

EXTRACTION_PROMPT = r'''
Extract detailed information from this Hamburg-based Circular Economy entity:

1. Entity Name: The official organization or company name
2. Ecosystem Role: Choose ONE from: Students, Researchers, Higher Education Institutions, Research Institutes, Non-Governmental Organizations, Industry Partners, Startups and Entrepreneurs, Public Authorities, Policy Makers, End-Users, Citizen Associations, Media and Communication Partners, Funding Bodies, Knowledge and Innovation Communities
3. Contact Persons: Names of key people (CEO, directors, managers)
4. Emails: Email addresses found on the website
5. Phone Numbers: Phone numbers in any format
6. Brief Description: A 2-3 sentence description of what the entity does
7. CE Relation: Explain how this entity contributes to Circular Economy
8. CE Activities: List specific CE-related activities, services, or products
9. Partners: Names of partner organizations mentioned
10. Partner URLs: Website URLs of partners if available
11. Address: Physical address in Hamburg (street, postal code, city)
'''

__all__ = ["VERIFICATION_PROMPT", "EXTRACTION_PROMPT"]
