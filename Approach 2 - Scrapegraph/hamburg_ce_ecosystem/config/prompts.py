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

EXTRACTION_PROMPT = r'''
Structure the provided information into the following format. ONLY use information explicitly stated in the text. Do not infer or generate information.

1. Entity Name: Extract the official name (if not found, use "Unknown")
2. Ecosystem Role: Based on organization type mentioned, choose EXACTLY ONE from:
   - Students (student organizations, initiatives)
   - Researchers (individual researchers, research groups)
   - Higher Education Institutions (universities, colleges, Hochschule)
   - Research Institutes (research centers, Fraunhofer, Max Planck, etc.)
   - Non-Governmental Organizations (NGOs, e.V., foundations, Vereine)
   - Industry Partners (companies, GmbH, AG, manufacturers, consultancies)
   - Startups and Entrepreneurs (startups, new ventures, incubators)
   - Public Authorities (Behörde, government bodies, municipal services)
   - Policy Makers (legislative bodies, policy institutes)
   - End-Users (consumer groups, user communities)
   - Citizen Associations (community groups, neighborhood initiatives, Bürgervereine)
   - Media and Communication Partners (news outlets, PR agencies, media companies)
   - Funding Bodies (grant organizations, investment funds, Förderer)
   - Knowledge and Innovation Communities (innovation hubs, networks, clusters)
3. Contact Persons: Extract names with titles (empty list if not found)
4. Emails: Extract all email addresses (empty list if not found)
5. Phone Numbers: Extract all phone numbers (empty list if not found)
6. Brief Description: Extract 2-3 sentence description (empty string if not clear)
7. CE Relation: Extract how they contribute to CE (empty string if not mentioned)
8. CE Activities: Extract specific CE activities (empty list if not found)
9. Partners: Extract partner names (empty list if not mentioned)
10. Partner URLs: Extract full partner URLs (empty list if not found)
11. Address: Extract complete Hamburg address (empty string if not found)

CRITICAL: Only extract information that is explicitly present. Use empty strings/lists when information is missing. Do not guess or generate data.
'''

__all__ = ["VERIFICATION_PROMPT", "EXTRACTION_PROMPT"]
