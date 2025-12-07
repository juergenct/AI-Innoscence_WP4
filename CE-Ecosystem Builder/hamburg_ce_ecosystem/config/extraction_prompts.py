"""Extraction prompts for parallel focused field-by-field extraction.

This module contains 6 focused prompt templates used by the Instructor-enhanced
extraction pipeline. Each prompt targets a specific domain (basic info, contacts,
CE capabilities, CE activities, CE needs, partnerships) for better LLM performance
and reliability.

The prompts are designed to work with Pydantic models and Instructor's automatic
retry mechanism for robust structured data extraction.

IMPORTANT: All output MUST be in English regardless of source language.
"""
from __future__ import annotations

# Import taxonomy for activity classification
try:
    from .ce_activities_taxonomy import get_taxonomy_for_prompt, CE_ACTIVITIES_TAXONOMY
except ImportError:
    # Fallback if taxonomy not available
    def get_taxonomy_for_prompt():
        return ""
    CE_ACTIVITIES_TAXONOMY = {}


# =============================================================================
# LANGUAGE INSTRUCTION (included in all prompts)
# =============================================================================
ENGLISH_OUTPUT_INSTRUCTION = """
CRITICAL LANGUAGE REQUIREMENT:
- The source website may be in German, English, or other languages
- You MUST translate ALL output fields to English
- ALL text in your response must be in English
- DO NOT include German, Serbian, Romanian, or any non-English text in output
- Translate names of activities, descriptions, and other text to English
"""

BASIC_INFO_PROMPT = r'''
Extract the core identification information for this entity.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: You are analyzing an organization's website to understand who they are,
what they do, and where they're located in the circular economy ecosystem.

1. ENTITY NAME:
   Extract the official organization/company name from:
   - Website header/logo text
   - <title> tag or page title
   - "About Us" section (may be "Über uns" in German)
   - Footer company information
   - Impressum/Imprint section
   - Meta tags

   CRITICAL RULES:
   - DO NOT use "Unknown"
   - If official name not clearly stated, derive from domain name
     Examples: "recyclabs.de" → "Recyclabs", "meha-umwelt.de" → "MeHa Umwelt"
   - ONLY use "Unknown" if website is completely broken/inaccessible
   - Keep company names in original form (do not translate company names)

2. ECOSYSTEM ROLE:
   Carefully analyze what this entity DOES and WHO they are.
   DO NOT just look at organization type - understand their PRIMARY ACTIVITY.

   Choose EXACTLY ONE from:
   - Students: Student organizations, student initiatives, student-led projects
   - Researchers: Individual researchers, research groups (not institutes)
   - Higher Education Institutions: Universities, colleges, technical schools
   - Research Institutes: Research centers, independent research organizations
   - Non-Governmental Organizations: NGOs, foundations, advocacy organizations
   - Industry Partners: Established companies, manufacturers, consultancies, service providers
   - Startups and Entrepreneurs: New ventures, startups, incubators, young companies
   - Public Authorities: Government bodies, municipal services, city administration
   - Policy Makers: Legislative bodies, policy institutes, think tanks
   - End-Users: Consumer groups, user communities, customer organizations
   - Citizen Associations: Community groups, neighborhood initiatives
   - Media and Communication Partners: News outlets, journalists, PR agencies, publishers
   - Funding Bodies: Grant organizations, investment funds, venture capital
   - Knowledge and Innovation Communities: Innovation hubs, networks, clusters, platforms

   Classification guidelines:
   - Primarily conduct research → Research Institutes or Researchers
   - Teach students and grant degrees → Higher Education Institutions
   - Produce/sell products or services → Industry Partners or Startups
   - Advocate for causes (non-governmental) → Non-Governmental Organizations
   - Government entity → Public Authorities
   - Facilitate connections/collaboration → Knowledge and Innovation Communities
   - Young/new company with innovative model → Startups and Entrepreneurs
   - Established company → Industry Partners

3. BRIEF DESCRIPTION:
   Extract 2-3 sentence description of what they do.
   Focus on their main activities and purpose.
   OUTPUT MUST BE IN ENGLISH - translate if source is in another language.
   Empty string if not clear.

4. ADDRESS:
   Extract the COMPLETE physical address with ALL components.

   CRITICAL: Check MULTIPLE locations on website:
   - Impressum section (legally required in Germany)
   - Contact page
   - Footer (bottom of every page)
   - About page
   - Location page

   Extract in format: "Street Number, Postal Code City, Country"
   Examples:
   - "Hovestrasse 30, 20539 Hamburg, Germany"
   - "Harburger Schlossstrasse 6-12, 21079 Hamburg, Germany"

   Note: Keep addresses in original format but add country if missing.
   Empty string ONLY if absolutely no address found.

CRITICAL: Only extract information explicitly present. Use empty strings when missing.
DO NOT guess or generate data. ALL output must be in English.
'''

CONTACT_INFO_PROMPT = r'''
Extract ALL contact information for this organization.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: We need to be able to reach this entity for circular economy collaboration
opportunities. Prioritize organizational contacts over personal ones.

1. EMAILS:
   Prioritize organizational emails:
   - info@, contact@, hello@, office@
   - Department emails: admin@, secretary@

   Where to look:
   - Contact page
   - Footer (bottom of every page)
   - Impressum section
   - Team/About pages
   - Header contact bar

   ONLY extract explicitly shown emails - do NOT construct or infer.
   Empty list if no emails found.

2. PHONE NUMBERS:
   Extract all phone numbers exactly as shown.

   Common formats:
   - +49 40 123456 (with country code)
   - 040/123456 (local format)
   - +49 172 1234567 (mobile)

   Where to look:
   - Contact page
   - Footer
   - Impressum section
   - Header contact bar

   Extract exactly as written, including formatting.
   Empty list if not found.

3. CONTACT PERSONS:
   Extract names WITH titles/roles - TRANSLATE role titles to English.

   Format: "Title/Dr. First Last, Role/Position"
   Examples:
   - "Dr. Christoph Alberti, CEO"
   - "Maria Schmidt, Project Manager" (translate from "Projektleiterin")
   - "Prof. Dr. Thomas Mueller, Managing Director" (translate from "Geschäftsführer")

   Where to look:
   - Team page
   - About/Leadership section
   - Impressum
   - Contact page

   ONLY extract if both name AND role/title are clearly stated.
   TRANSLATE role titles to English.
   Empty list if not found.

IMPORTANT: Empty lists if not found. Do not guess or generate contact information.
Only extract contacts that are explicitly visible on the website.
'''

CE_CAPABILITIES_PROMPT = r'''
Extract the circular economy capabilities this entity can PROVIDE to others.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: We are mapping the CE ecosystem to identify who can provide what
services, technologies, materials, or expertise. Many entities contribute to CE
through their CORE BUSINESS without using "circular economy" terminology explicitly.

CRITICAL - YOU MUST INFER CE RELEVANCE:
Many businesses ARE circular economy actors even if they don't say "circular economy"!

INFERENCE RULES - Apply to every entity:

**IF YOU SEE recycling/waste/materials → THEN populate CE capabilities:**

1. RECYCLING & MATERIAL RECOVERY:
   Keywords: "recycling", "scrap metal", "waste processing", "material recovery"
   → IS a CE capability (Category: "Material Recovery")

   Examples:
   - "offers recycling services" → Capability: "Recycling Services"
   - "scrap metal processing" → Capability: "Scrap Metal Processing"
   - "PET recycling technology" → Capability: "PET Recycling Technology"

2. WASTE MANAGEMENT:
   Keywords: "waste management", "waste collection", "disposal"
   → IS a CE capability (Category: "Waste Management")

   Examples:
   - "waste collection services" → Capability: "Waste Collection Services"
   - "operates recycling centers" → Capability: "Recycling Center Operations"

3. PRODUCT LIFE EXTENSION:
   Keywords: "repair", "refurbishment", "remanufacturing", "maintenance", "spare parts"
   → IS a CE capability (Category: "Product Life Extension")

   Examples:
   - "repair services for electronics" → Capability: "Electronics Repair"
   - "refurbishment of industrial equipment" → Capability: "Industrial Equipment Refurbishment"

4. SUSTAINABLE MATERIALS:
   Keywords: "recycled materials", "secondary raw materials", "bio-based", "sustainable"
   → IS a CE capability (Category: "Sustainable Materials")

   Examples:
   - "supplies recycled plastics" → Capability: "Recycled Plastic Supply"
   - "bio-based packaging materials" → Capability: "Bio-Based Packaging"

5. REUSE & REDISTRIBUTION:
   Keywords: "reuse", "second-hand", "sharing", "rental", "leasing"
   → IS a CE capability (Category: "Reuse & Redistribution")

6. CIRCULAR DESIGN & CONSULTING:
   Keywords: "circular design", "sustainability consulting", "LCA", "eco-design"
   → IS a CE capability (Category: "CE Consulting")

**OUTPUT FORMAT:**
List of capabilities, each with:
- capability_name: Short, descriptive name IN ENGLISH
- description: What they provide (1-2 sentences) IN ENGLISH
- category: One of [Recycling Infrastructure, Material Recovery, Product Life Extension,
            Repair Services, Sustainable Materials, Waste Management, CE Consulting,
            Circular Design, Reuse & Redistribution]

IMPORTANT: ALL output must be in English. Translate any German/other language content.
Empty list ONLY if entity truly has no CE-related capabilities whatsoever.
'''

# Generate the taxonomy reference for the prompt
_TAXONOMY_REFERENCE = """
PREDEFINED CE ACTIVITIES - You MUST select from this list:

Design & Production:
  - Eco-design and design for circularity
  - Design for disassembly and modularity
  - Design for longevity and durability
  - Design for repair and maintenance
  - Design for recyclability
  - Sustainable materials selection
  - Biomimicry and bio-based design
  - Cradle-to-cradle product design
  - Lightweighting and material optimization
  - Additive manufacturing and 3D printing
  - Remanufacturing process development
  - Circular product certification

Use & Consumption:
  - Product-as-a-Service models
  - Sharing platforms and services
  - Rental and leasing services
  - Repair services and workshops
  - Maintenance and servicing
  - Product life extension services
  - Second-hand and resale platforms
  - Refurbishment services
  - Upcycling and creative reuse
  - Collaborative consumption platforms
  - Tool and equipment libraries
  - Subscription-based product access

Collection & Logistics:
  - Waste collection and sorting
  - Separate collection systems
  - Take-back schemes and programs
  - Reverse logistics operations
  - Collection point management
  - Door-to-door collection services
  - Commercial waste collection
  - Hazardous waste collection
  - E-waste collection programs
  - Textile collection services
  - Packaging return systems
  - Deposit-return schemes

Recycling & Processing:
  - Mechanical recycling
  - Chemical recycling
  - Plastic recycling and processing
  - Metal recycling and recovery
  - Paper and cardboard recycling
  - Glass recycling
  - Textile recycling
  - E-waste recycling and processing
  - Construction waste recycling
  - Battery recycling
  - Composite material recycling
  - Advanced sorting technologies

Resource Recovery:
  - Material recovery and extraction
  - Energy recovery from waste
  - Biogas production
  - Composting and organic processing
  - Anaerobic digestion
  - Precious metal recovery
  - Rare earth element recovery
  - Water recovery and reuse
  - Heat recovery systems
  - Nutrient recovery
  - Solvent recovery and recycling
  - Industrial byproduct recovery

Industrial Symbiosis:
  - Industrial symbiosis networks
  - Waste-to-resource exchanges
  - Byproduct synergy programs
  - Eco-industrial park development
  - Cross-sector material flows
  - Energy sharing between industries
  - Water sharing and cascading
  - Shared infrastructure development
  - Industrial ecosystem mapping
  - Symbiosis matchmaking platforms
  - Circular supply chain development
  - Regional material flow optimization

Digital & Technology:
  - Digital product passports
  - Material tracking and traceability
  - IoT for resource monitoring
  - AI-powered waste sorting
  - Blockchain for supply chain transparency
  - Digital twins for product lifecycle
  - Predictive maintenance systems
  - Online marketplaces for secondary materials
  - Circular economy data platforms
  - Smart waste management systems
  - Resource efficiency software
  - Life cycle assessment tools

Policy & Governance:
  - Circular economy policy development
  - Extended producer responsibility
  - Waste management regulation
  - Green public procurement
  - Circular economy standards development
  - Environmental certification programs
  - Regulatory compliance services
  - Policy advocacy and lobbying
  - Municipal circular economy programs
  - Regional circular economy strategies
  - International CE cooperation
  - Circular economy impact assessment

Education & Research:
  - Circular economy research
  - Sustainability education programs
  - Professional training and upskilling
  - Circular design education
  - Waste management training
  - Consumer awareness campaigns
  - Academic CE programs
  - Innovation labs and incubators
  - Circular economy consultancy
  - Knowledge transfer programs
  - Best practice documentation
  - Circular economy publications

Finance & Business Models:
  - Circular economy investment
  - Green financing and bonds
  - Impact investing for circularity
  - Circular business model development
  - Performance-based contracts
  - Leasing and rental business models
  - Pay-per-use business models
  - Circular startup funding
  - ESG reporting and metrics
  - Circular economy valuation
  - Risk assessment for circular projects
  - Circular economy venture capital
"""

CE_ACTIVITIES_PROMPT = r'''
Extract the circular economy activities of this entity.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: We want to understand HOW this entity contributes to the circular economy
and WHAT specific CE activities they engage in.

IMPORTANT: You MUST select activities from the PREDEFINED TAXONOMY below.
Do NOT create new activity names - match to the closest predefined activity.

''' + _TAXONOMY_REFERENCE + r'''

1. CE RELATION:
   Describe in 2-3 sentences HOW they contribute to circular economy.
   OUTPUT MUST BE IN ENGLISH.

   Be INFERENTIAL:
   - If they recycle → "Contributes by recovering materials from waste streams"
   - If they repair → "Extends product lifespans through repair services"
   - If they use recycled materials → "Uses secondary materials in production"

   Empty string if no CE contribution can be identified.

2. CE ACTIVITIES STRUCTURED:
   Extract specific CE activities they perform.
   YOU MUST SELECT FROM THE PREDEFINED TAXONOMY ABOVE.

   Format: List of dicts with keys:
   - activity_name: MUST be exactly from the taxonomy list above
   - description: Brief description of how they do this activity (IN ENGLISH)
   - category: The category from the taxonomy (e.g., "Recycling & Processing")

   **MATCHING RULES:**

   When you see this on website → Select this activity:

   - "recycling operations" → "Mechanical recycling" or "Chemical recycling"
   - "waste collection" → "Waste collection and sorting"
   - "repair workshops" → "Repair services and workshops"
   - "product rental" → "Rental and leasing services"
   - "take-back program" → "Take-back schemes and programs"
   - "circular design" → "Eco-design and design for circularity"
   - "remanufacturing" → "Remanufacturing process development"
   - "composting" → "Composting and organic processing"
   - "e-waste processing" → "E-waste recycling and processing"

   Examples:

   - Text: "processes PET waste into recycled materials"
     → {"activity_name": "Plastic recycling and processing", "description": "Processing PET waste into recycled materials", "category": "Recycling & Processing"}

   - Text: "operates recycling centers"
     → {"activity_name": "Waste collection and sorting", "description": "Operating recycling centers for material collection", "category": "Collection & Logistics"}

   - Text: "offers repair services"
     → {"activity_name": "Repair services and workshops", "description": "Providing repair services", "category": "Use & Consumption"}

3. CE ACTIVITIES (Legacy):
   Simple list of activity names from the taxonomy.
   Examples: ["Plastic recycling and processing", "Waste collection and sorting"]

CRITICAL RULES:
- ALL activity names MUST come from the predefined taxonomy
- ALL descriptions MUST be in English
- If website is in German/other language, translate the description to English
- Do NOT invent new activity names

Empty list/string if no CE activities identified.
'''

CE_NEEDS_PROMPT = r'''
Extract the circular economy needs and requirements of this entity.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: Understanding what entities NEED helps us identify collaboration opportunities
and ecosystem gaps. Look for explicitly stated needs and infer implicit requirements.

1. CE NEEDS/REQUIREMENTS STRUCTURED:
   What does this entity need to support their CE activities?

   Format: List of dicts with keys: need_name, description, category
   ALL OUTPUT MUST BE IN ENGLISH.

   Categories:
   - Sustainable Material Sourcing: Need for recycled/secondary materials
   - CE Partnerships: Looking for collaboration partners
   - Recycling Infrastructure Access: Need access to recycling facilities
   - CE Funding: Seeking grants or investment for CE projects
   - Circular Design Expertise: Need for CE design knowledge
   - Product Take-back Networks: Collection/reverse logistics
   - Waste Management Solutions: Processing capabilities
   - Market Access: Customers for circular products
   - Technology Access: CE technologies or processes

   **WHERE TO LOOK:**
   - "We are looking for..." statements
   - "We need..." or "We require..." mentions
   - "Seeking partners for..." sections
   - "Challenges" or "Goals" sections
   - Project descriptions mentioning needs
   - "Join our network" or collaboration calls

   **INFERENCE RULES:**

   IF entity type → THEN likely needs:

   - Recycler → may need: material inputs, processing partners, customers
   - Manufacturer → may need: recycled materials, CE design expertise
   - Startup → may need: funding, partnerships, infrastructure access
   - Research institute → may need: industry partners, funding, testbeds

   Examples:

   - Text: "seeking access to recycled plastics for production"
     → {"need_name": "Recycled Plastic Supply", "description": "Access to recycled plastics for manufacturing", "category": "Sustainable Material Sourcing"}

   - Text: "looking for industry partners"
     → {"need_name": "Industry Partnerships", "description": "Industry partners for collaboration", "category": "CE Partnerships"}

2. NEEDS REQUIREMENTS (Legacy):
   Simple list of need names for backward compatibility.
   Examples: ["Recycled Materials", "Partnership Opportunities", "Funding"]

3. CAPABILITY CATEGORIES (Legacy):
   High-level categories of their needs/capabilities.
   Empty list if unclear.

IMPORTANT: ALL output must be in English.
Empty lists if no CE-related needs identified.
'''

PARTNERSHIPS_PROMPT = r'''
Extract partnership and collaboration information from this entity's website.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CONTEXT: Understanding the network of partnerships helps us map the CE
ecosystem and identify collaboration patterns and potential new connections.

1. MENTIONED PARTNERS:
   Extract organizations/companies mentioned as partners, collaborators, or network members.

   Format: List of dicts with keys: name, url (optional), context

   **WHERE TO LOOK:**
   - Dedicated "Partners" page
   - "About Us" section mentioning collaborators
   - "Network" or "Members" pages
   - Project descriptions naming partners
   - Case studies mentioning clients/partners
   - "References" sections
   - Footer partner logos

   **WHAT TO EXTRACT:**
   - Partner organizations (by name)
   - Their website URL if provided
   - Context of partnership IN ENGLISH

   Examples:

   - Text: "In partnership with Hamburg Wasser for sustainable water management"
     → {"name": "Hamburg Wasser", "url": null, "context": "Partnership for sustainable water management"}

   - Text: "Founding member of Bio4Circularity Network"
     → {"name": "Bio4Circularity Network", "url": "https://www.bio4circularity.eu", "context": "Founding member"}

2. DISCOVERED ENTITIES:
   Extract OTHER organizations mentioned that could be relevant to CE ecosystem.

   **CRITICAL SCHEMA REQUIREMENTS:**
   Each discovered entity MUST be a dict with ALL 4 keys:
   - name: string (entity name) - REQUIRED
   - url: string (website URL) - REQUIRED
   - brief_description: string (1 sentence description IN ENGLISH) - REQUIRED
   - context: string (where/how mentioned IN ENGLISH) - REQUIRED

   **VALIDATION RULES:**
   ✓ INCLUDE entity IF AND ONLY IF:
     1. Entity has a complete, valid website URL
     2. You can write a brief description of what they do
     3. All 4 fields can be filled with non-empty strings

   ✗ SKIP entity IF:
     1. No website URL provided
     2. Cannot determine what entity does
     3. Already listed in "mentioned_partners"

3. PARTNERS (Legacy):
   Simple list of partner names for backward compatibility.

4. PARTNER URLS (Legacy):
   Simple list of partner URLs for backward compatibility.

5. CAPABILITIES OFFERED (Legacy):
   Simple list of general capabilities. Can be empty list.

IMPORTANT:
- ALL descriptions and contexts must be IN ENGLISH
- Only extract partnerships/entities explicitly mentioned on the website
- For discovered_entities: URL is REQUIRED (skip if no URL)
- Empty lists if no partners/entities found
'''

__all__ = [
    "BASIC_INFO_PROMPT",
    "CONTACT_INFO_PROMPT",
    "CE_CAPABILITIES_PROMPT",
    "CE_ACTIVITIES_PROMPT",
    "CE_NEEDS_PROMPT",
    "PARTNERSHIPS_PROMPT",
    "ENGLISH_OUTPUT_INSTRUCTION",
]
