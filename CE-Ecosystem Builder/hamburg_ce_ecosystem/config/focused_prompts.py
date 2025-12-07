from __future__ import annotations

BASIC_INFO_PROMPT = r'''
Extract the core identification information for this Hamburg-based entity.

CONTEXT: You are analyzing an organization's website to understand who they are,
what they do, and where they're located in the Hamburg circular economy ecosystem.

1. ENTITY NAME:
   Extract the official organization/company name from:
   - Website header/logo text
   - <title> tag or page title
   - "About Us" or "Über uns" section
   - Footer company information
   - Impressum/Imprint section
   - Meta tags

   CRITICAL RULES:
   - DO NOT use "Unknown"
   - If official name not clearly stated, derive from domain name
     Examples: "recyclabs.de" → "Recyclabs", "meha-umwelt.de" → "MeHa Umwelt"
   - ONLY use "Unknown" if website is completely broken/inaccessible

2. ECOSYSTEM ROLE:
   Carefully analyze what this entity DOES and WHO they are.
   DO NOT just look at organization type (e.g., GmbH, e.V.) - understand their PRIMARY ACTIVITY.

   Choose EXACTLY ONE from:
   - Students: Student organizations, student initiatives, student-led projects
   - Researchers: Individual researchers, research groups (not institutes)
   - Higher Education Institutions: Universities, colleges, Hochschule, HAW, TUHH
   - Research Institutes: Research centers, Fraunhofer, Max Planck, independent research
   - Non-Governmental Organizations: NGOs, e.V., foundations, Vereine (advocacy/social causes)
   - Industry Partners: Established companies, manufacturers, consultancies, service providers
   - Startups and Entrepreneurs: New ventures, startups, incubators, young companies
   - Public Authorities: Government bodies, Behörde, municipal services, Stadt Hamburg
   - Policy Makers: Legislative bodies, policy institutes, think tanks
   - End-Users: Consumer groups, user communities, customer organizations
   - Citizen Associations: Community groups, neighborhood initiatives, Bürgervereine
   - Media and Communication Partners: News outlets, journalists, PR agencies, publishers
   - Funding Bodies: Grant organizations, investment funds, Förderer, venture capital
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
   Empty string if not clear.

4. ADDRESS:
   Extract the COMPLETE physical address with ALL components.

   CRITICAL: Check MULTIPLE locations on website:
   - Impressum section (legally required in Germany - HIGHEST PRIORITY!)
   - Contact page ("Kontakt", "Contact Us")
   - Footer (bottom of every page)
   - About page ("Über uns")
   - Location/Standort page

   Extract in format: "Street Number, Postal Code City"
   Examples:
   - "Hovestraße 30, 20539 Hamburg"
   - "Harburger Schloßstraße 6-12, 21079 Hamburg"
   - "Blohmstr. 15, 21079 Hamburg"

   IMPORTANT: German companies MUST have Impressum by law.
   Look for postal code pattern: 5 digits starting with 2 (Hamburg: 20000-22999)

   Empty string ONLY if absolutely no address found after checking all locations.

CRITICAL: Only extract information explicitly present. Use empty strings when missing.
DO NOT guess or generate data.
'''

CONTACT_INFO_PROMPT = r'''
Extract ALL contact information for this Hamburg organization.

CONTEXT: We need to be able to reach this entity for circular economy collaboration
opportunities. Prioritize organizational contacts over personal ones.

1. EMAILS:
   Prioritize organizational emails:
   - info@, kontakt@, contact@, hello@, hallo@, mail@, office@
   - Department emails: verwaltung@, sekretariat@, buchhaltung@
   - General company emails

   Where to look:
   - Contact page ("Kontakt", "Contact Us")
   - Footer (bottom of every page)
   - Impressum section (legal requirement in Germany - check here first!)
   - Team/About pages (Über uns, Team)
   - Header contact bar

   ONLY extract explicitly shown emails - do NOT construct or infer.
   Examples of what to extract:
   - "info@recyclabs.de" → extract it
   - "christoph.alberti@recyclabs.de" → extract it
   - "For inquiries: kontakt@meha-umwelt.de" → extract "kontakt@meha-umwelt.de"

   Empty list if no emails found.

2. PHONE NUMBERS:
   Extract all phone numbers exactly as shown.

   Common German formats:
   - +49 40 123456 (Hamburg landline with country code)
   - 040/123456 (Hamburg landline without country code)
   - +49 172 1234567 (mobile)
   - (040) 123-456 (alternative format)

   Where to look:
   - Contact page
   - Footer
   - Impressum section
   - Header contact bar

   Extract exactly as written, including formatting.
   Empty list if not found.

3. CONTACT PERSONS:
   Extract names WITH titles/roles.

   Format: "Title/Dr. First Last, Role/Position"
   Examples:
   - "Dr. Christoph Alberti, CEO"
   - "Maria Schmidt, Projektleiterin"
   - "Prof. Dr. Thomas Müller, Geschäftsführer"
   - "Hans Weber, Vertrieb"

   Where to look:
   - Team page ("Team", "Über uns", "About Us")
   - About/Leadership section
   - Impressum (Geschäftsführer, Vertretungsberechtigte)
   - Contact page

   ONLY extract if both name AND role/title are clearly stated.
   Empty list if not found.

IMPORTANT: Empty lists if not found. Do not guess or generate contact information.
Only extract contacts that are explicitly visible on the website.
'''

CE_CAPABILITIES_PROMPT = r'''
Extract the circular economy capabilities this entity can PROVIDE to others.

CONTEXT: We are mapping the Hamburg CE ecosystem to identify who can provide what
services, technologies, materials, or expertise. Many entities contribute to CE
through their CORE BUSINESS without using "circular economy" terminology explicitly.

CRITICAL - YOU MUST INFER CE RELEVANCE:
Many businesses ARE circular economy actors even if they don't say "circular economy"!

INFERENCE RULES - Apply to every entity:

**IF YOU SEE recycling/waste/materials → THEN populate CE capabilities:**

1. RECYCLING & MATERIAL RECOVERY:
   Keywords: "recycling", "recycler", "scrap metal", "waste processing", "material recovery"
   → IS a CE capability (Category: "Material Recovery")

   Examples:
   - "offers recycling services" → Capability: "Recycling Services"
   - "scrap metal processing" → Capability: "Scrap Metal Processing"
   - "PET recycling technology" → Capability: "PET Recycling Technology"

2. WASTE MANAGEMENT:
   Keywords: "waste management", "waste collection", "disposal", "Abfallwirtschaft"
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

   Examples:
   - "second-hand marketplace" → Capability: "Second-Hand Trading Platform"

6. CIRCULAR DESIGN & CONSULTING:
   Keywords: "circular design", "sustainability consulting", "LCA", "eco-design"
   → IS a CE capability (Category: "CE Consulting")

**CONCRETE EXAMPLES - Follow these exact patterns:**

EXAMPLE 1: Text says "offers recycling services for scrap metal"
→ YOU MUST extract:
{
  "capability_name": "Scrap Metal Recycling",
  "description": "Recycling services for scrap metal",
  "category": "Material Recovery"
}

EXAMPLE 2: Text says "manages recycling centers and waste disposal"
→ YOU MUST extract:
{
  "capability_name": "Recycling Center Management",
  "description": "Operating recycling centers and waste disposal facilities",
  "category": "Recycling Infrastructure"
}

EXAMPLE 3: Text says "repair services for electronics"
→ YOU MUST extract:
{
  "capability_name": "Electronics Repair",
  "description": "Repair services for electronic devices",
  "category": "Product Life Extension"
}

EXAMPLE 4: Text says "PET-Verpackungs- und Textilabfälle in recyceltes PET"
→ YOU MUST extract:
{
  "capability_name": "PET Recycling Technology",
  "description": "Technology for recycling PET packaging and textile waste into food-grade recycled PET",
  "category": "Material Recovery"
}

**OUTPUT FORMAT:**
List of capabilities, each with:
- capability_name: Short, descriptive name
- description: What they provide (1-2 sentences)
- category: One of [Recycling Infrastructure, Material Recovery, Product Life Extension,
            Repair Services, Sustainable Materials, Waste Management, CE Consulting,
            Circular Design, Reuse & Redistribution]

IMPORTANT: Be PROACTIVE! If you see ANY recycling, repair, reuse, waste management,
sustainable materials, or circular activities → EXTRACT as CE capabilities.

Only EXCLUDE purely administrative functions with no CE relevance (e.g., general HR,
generic marketing, unrelated IT services).

Empty list ONLY if entity truly has no CE-related capabilities whatsoever.
'''

CE_ACTIVITIES_PROMPT = r'''
Extract the circular economy activities and contributions of this entity.

CONTEXT: We want to understand HOW this entity contributes to the circular economy
and WHAT specific CE activities they engage in. Look for both explicit CE mentions
and implicit CE activities.

1. CE RELATION:
   Describe in 2-3 sentences HOW they contribute to circular economy.

   Look for:
   - Explicit CE statements: "We contribute to circular economy by..."
   - Sustainability mission statements
   - CE projects or initiatives they run
   - How their core business supports CE principles

   Be INFERENTIAL:
   - If they recycle → "Contributes by recovering materials from waste streams"
   - If they repair → "Extends product lifespans through repair services"
   - If they use recycled materials → "Uses secondary materials in production"

   Empty string if no CE contribution can be identified.

2. CE ACTIVITIES STRUCTURED:
   Extract specific CE activities they perform.

   Format: List of dicts with keys: activity_name, description, category

   Categories:
   - Waste Management
   - Material Recovery
   - Circular Design
   - Recycling
   - Repair Services
   - Remanufacturing
   - Product Life Extension
   - Sustainable Procurement
   - Reuse & Redistribution
   - CE Education/Training

   **INFERENCE RULES:**

   IF text mentions → THEN create activity:

   - "recycling operations" →
     {"activity_name": "Recycling Operations", "description": "...", "category": "Recycling"}

   - "waste collection" →
     {"activity_name": "Waste Collection", "description": "...", "category": "Waste Management"}

   - "repair workshops" →
     {"activity_name": "Repair Workshops", "description": "...", "category": "Repair Services"}

   - "sustainable procurement" →
     {"activity_name": "Sustainable Sourcing", "description": "...", "category": "Sustainable Procurement"}

   Examples:
   - Text: "processes PET waste into recycled materials"
     → {"activity_name": "PET Waste Processing", "description": "Processing PET waste into recycled materials", "category": "Material Recovery"}

   - Text: "operates 12 recycling centers in Hamburg"
     → {"activity_name": "Recycling Center Operations", "description": "Operating 12 recycling centers across Hamburg", "category": "Waste Management"}

   - Text: "chemical recycling of plastics"
     → {"activity_name": "Chemical Plastic Recycling", "description": "Chemical recycling processes for plastic materials", "category": "Material Recovery"}

3. CE ACTIVITIES (Legacy):
   Simple list of CE activity names (backward compatibility).
   Extract the same activities as above but as simple strings.

   Examples: ["PET Recycling", "Waste Processing", "Material Recovery"]

IMPORTANT: Look at what the entity DOES, not just what labels they use.
A recycling company IS doing CE activities even if they never say "circular economy".

Empty list/string if no CE activities identified.
'''

CE_NEEDS_PROMPT = r'''
Extract the circular economy needs and requirements of this entity.

CONTEXT: Understanding what entities NEED helps us identify collaboration opportunities
and ecosystem gaps. Look for explicitly stated needs and infer implicit requirements.

1. CE NEEDS/REQUIREMENTS STRUCTURED:
   What does this entity need to support their CE activities?

   Format: List of dicts with keys: need_name, description, category

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
   - Research institute → may need: industry partners, funding, real-world testbeds

   Examples:

   - Text: "seeking access to recycled plastics for production"
     → {"need_name": "Recycled Plastic Supply", "description": "Access to recycled plastics for manufacturing", "category": "Sustainable Material Sourcing"}

   - Text: "looking for industry partners for pilot projects"
     → {"need_name": "Industry Partnerships", "description": "Industry partners for pilot project implementation", "category": "CE Partnerships"}

   - Text: "require funding for scaling recycling technology"
     → {"need_name": "Scale-up Funding", "description": "Investment for scaling recycling technology", "category": "CE Funding"}

   - Text: "need collection infrastructure for product take-back"
     → {"need_name": "Take-back Infrastructure", "description": "Collection network for product returns", "category": "Product Take-back Networks"}

2. NEEDS REQUIREMENTS (Legacy):
   Simple list of need names for backward compatibility.

   Examples: ["Recycled Materials", "Partnership Opportunities", "Funding"]

3. CAPABILITY CATEGORIES (Legacy):
   High-level categories of their needs/capabilities.
   Empty list if unclear.

IMPORTANT: Be thoughtful but don't over-infer. Only extract needs that are:
1. Explicitly stated on the website, OR
2. Clearly implied by their CE activities

Do NOT assume generic business needs (office space, general IT) unless they're
specifically CE-related.

Empty lists if no CE-related needs identified.
'''

PARTNERSHIPS_PROMPT = r'''
Extract partnership and collaboration information from this entity's website.

CONTEXT: Understanding the network of partnerships helps us map the Hamburg CE
ecosystem and identify collaboration patterns and potential new connections.

1. MENTIONED PARTNERS:
   Extract organizations/companies mentioned as partners, collaborators, or network members.

   Format: List of dicts with keys: name, url (optional), context

   **WHERE TO LOOK:**
   - Dedicated "Partners" or "Partner" page
   - "About Us" section mentioning collaborators
   - "Network" or "Members" pages
   - Project descriptions naming partners
   - Case studies mentioning clients/partners
   - "References" or "Referenzen" sections
   - Footer partner logos

   **WHAT TO EXTRACT:**
   - Partner organizations (by name)
   - Their website URL if provided
   - Context of partnership

   Examples:

   - Text: "In partnership with Hamburg Wasser for sustainable water management"
     → {"name": "Hamburg Wasser", "url": null, "context": "Partnership for sustainable water management"}

   - Text: "Founding member of Bio4Circularity Network (www.bio4circularity.eu)"
     → {"name": "Bio4Circularity Network", "url": "https://www.bio4circularity.eu", "context": "Founding member"}

   - Text: "Project partners: TUHH, Fraunhofer ISI, Stadt Hamburg"
     → [
       {"name": "TUHH", "url": null, "context": "Project partner"},
       {"name": "Fraunhofer ISI", "url": null, "context": "Project partner"},
       {"name": "Stadt Hamburg", "url": null, "context": "Project partner"}
     ]

   Include:
   - Formal partners ("Partner", "Kooperationspartner")
   - Clients (if named in case studies/references)
   - Network members (if entity is a network/cluster)
   - Collaboration partners in projects

2. DISCOVERED ENTITIES:
   Extract OTHER organizations mentioned that could be relevant to Hamburg CE ecosystem.

   **CRITICAL SCHEMA REQUIREMENTS:**
   Each discovered entity MUST be a dict with ALL 4 keys:
   - name: string (entity name) - REQUIRED, never null/None
   - url: string (website URL) - REQUIRED, never null/None
   - brief_description: string (1 sentence description) - REQUIRED, never null/None
   - context: string (where/how mentioned) - REQUIRED, never null/None

   **VALIDATION RULES - FOLLOW EXACTLY:**
   ✓ INCLUDE entity IF AND ONLY IF:
     1. Entity has a complete, valid website URL (http:// or https://)
     2. You can write a brief description of what they do
     3. All 4 fields can be filled with non-empty strings

   ✗ SKIP entity IF ANY of these are true:
     1. No website URL provided in text
     2. URL is incomplete/unclear (e.g., just "contact us at...")
     3. Cannot determine what entity does (no brief_description possible)
     4. Already listed in "mentioned_partners" above
     5. Generic suppliers/vendors not CE-related

   **CORRECT OUTPUT EXAMPLES:**

   Text: "Read more about EcoTech GmbH's innovative recycling at www.ecotech.de"
   ✓ CORRECT:
   {
     "name": "EcoTech GmbH",
     "url": "https://www.ecotech.de",
     "brief_description": "Innovative recycling company",
     "context": "Mentioned in blog post about recycling innovations"
   }

   Text: "Green Startup Hub (greenstartup.hamburg) supports CE startups"
   ✓ CORRECT:
   {
     "name": "Green Startup Hub",
     "url": "https://greenstartup.hamburg",
     "brief_description": "Startup incubator supporting circular economy ventures",
     "context": "Referenced as supporter of CE startups"
   }

   **INCORRECT OUTPUT EXAMPLES (DO NOT DO THIS!):**

   ✗ WRONG - null/None for url:
   {"name": "Some Company", "url": null, "brief_description": "...", "context": "..."}
   → INSTEAD: Skip this entity entirely (do not include it)

   ✗ WRONG - missing brief_description key:
   {"name": "Some Company", "url": "https://...", "context": "..."}
   → INSTEAD: Add brief_description or skip entity if you cannot describe it

   ✗ WRONG - empty string for required field:
   {"name": "Company", "url": "", "brief_description": "...", "context": "..."}
   → INSTEAD: Skip entity if any required field would be empty

   **WHERE TO FIND DISCOVERED ENTITIES:**
   - Blog posts mentioning other CE organizations with links
   - News sections referencing external entities
   - Resource/link pages with URLs
   - "Related organizations" or "Netzwerk" sections with websites
   - Project descriptions with partner websites

   **IMPORTANT:**
   - Return EMPTY LIST if no valid discovered entities found (this is OK!)
   - Quality over quantity - only include entities with complete, valid data
   - When in doubt, SKIP the entity rather than include with null/incomplete data

3. PARTNERS (Legacy):
   Simple list of partner names for backward compatibility.
   Extract from mentioned_partners.

   Examples: ["Hamburg Wasser", "TUHH", "Fraunhofer ISI"]

4. PARTNER URLS (Legacy):
   Simple list of partner URLs for backward compatibility.
   Extract from mentioned_partners where available.

   Examples: ["https://www.bio4circularity.eu", "https://www.tuhh.de"]

5. CAPABILITIES OFFERED (Legacy):
   Simple list of general capabilities (non-CE specific).
   This is for backward compatibility - can be empty list.

IMPORTANT:
- Only extract partnerships/entities explicitly mentioned on the website
- For discovered_entities: URL is REQUIRED (skip if no URL)
- Distinguish between formal partners and casually mentioned entities
- Empty lists if no partners/entities found

Do not generate or assume partnerships that aren't explicitly stated.
'''

__all__ = [
    "BASIC_INFO_PROMPT",
    "CONTACT_INFO_PROMPT",
    "CE_CAPABILITIES_PROMPT",
    "CE_ACTIVITIES_PROMPT",
    "CE_NEEDS_PROMPT",
    "PARTNERSHIPS_PROMPT",
]
