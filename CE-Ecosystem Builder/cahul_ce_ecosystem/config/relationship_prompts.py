"""Relationship analysis prompts for CE ecosystem mapping.

This module contains prompts for:
- Knowledge transfer analysis
- Synergy detection (conservative, high-confidence only)
- Ecosystem gap analysis (cluster-based)
- Entity deduplication
- Entity matching
- Clustering (capabilities, needs, activities)

IMPORTANT: All output MUST be in English regardless of source language.
"""
from __future__ import annotations

# =============================================================================
# LANGUAGE INSTRUCTION (included in relevant prompts)
# =============================================================================
ENGLISH_OUTPUT_INSTRUCTION = """
CRITICAL LANGUAGE REQUIREMENT:
- ALL output text must be in English
- Translate any German, Serbian, Romanian, or other language content to English
- Entity names may remain in original language, but descriptions must be in English
"""

KNOWLEDGE_TRANSFER_ANALYSIS_PROMPT = r'''
You are analyzing the potential for knowledge transfer between two entities in a Circular Economy ecosystem.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

ENTITY 1:
Name: {entity1_name}
Role: {entity1_role}
Description: {entity1_description}
CE Activities: {entity1_activities}
CE Relation: {entity1_ce_relation}

ENTITY 2:
Name: {entity2_name}
Role: {entity2_role}
Description: {entity2_description}
CE Activities: {entity2_activities}
CE Relation: {entity2_ce_relation}

TASK: Determine if there is potential for knowledge transfer between these entities.

Knowledge transfer typically occurs between:
- Higher Education Institutions / Research Institutes → Industry Partners / Startups
- Industry Partners → Research Institutes (applied research feedback)
- Higher Education Institutions ↔ Public Authorities (policy research)
- Research Institutes ↔ NGOs (sustainability research)
- Knowledge and Innovation Communities → any stakeholder (dissemination)

Consider:
1. Do their CE activities complement each other? (e.g., one does research, other needs implementation)
2. Could one entity benefit from the other's expertise?
3. Are they working in similar CE domains where knowledge sharing would be valuable?
4. Do their roles naturally suggest a knowledge transfer relationship?

Evaluate:
- has_potential: True if meaningful knowledge transfer could occur
- confidence: 0.0-1.0 (0.7+ for strong evidence, 0.5-0.7 for moderate, 0.3-0.5 for weak)
- transfer_type: 'research_to_industry', 'industry_to_research', 'peer_collaboration', or empty
- reasoning: Explain WHY this knowledge transfer makes sense (or doesn't) - IN ENGLISH
- suggested_collaboration_areas: Specific CE topics or projects they could collaborate on - IN ENGLISH

Be conservative: Only suggest knowledge transfer if there's clear complementarity.
'''

SYNERGY_DETECTION_PROMPT = r'''
You are identifying potential synergies in a Circular Economy ecosystem.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

ENTITIES:
{entities_summary}

TASK: Identify synergies where multiple entities could benefit from collaboration.

IMPORTANT: Be CONSERVATIVE. Only report synergies with HIGH confidence (0.7+).
Require CLEAR EVIDENCE of potential collaboration, not just thematic similarity.

Types of synergies to look for:
1. COMPLEMENTARY_ACTIVITIES: Entities doing different parts of a CE value chain
   Example: Waste collector + Recycler + Manufacturer using recycled materials
   REQUIRES: Clear evidence both entities work in the same material flow

2. VALUE_CHAIN: Sequential relationships in circular economy processes
   Example: Producer → Repair service → Recycler → Raw material supplier
   REQUIRES: Evidence of actual material/product flow possibility

3. GEOGRAPHIC_CLUSTER: Entities in same area working on similar CE topics
   Example: Multiple startups in same district working on sustainable packaging
   REQUIRES: Both geographic proximity AND thematic alignment

4. SHARED_RESOURCES: Entities that could share infrastructure, knowledge, or networks
   Example: Research institute + University + Industry partner forming innovation hub
   REQUIRES: Evidence of complementary resources and mutual benefit

For EACH synergy you identify, provide:
- entity_names: List of 2+ entity names involved
- synergy_type: One of the 4 types above
- description: How these entities could work together and mutual benefits - IN ENGLISH
- confidence: 0.7-1.0 ONLY (do NOT include synergies below 0.7 confidence)
- potential_impact: 'high' (major CE impact), 'medium' (notable benefit)

CRITICAL RULES:
- MINIMUM confidence threshold is 0.7 - do NOT include lower confidence synergies
- Only identify synergies with CLEAR mutual benefit
- Focus on CIRCULAR ECONOMY synergies (not just any collaboration)
- Be specific about what each entity contributes
- Quality over quantity - fewer high-quality synergies is better than many weak ones
- It's OK to return an empty list if no strong synergies found

Return a list of synergy candidates (may be empty if no 0.7+ synergies found).
'''

# New prompt for cluster-based ecosystem analysis
CLUSTER_BASED_GAP_ANALYSIS_PROMPT = r'''
You are analyzing gaps in a Circular Economy ecosystem based on cluster summaries.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

ECOSYSTEM SUMMARY:
Total entities: {total_entities}

ROLE DISTRIBUTION:
{roles_distribution}

CAPABILITY CLUSTERS (what the ecosystem can DO):
{capability_clusters}

ACTIVITY CLUSTERS (what entities are DOING):
{activity_clusters}

NEED CLUSTERS (what entities NEED):
{need_clusters}

GEOGRAPHIC DISTRIBUTION:
{geographic_summary}

TASK: Analyze gaps and provide recommendations based on the FULL ecosystem view.

Analyze the following dimensions:

1. CAPABILITY-NEED GAPS: Where needs exist but no matching capabilities
   - Compare need clusters to capability clusters
   - Identify needs with no providers
   - Quantify the gap (e.g., "5 entities need X but only 1 provides it")

2. VALUE CHAIN GAPS: Missing stages in circular economy processes
   - Design → Production → Use → Collection → Recycling → Materials
   - Identify missing or weak stages
   - Note bottlenecks in material flows

3. ROLE DISTRIBUTION GAPS: Underrepresented ecosystem roles
   - Which of the 14 roles have <3 entities?
   - Which critical roles are missing entirely?
   - Imbalance between knowledge producers and implementers

4. GEOGRAPHIC GAPS: Uneven distribution of CE actors
   - Areas with high concentration
   - Areas with no CE presence
   - Accessibility issues

5. ACTIVITY COVERAGE GAPS: CE activities not represented
   - Compare to typical CE activity categories
   - Missing business models (e.g., no repair services, no sharing platforms)
   - Overrepresented vs underrepresented activities

For each gap identified:
- gap_type: One of the 5 types above
- title: Short descriptive title - IN ENGLISH
- description: Detailed explanation - IN ENGLISH
- severity: 'critical' (ecosystem can't function), 'significant' (major limitation), 'moderate' (notable gap)
- affected_entities: List of entities affected by this gap
- evidence: Specific data supporting this gap (from clusters/distribution)

6. RECOMMENDATIONS: Actionable steps to address gaps
For each recommendation:
- priority: 'high', 'medium', 'low'
- action: What should be done - IN ENGLISH
- target: Who should take action (entity type or specific entities)
- expected_impact: What improvement this would bring - IN ENGLISH
- related_gaps: Which gaps this addresses

Be specific and data-driven. Base all insights on the cluster summaries provided.
'''

ECOSYSTEM_GAP_ANALYSIS_PROMPT = r'''
You are analyzing gaps in a Circular Economy ecosystem.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CURRENT ECOSYSTEM:
Total entities: {total_entities}
Roles distribution:
{roles_distribution}

ENTITIES OVERVIEW:
{entities_summary}

TASK: Identify gaps, weaknesses, and missing elements in this ecosystem.

Analyze the following dimensions:

1. IDENTIFIED_GAPS: Missing or weak elements in the CE ecosystem
   - Missing value chain stages (e.g., no repair services, no remanufacturing)
   - Underrepresented CE sectors (e.g., textiles, construction, electronics)
   - Missing CE business models (e.g., no sharing platforms, no product-as-service)
   - Weak areas (e.g., only 1-2 entities in a critical CE function)

2. UNDERREPRESENTED_ROLES: Which of the 14 ecosystem roles are missing or too few?
   - Students, Researchers, Higher Education Institutions, Research Institutes
   - Non-Governmental Organizations, Industry Partners, Startups and Entrepreneurs
   - Public Authorities, Policy Makers, End-Users, Citizen Associations
   - Media and Communication Partners, Funding Bodies, Knowledge and Innovation Communities

3. MISSING_CONNECTIONS: Relationships that SHOULD exist but don't appear to
   - Research institutions not connected to industry
   - Startups isolated without mentors or partners
   - Public authorities disconnected from implementation actors
   - Funding bodies not visible to entities that need funding

4. GEOGRAPHIC_GAPS: Areas or districts lacking CE actors
   - Note if all entities cluster in one area vs. spread across city/region

5. RECOMMENDATIONS: Actionable steps to address gaps - ALL IN ENGLISH
   - Specific entity types to recruit (e.g., "Add 2-3 repair services")
   - Connection-building actions
   - Policy or infrastructure needs

Be honest and specific. It's better to identify real gaps than to say "everything is fine".
ALL output must be in English.
'''

ENTITY_DEDUPLICATION_PROMPT = r'''
You are analyzing whether different discovered entities refer to the same organization.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

DISCOVERED ENTITIES:
{entities_batch}

TASK: Identify which of these entities refer to the SAME organization or entity.

Consider:
1. Same or very similar names (e.g., "HCU" vs "HafenCity University Hamburg")
2. Same domain/URL (e.g., different subdomains of same website)
3. Abbreviations vs full names
4. Similar descriptions suggesting same organization
5. Subsidiaries or departments of same parent organization
6. Legal entity variations (GmbH, e.V., d.o.o., S.R.L., etc. are the same entity)

For each group of duplicate entities, provide:
- entity_urls: List of URLs that refer to the same organization
- canonical_name: The most complete/official name to use - IN ENGLISH if translatable
- confidence: 0.0-1.0 that these are truly duplicates

Return groups of duplicates. Entities not listed are considered unique.

Be CONSERVATIVE: Only group if you're confident they're the same entity (confidence > 0.7).
'''

ENTITY_MATCHING_PROMPT = r'''
You are matching mentioned partners/entities from a website to entities in our database.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

MENTIONED PARTNERS FROM WEBSITE:
{mentioned_partners}

DATABASE ENTITIES (names and URLs):
{database_entities}

TASK: For EACH mentioned partner, identify if it matches any entity in the database.

Consider:
1. Exact name matches
2. Abbreviations (e.g., "TUHH" = "Technical University of Hamburg")
3. Slight name variations
4. URL/domain matches
5. Contextual clues (e.g., description matches known entity)
6. Legal entity type variations (GmbH, e.V., d.o.o., S.R.L., Ltd, etc.)

For EACH mentioned partner, return:
- mentioned_name: The name as mentioned on the website
- matched_entity: The database entity name it matches (or null if no match)
- matched_url: The database entity URL it matches (or null)
- confidence: 0.0-1.0 (0.8+ for strong match, 0.6-0.8 for likely match, <0.6 for uncertain)
- reasoning: Brief explanation of why they match (or don't match) - IN ENGLISH

IMPORTANT:
- Only match if confident (confidence >= 0.6)
- Semantic/contextual matching is valuable (not just string matching)
- It's okay to return null for no match

Return a list of match results, one for each mentioned partner.
'''

CE_CAPABILITY_CLUSTERING_PROMPT = r'''
You are clustering circular economy capabilities to identify common categories.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CE CAPABILITIES FROM ALL ENTITIES:
{capabilities_list}

TASK: Group these CE capabilities into 10-15 meaningful clusters/categories.

Guidelines:
1. Focus on CIRCULAR ECONOMY themes (waste management, recycling, repair, reuse, circular design, etc.)
2. Create specific, actionable categories (not too broad)
3. Each category should have 2+ capabilities
4. Aim for balanced cluster sizes (avoid one huge cluster and many tiny ones)

Standard CE capability categories to consider:
- Recycling Infrastructure & Services
- Circular Design & Product Development
- Waste-to-Resource Technology
- Repair & Maintenance Services
- CE Consulting & Strategy
- Sustainable Materials & Sourcing
- Product Life Extension & Remanufacturing
- CE Education & Training
- Take-back Systems & Reverse Logistics
- Sharing Economy Platforms

For EACH cluster, provide:
- cluster_id: Short unique ID (e.g., "recycling_infrastructure")
- cluster_name: Human-readable name - IN ENGLISH
- description: What this cluster represents - IN ENGLISH
- items: List of capability names in this cluster
- entities: List of entity names providing these capabilities
- confidence: 0.0-1.0 (how confident you are in this clustering)

Return 10-15 capability clusters.
'''

CE_NEED_CLUSTERING_PROMPT = r'''
You are clustering circular economy needs to identify common patterns.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CE NEEDS FROM ALL ENTITIES:
{needs_list}

TASK: Group these CE needs into 10-15 meaningful clusters/categories.

Guidelines:
1. Focus on CIRCULAR ECONOMY needs (sustainable materials, recycling partners, CE expertise, etc.)
2. Create specific, actionable categories
3. Each category should have 2+ needs
4. Identify systemic gaps (many entities needing same thing)

Standard CE need categories to consider:
- Sustainable Material Sourcing
- Recycling & Waste Management Partnerships
- CE Funding & Investment
- Circular Design Expertise
- Product Take-back Infrastructure
- CE Technology & Software
- Policy Support & Advocacy
- Customer Education & Engagement
- Reverse Logistics Solutions
- Life Cycle Assessment Services

For EACH cluster, provide:
- cluster_id: Short unique ID (e.g., "sustainable_materials")
- cluster_name: Human-readable name - IN ENGLISH
- description: What entities in this cluster need - IN ENGLISH
- items: List of need names in this cluster
- entities: List of entity names with these needs
- confidence: 0.0-1.0 (clustering confidence)

Return 10-15 need clusters. Identify high-demand areas (many entities needing same thing).
'''

CE_ACTIVITY_CLUSTERING_PROMPT = r'''
You are clustering circular economy activities to identify activity types.

''' + ENGLISH_OUTPUT_INSTRUCTION + r'''

CE ACTIVITIES FROM ALL ENTITIES:
{activities_list}

TASK: Group these CE activities into 8-12 meaningful clusters/categories.

Guidelines:
1. Focus on types of CIRCULAR ECONOMY actions/projects/initiatives
2. Distinguish between different stages of circular economy (design, production, use, recovery)
3. Each category should have 2+ activities
4. Create clusters that help understand the ecosystem's CE landscape

Standard CE activity categories (based on taxonomy):
- Design & Production (eco-design, design for circularity)
- Use & Consumption (repair, sharing, rental)
- Collection & Logistics (waste collection, take-back)
- Recycling & Processing (mechanical, chemical recycling)
- Resource Recovery (energy recovery, composting)
- Industrial Symbiosis (byproduct exchange)
- Digital & Technology (tracking, IoT, AI)
- Policy & Governance (standards, certification)
- Education & Research (training, R&D)
- Finance & Business Models (circular business models)

For EACH cluster, provide:
- cluster_id: Short unique ID (e.g., "waste_collection")
- cluster_name: Human-readable name - IN ENGLISH
- description: What activities are in this cluster - IN ENGLISH
- items: List of activity names in this cluster
- entities: List of entity names performing these activities
- confidence: 0.0-1.0 (clustering confidence)

Return 8-12 activity clusters.
'''

__all__ = [
    "KNOWLEDGE_TRANSFER_ANALYSIS_PROMPT",
    "SYNERGY_DETECTION_PROMPT",
    "ECOSYSTEM_GAP_ANALYSIS_PROMPT",
    "CLUSTER_BASED_GAP_ANALYSIS_PROMPT",
    "ENTITY_DEDUPLICATION_PROMPT",
    "ENTITY_MATCHING_PROMPT",
    "CE_CAPABILITY_CLUSTERING_PROMPT",
    "CE_NEED_CLUSTERING_PROMPT",
    "CE_ACTIVITY_CLUSTERING_PROMPT",
    "ENGLISH_OUTPUT_INSTRUCTION"
]
