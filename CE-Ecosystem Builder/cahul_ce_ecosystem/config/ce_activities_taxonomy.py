"""
Circular Economy Activities Taxonomy

A standardized taxonomy of 120 CE activities based on:
- Ellen MacArthur Foundation CE framework
- EU Circular Economy Action Plan (2020)
- ReSOLVE framework (Regenerate, Share, Optimise, Loop, Virtualise, Exchange)
- Academic literature on circular economy

This taxonomy ensures consistent, English-only activity classification across all ecosystems.
"""

from typing import Dict, List, Set

# =============================================================================
# CE ACTIVITIES TAXONOMY
# Organized into 10 categories with ~12 activities each = ~120 total
# =============================================================================

CE_ACTIVITIES_TAXONOMY: Dict[str, List[str]] = {

    # =========================================================================
    # 1. DESIGN & PRODUCTION (ReSOLVE: Regenerate, Optimise)
    # =========================================================================
    "Design & Production": [
        "Eco-design and design for circularity",
        "Design for disassembly and modularity",
        "Design for longevity and durability",
        "Design for repair and maintenance",
        "Design for recyclability",
        "Sustainable materials selection",
        "Biomimicry and bio-based design",
        "Cradle-to-cradle product design",
        "Lightweighting and material optimization",
        "Additive manufacturing and 3D printing",
        "Remanufacturing process development",
        "Circular product certification",
    ],

    # =========================================================================
    # 2. USE & CONSUMPTION (ReSOLVE: Share, Virtualise)
    # =========================================================================
    "Use & Consumption": [
        "Product-as-a-Service models",
        "Sharing platforms and services",
        "Rental and leasing services",
        "Repair services and workshops",
        "Maintenance and servicing",
        "Product life extension services",
        "Second-hand and resale platforms",
        "Refurbishment services",
        "Upcycling and creative reuse",
        "Collaborative consumption platforms",
        "Tool and equipment libraries",
        "Subscription-based product access",
    ],

    # =========================================================================
    # 3. COLLECTION & LOGISTICS (ReSOLVE: Loop)
    # =========================================================================
    "Collection & Logistics": [
        "Waste collection and sorting",
        "Separate collection systems",
        "Take-back schemes and programs",
        "Reverse logistics operations",
        "Collection point management",
        "Door-to-door collection services",
        "Commercial waste collection",
        "Hazardous waste collection",
        "E-waste collection programs",
        "Textile collection services",
        "Packaging return systems",
        "Deposit-return schemes",
    ],

    # =========================================================================
    # 4. RECYCLING & PROCESSING (ReSOLVE: Loop)
    # =========================================================================
    "Recycling & Processing": [
        "Mechanical recycling",
        "Chemical recycling",
        "Plastic recycling and processing",
        "Metal recycling and recovery",
        "Paper and cardboard recycling",
        "Glass recycling",
        "Textile recycling",
        "E-waste recycling and processing",
        "Construction waste recycling",
        "Battery recycling",
        "Composite material recycling",
        "Advanced sorting technologies",
    ],

    # =========================================================================
    # 5. RESOURCE RECOVERY (ReSOLVE: Regenerate, Loop)
    # =========================================================================
    "Resource Recovery": [
        "Material recovery and extraction",
        "Energy recovery from waste",
        "Biogas production",
        "Composting and organic processing",
        "Anaerobic digestion",
        "Precious metal recovery",
        "Rare earth element recovery",
        "Water recovery and reuse",
        "Heat recovery systems",
        "Nutrient recovery",
        "Solvent recovery and recycling",
        "Industrial byproduct recovery",
    ],

    # =========================================================================
    # 6. INDUSTRIAL SYMBIOSIS (ReSOLVE: Exchange, Optimise)
    # =========================================================================
    "Industrial Symbiosis": [
        "Industrial symbiosis networks",
        "Waste-to-resource exchanges",
        "Byproduct synergy programs",
        "Eco-industrial park development",
        "Cross-sector material flows",
        "Energy sharing between industries",
        "Water sharing and cascading",
        "Shared infrastructure development",
        "Industrial ecosystem mapping",
        "Symbiosis matchmaking platforms",
        "Circular supply chain development",
        "Regional material flow optimization",
    ],

    # =========================================================================
    # 7. DIGITAL & TECHNOLOGY (ReSOLVE: Virtualise, Optimise)
    # =========================================================================
    "Digital & Technology": [
        "Digital product passports",
        "Material tracking and traceability",
        "IoT for resource monitoring",
        "AI-powered waste sorting",
        "Blockchain for supply chain transparency",
        "Digital twins for product lifecycle",
        "Predictive maintenance systems",
        "Online marketplaces for secondary materials",
        "Circular economy data platforms",
        "Smart waste management systems",
        "Resource efficiency software",
        "Life cycle assessment tools",
    ],

    # =========================================================================
    # 8. POLICY & GOVERNANCE
    # =========================================================================
    "Policy & Governance": [
        "Circular economy policy development",
        "Extended producer responsibility",
        "Waste management regulation",
        "Green public procurement",
        "Circular economy standards development",
        "Environmental certification programs",
        "Regulatory compliance services",
        "Policy advocacy and lobbying",
        "Municipal circular economy programs",
        "Regional circular economy strategies",
        "International CE cooperation",
        "Circular economy impact assessment",
    ],

    # =========================================================================
    # 9. EDUCATION & RESEARCH
    # =========================================================================
    "Education & Research": [
        "Circular economy research",
        "Sustainability education programs",
        "Professional training and upskilling",
        "Circular design education",
        "Waste management training",
        "Consumer awareness campaigns",
        "Academic CE programs",
        "Innovation labs and incubators",
        "Circular economy consultancy",
        "Knowledge transfer programs",
        "Best practice documentation",
        "Circular economy publications",
    ],

    # =========================================================================
    # 10. FINANCE & BUSINESS MODELS
    # =========================================================================
    "Finance & Business Models": [
        "Circular economy investment",
        "Green financing and bonds",
        "Impact investing for circularity",
        "Circular business model development",
        "Performance-based contracts",
        "Leasing and rental business models",
        "Pay-per-use business models",
        "Circular startup funding",
        "ESG reporting and metrics",
        "Circular economy valuation",
        "Risk assessment for circular projects",
        "Circular economy venture capital",
    ],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_activities() -> List[str]:
    """Get a flat list of all CE activities."""
    activities = []
    for category_activities in CE_ACTIVITIES_TAXONOMY.values():
        activities.extend(category_activities)
    return activities


def get_activity_categories() -> List[str]:
    """Get list of all activity categories."""
    return list(CE_ACTIVITIES_TAXONOMY.keys())


def get_activities_by_category(category: str) -> List[str]:
    """Get activities for a specific category."""
    return CE_ACTIVITIES_TAXONOMY.get(category, [])


def find_activity_category(activity: str) -> str:
    """Find which category an activity belongs to."""
    activity_lower = activity.lower()
    for category, activities in CE_ACTIVITIES_TAXONOMY.items():
        for cat_activity in activities:
            if cat_activity.lower() == activity_lower:
                return category
    return "Unknown"


def get_activity_count() -> int:
    """Get total number of predefined activities."""
    return sum(len(activities) for activities in CE_ACTIVITIES_TAXONOMY.values())


# =============================================================================
# ACTIVITY MATCHING (for extraction)
# =============================================================================

# Keywords mapped to taxonomy activities for extraction
# These help match free-text descriptions to predefined activities
ACTIVITY_KEYWORDS: Dict[str, str] = {
    # Design & Production
    "eco-design": "Eco-design and design for circularity",
    "ecodesign": "Eco-design and design for circularity",
    "circular design": "Eco-design and design for circularity",
    "design for disassembly": "Design for disassembly and modularity",
    "modular design": "Design for disassembly and modularity",
    "durable": "Design for longevity and durability",
    "durability": "Design for longevity and durability",
    "longevity": "Design for longevity and durability",
    "repairable": "Design for repair and maintenance",
    "recyclable": "Design for recyclability",
    "sustainable materials": "Sustainable materials selection",
    "bio-based": "Biomimicry and bio-based design",
    "biobased": "Biomimicry and bio-based design",
    "cradle to cradle": "Cradle-to-cradle product design",
    "c2c": "Cradle-to-cradle product design",
    "lightweighting": "Lightweighting and material optimization",
    "3d printing": "Additive manufacturing and 3D printing",
    "additive manufacturing": "Additive manufacturing and 3D printing",
    "remanufacturing": "Remanufacturing process development",

    # Use & Consumption
    "product as a service": "Product-as-a-Service models",
    "paas": "Product-as-a-Service models",
    "servitization": "Product-as-a-Service models",
    "sharing platform": "Sharing platforms and services",
    "sharing economy": "Sharing platforms and services",
    "rental": "Rental and leasing services",
    "leasing": "Rental and leasing services",
    "repair": "Repair services and workshops",
    "repair cafe": "Repair services and workshops",
    "maintenance": "Maintenance and servicing",
    "refurbishment": "Refurbishment services",
    "refurbish": "Refurbishment services",
    "second-hand": "Second-hand and resale platforms",
    "secondhand": "Second-hand and resale platforms",
    "resale": "Second-hand and resale platforms",
    "upcycling": "Upcycling and creative reuse",
    "upcycle": "Upcycling and creative reuse",
    "tool library": "Tool and equipment libraries",
    "subscription": "Subscription-based product access",

    # Collection & Logistics
    "waste collection": "Waste collection and sorting",
    "sorting": "Waste collection and sorting",
    "separate collection": "Separate collection systems",
    "take-back": "Take-back schemes and programs",
    "takeback": "Take-back schemes and programs",
    "reverse logistics": "Reverse logistics operations",
    "collection point": "Collection point management",
    "e-waste collection": "E-waste collection programs",
    "ewaste collection": "E-waste collection programs",
    "textile collection": "Textile collection services",
    "deposit return": "Deposit-return schemes",
    "deposit-return": "Deposit-return schemes",

    # Recycling & Processing
    "mechanical recycling": "Mechanical recycling",
    "chemical recycling": "Chemical recycling",
    "plastic recycling": "Plastic recycling and processing",
    "metal recycling": "Metal recycling and recovery",
    "paper recycling": "Paper and cardboard recycling",
    "cardboard recycling": "Paper and cardboard recycling",
    "glass recycling": "Glass recycling",
    "textile recycling": "Textile recycling",
    "e-waste recycling": "E-waste recycling and processing",
    "electronics recycling": "E-waste recycling and processing",
    "construction waste": "Construction waste recycling",
    "demolition waste": "Construction waste recycling",
    "battery recycling": "Battery recycling",

    # Resource Recovery
    "material recovery": "Material recovery and extraction",
    "energy recovery": "Energy recovery from waste",
    "waste to energy": "Energy recovery from waste",
    "biogas": "Biogas production",
    "composting": "Composting and organic processing",
    "compost": "Composting and organic processing",
    "anaerobic digestion": "Anaerobic digestion",
    "precious metal": "Precious metal recovery",
    "rare earth": "Rare earth element recovery",
    "water reuse": "Water recovery and reuse",
    "water recycling": "Water recovery and reuse",
    "heat recovery": "Heat recovery systems",
    "nutrient recovery": "Nutrient recovery",

    # Industrial Symbiosis
    "industrial symbiosis": "Industrial symbiosis networks",
    "eco-industrial": "Eco-industrial park development",
    "eco industrial": "Eco-industrial park development",
    "byproduct": "Byproduct synergy programs",
    "by-product": "Byproduct synergy programs",
    "material exchange": "Waste-to-resource exchanges",
    "waste exchange": "Waste-to-resource exchanges",
    "circular supply chain": "Circular supply chain development",

    # Digital & Technology
    "digital passport": "Digital product passports",
    "product passport": "Digital product passports",
    "traceability": "Material tracking and traceability",
    "tracking": "Material tracking and traceability",
    "iot": "IoT for resource monitoring",
    "smart waste": "Smart waste management systems",
    "ai sorting": "AI-powered waste sorting",
    "blockchain": "Blockchain for supply chain transparency",
    "digital twin": "Digital twins for product lifecycle",
    "predictive maintenance": "Predictive maintenance systems",
    "secondary materials marketplace": "Online marketplaces for secondary materials",
    "lca": "Life cycle assessment tools",
    "life cycle assessment": "Life cycle assessment tools",

    # Policy & Governance
    "circular economy policy": "Circular economy policy development",
    "ce policy": "Circular economy policy development",
    "epr": "Extended producer responsibility",
    "extended producer responsibility": "Extended producer responsibility",
    "green procurement": "Green public procurement",
    "public procurement": "Green public procurement",
    "certification": "Environmental certification programs",
    "compliance": "Regulatory compliance services",
    "advocacy": "Policy advocacy and lobbying",

    # Education & Research
    "research": "Circular economy research",
    "sustainability education": "Sustainability education programs",
    "training": "Professional training and upskilling",
    "awareness": "Consumer awareness campaigns",
    "incubator": "Innovation labs and incubators",
    "accelerator": "Innovation labs and incubators",
    "consultancy": "Circular economy consultancy",
    "consulting": "Circular economy consultancy",

    # Finance & Business Models
    "investment": "Circular economy investment",
    "green finance": "Green financing and bonds",
    "green bond": "Green financing and bonds",
    "impact investing": "Impact investing for circularity",
    "circular business model": "Circular business model development",
    "performance contract": "Performance-based contracts",
    "pay per use": "Pay-per-use business models",
    "pay-per-use": "Pay-per-use business models",
    "esg": "ESG reporting and metrics",
    "venture capital": "Circular economy venture capital",
}


def match_activity_to_taxonomy(text: str) -> List[str]:
    """
    Match free-text description to predefined taxonomy activities.
    Returns list of matched activities (may be empty).
    """
    text_lower = text.lower()
    matched = set()

    # First, check exact keyword matches
    for keyword, activity in ACTIVITY_KEYWORDS.items():
        if keyword in text_lower:
            matched.add(activity)

    # Then, check if any taxonomy activity is mentioned directly
    for category_activities in CE_ACTIVITIES_TAXONOMY.values():
        for activity in category_activities:
            if activity.lower() in text_lower:
                matched.add(activity)

    return list(matched)


def get_taxonomy_for_prompt() -> str:
    """
    Generate a formatted string of the taxonomy for use in LLM prompts.
    """
    lines = ["Available CE Activities (select ONLY from this list):"]
    for category, activities in CE_ACTIVITIES_TAXONOMY.items():
        lines.append(f"\n{category}:")
        for activity in activities:
            lines.append(f"  - {activity}")
    return "\n".join(lines)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_activity(activity: str) -> bool:
    """Check if an activity is in the taxonomy."""
    all_activities = get_all_activities()
    return activity in all_activities


def validate_activities(activities: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate a list of activities against the taxonomy.
    Returns (valid_activities, invalid_activities).
    """
    all_activities = set(get_all_activities())
    valid = []
    invalid = []

    for activity in activities:
        if activity in all_activities:
            valid.append(activity)
        else:
            invalid.append(activity)

    return valid, invalid


# =============================================================================
# STATISTICS
# =============================================================================

if __name__ == "__main__":
    print(f"CE Activities Taxonomy Statistics:")
    print(f"{'='*50}")
    print(f"Total categories: {len(CE_ACTIVITIES_TAXONOMY)}")
    print(f"Total activities: {get_activity_count()}")
    print(f"{'='*50}")
    for category in get_activity_categories():
        count = len(get_activities_by_category(category))
        print(f"  {category}: {count} activities")
