#!/usr/bin/env python3
"""
Google Places API Extractor for Circular Economy Actors
Extracts CE-related entities for Novi Sad (Serbia) and Cahul (Moldova)
"""

import os
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Set
from pathlib import Path
import pandas as pd
from tqdm import tqdm

try:
    import googlemaps
except ImportError:
    print("ERROR: googlemaps library not installed. Installing...")
    os.system("pip install -U googlemaps")
    import googlemaps


# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Reduced CE keywords for budget-conscious extraction
# These 5 keywords cover the most important concepts while staying within budget
CE_KEYWORDS_REDUCED = {
    "novi_sad": [
        "reciklaža",  # Serbian: recycling
        "održivost",  # Serbian: sustainability
        "circular economy",  # English
        "recycling",  # English
        "zero waste"  # English
    ],
    "cahul": [
        "reciclare",  # Romanian: recycling
        "sustenabilitate",  # Romanian: sustainability
        "circular economy",  # English
        "recycling",  # English
        "zero waste"  # English
    ]
}

# City configurations
CITIES = {
    "novi_sad": {
        "name": "Novi Sad",
        "location": {"lat": 45.2671, "lng": 19.8335},
        "country": "Serbia",
        "radius": 20000,  # 20km - city proper + immediate suburbs
        "output_dir": "Input/Novi Sad",
        "languages": ["sr", "en"],
        "ce_keywords": [
            # Serbian (Latin)
            "cirkularna ekonomija",
            "reciklaža",
            "održivost",
            "reciklažni centar",
            "upravljanje otpadom",
            "popravka",
            "održiva proizvodnja",
            # Serbian (Cyrillic)
            "циркуларна економија",
            "рециклажа",
            "одржи вост",
            # English
            "circular economy",
            "recycling",
            "sustainability",
            "zero waste"
        ]
    },
    "cahul": {
        "name": "Cahul",
        "location": {"lat": 45.9074, "lng": 28.1947},
        "country": "Moldova",
        "radius": 12000,  # 12km - town center + immediate surroundings
        "output_dir": "Input/Cahul",
        "languages": ["ro", "ru", "en"],
        "ce_keywords": [
            # Romanian
            "economie circulară",
            "reciclare",
            "sustenabilitate",
            "colectare deșeuri",
            "gestionare deșeuri",
            "reparare",
            "producție durabilă",
            # Russian
            "циркулярная экономика",
            "переработка",
            "устойчивость",
            # English
            "circular economy",
            "recycling",
            "sustainability",
            "zero waste"
        ]
    }
}

# 14 Ecosystem Roles with Google Places search terms
ECOSYSTEM_ROLES = {
    "Students": [
        "student organization",
        "student initiative",
        "student association",
        "student group"
    ],
    "Researchers": [
        "research group",
        "researcher",
        "research team",
        "research laboratory"
    ],
    "Higher Education Institutions": [
        "university",
        "college",
        "higher education",
        "institute of technology"
    ],
    "Research Institutes": [
        "research institute",
        "research center",
        "research facility",
        "scientific institute"
    ],
    "Non-Governmental Organizations": [
        "NGO",
        "non-profit",
        "foundation",
        "association",
        "non-governmental organization"
    ],
    "Industry Partners": [
        "company",
        "manufacturer",
        "consultancy",
        "service provider",
        "enterprise"
    ],
    "Startups and Entrepreneurs": [
        "startup",
        "entrepreneur",
        "incubator",
        "accelerator",
        "new venture"
    ],
    "Public Authorities": [
        "government office",
        "municipal service",
        "public authority",
        "city administration"
    ],
    "Policy Makers": [
        "policy institute",
        "think tank",
        "policy organization"
    ],
    "End-Users": [
        "consumer group",
        "user community",
        "customer organization"
    ],
    "Citizen Associations": [
        "community group",
        "neighborhood association",
        "citizen initiative",
        "local association"
    ],
    "Media and Communication Partners": [
        "news outlet",
        "media company",
        "press agency",
        "publisher",
        "communication agency"
    ],
    "Funding Bodies": [
        "grant organization",
        "investment fund",
        "venture capital",
        "funding agency"
    ],
    "Knowledge and Innovation Communities": [
        "innovation hub",
        "innovation center",
        "business network",
        "cluster organization",
        "innovation community"
    ]
}

# Expanded search terms including Serbian/local language terms for broader coverage
EXPANDED_ROLE_TERMS = {
    "Students": [
        "student organization", "student initiative", "student association", "student group",
        # Serbian terms
        "studentska organizacija", "studentski centar", "studentski dom",
        "omladinska organizacija", "omladinski centar"
    ],
    "Researchers": [
        "research group", "researcher", "research team", "research laboratory",
        # Serbian terms
        "istraživač", "istraživački tim", "laboratorija", "naučni rad"
    ],
    "Higher Education Institutions": [
        "university", "college", "higher education", "institute of technology",
        # Serbian terms
        "univerzitet", "fakultet", "visoka škola", "akademija", "institut",
        "obrazovanje", "visoko obrazovanje"
    ],
    "Research Institutes": [
        "research institute", "research center", "research facility", "scientific institute",
        # Serbian terms
        "istraživački institut", "istraživački centar", "naučni institut",
        "zavod", "centar za istraživanje"
    ],
    "Non-Governmental Organizations": [
        "NGO", "non-profit", "foundation", "association", "non-governmental organization",
        # Serbian terms
        "udruženje", "fondacija", "organizacija", "nevladina organizacija",
        "civilno društvo", "humanitarna organizacija", "dobrotvorna organizacija"
    ],
    "Industry Partners": [
        "company", "manufacturer", "consultancy", "service provider", "enterprise",
        # Serbian terms
        "fabrika", "proizvodnja", "preduzeće", "firma", "kompanija",
        "industrija", "d.o.o.", "a.d.", "proizvođač"
    ],
    "Startups and Entrepreneurs": [
        "startup", "entrepreneur", "incubator", "accelerator", "new venture",
        # Serbian terms
        "inkubator", "akcelerator", "biznis centar", "poslovni centar",
        "coworking", "tech hub", "IT kompanija", "preduzetnik"
    ],
    "Public Authorities": [
        "government office", "municipal service", "public authority", "city administration",
        # Serbian terms
        "gradska uprava", "opština", "javna ustanova", "ministarstvo",
        "sekretarijat", "pokrajinska vlada", "lokalna samouprava"
    ],
    "Policy Makers": [
        "policy institute", "think tank", "policy organization",
        # Serbian terms
        "institut za javne politike", "centar za politike", "analitički centar"
    ],
    "End-Users": [
        "consumer group", "user community", "customer organization",
        # Serbian terms
        "udruženje potrošača", "zajednica korisnika", "potrošački centar"
    ],
    "Citizen Associations": [
        "community group", "neighborhood association", "citizen initiative", "local association",
        # Serbian terms
        "mesna zajednica", "građanska inicijativa", "lokalno udruženje",
        "kulturni centar", "dom kulture", "društvo"
    ],
    "Media and Communication Partners": [
        "news outlet", "media company", "press agency", "publisher", "communication agency",
        # Serbian terms
        "medijska kuća", "novinska agencija", "izdavač", "radio", "televizija",
        "marketing agencija", "PR agencija"
    ],
    "Funding Bodies": [
        "grant organization", "investment fund", "venture capital", "funding agency",
        # Serbian terms
        "investicioni fond", "razvojni fond", "banka", "finansijska institucija",
        "fond za razvoj"
    ],
    "Knowledge and Innovation Communities": [
        "innovation hub", "innovation center", "business network", "cluster organization",
        "innovation community",
        # Serbian terms
        "inovacioni centar", "naučno-tehnološki park", "klaster",
        "poslovna mreža", "razvojni centar", "centar za inovacije"
    ]
}

# Google Places built-in types mapped to ecosystem roles
ROLE_TO_PLACE_TYPES = {
    "Higher Education Institutions": ["university", "school", "secondary_school", "primary_school"],
    "Research Institutes": [],  # No direct mapping
    "Public Authorities": ["local_government_office", "city_hall", "courthouse"],
    "Industry Partners": ["store", "shopping_mall", "electronics_store", "hardware_store"],
    "Funding Bodies": ["bank", "finance", "accounting"],
    "Media and Communication Partners": ["book_store"],
    "Non-Governmental Organizations": [],  # No direct mapping
    "Knowledge and Innovation Communities": ["library"],
}


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Simple file-based cache for API results"""

    def __init__(self, cache_dir: str = ".cache/google_places"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "query_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Optional[List[Dict]]:
        return self.cache.get(key)

    def set(self, key: str, value: List[Dict]):
        self.cache[key] = value
        self._save_cache()


# ============================================================================
# QUOTA MANAGER
# ============================================================================

class QuotaManager:
    """Manages API quota limits to stay within free tier"""

    # Free tier limits (conservative buffer)
    MAX_TEXT_SEARCHES = 9900
    MAX_PLACE_DETAILS = 4900

    def __init__(self, quota_file: str = ".cache/google_places/quota.json"):
        self.quota_file = Path(quota_file)
        self.quota_file.parent.mkdir(parents=True, exist_ok=True)
        self.quotas = self._load_quotas()

    def _load_quotas(self) -> Dict:
        if self.quota_file.exists():
            with open(self.quota_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'text_searches': 0,
            'place_details': 0,
            'last_reset': datetime.now().isoformat()
        }

    def _save_quotas(self):
        with open(self.quota_file, 'w', encoding='utf-8') as f:
            json.dump(self.quotas, f, ensure_ascii=False, indent=2)

    def can_make_search(self) -> bool:
        """Check if we can make a text search request"""
        return self.quotas['text_searches'] < self.MAX_TEXT_SEARCHES

    def can_make_details(self) -> bool:
        """Check if we can make a place details request"""
        return self.quotas['place_details'] < self.MAX_PLACE_DETAILS

    def increment_searches(self):
        """Increment text search counter"""
        self.quotas['text_searches'] += 1
        self._save_quotas()
        self._check_warnings('text_searches', self.MAX_TEXT_SEARCHES)

    def increment_details(self):
        """Increment place details counter"""
        self.quotas['place_details'] += 1
        self._save_quotas()
        self._check_warnings('place_details', self.MAX_PLACE_DETAILS)

    def _check_warnings(self, quota_type: str, limit: int):
        """Print warnings at 80% and 90% usage"""
        usage = self.quotas[quota_type]
        percentage = (usage / limit) * 100

        if usage == int(limit * 0.8):
            print(f"\n⚠️  WARNING: {quota_type} at 80% ({usage}/{limit})")
        elif usage == int(limit * 0.9):
            print(f"\n⚠️  WARNING: {quota_type} at 90% ({usage}/{limit})")

    def get_status(self) -> Dict:
        """Get current quota usage"""
        return {
            'text_searches': {
                'used': self.quotas['text_searches'],
                'limit': self.MAX_TEXT_SEARCHES,
                'remaining': self.MAX_TEXT_SEARCHES - self.quotas['text_searches'],
                'percentage': (self.quotas['text_searches'] / self.MAX_TEXT_SEARCHES) * 100
            },
            'place_details': {
                'used': self.quotas['place_details'],
                'limit': self.MAX_PLACE_DETAILS,
                'remaining': self.MAX_PLACE_DETAILS - self.quotas['place_details'],
                'percentage': (self.quotas['place_details'] / self.MAX_PLACE_DETAILS) * 100
            }
        }

    def reset_quotas(self):
        """Reset quotas (e.g., for new billing cycle)"""
        self.quotas = {
            'text_searches': 0,
            'place_details': 0,
            'last_reset': datetime.now().isoformat()
        }
        self._save_quotas()
        print("✓ Quotas have been reset")


# ============================================================================
# GOOGLE PLACES API CLIENT
# ============================================================================

class GooglePlacesClient:
    """Wrapper for Google Places API with rate limiting and caching"""

    def __init__(self, api_key: str, requests_per_second: int = 10):
        self.client = googlemaps.Client(key=api_key)
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.request_count = 0
        self.cache = CacheManager()
        self.quota = QuotaManager()

    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1

    def search_places(self, query: str, location: Dict[str, float], radius: int,
                       max_pages: int = 3) -> List[Dict]:
        """Search for places using text search with pagination and exponential backoff retry.

        Args:
            query: Search query string
            location: Dict with 'lat' and 'lng' keys
            radius: Search radius in meters
            max_pages: Maximum number of pages to fetch (up to 3, each page has ~20 results)

        Returns:
            List of place dictionaries (up to 60 results with pagination)
        """
        cache_key = f"search:{query}:{location['lat']},{location['lng']}:{radius}:p{max_pages}"

        # Check cache first (doesn't count against quota)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        all_places = []
        next_page_token = None
        max_retries = 5
        base_delay = 2  # seconds

        for page in range(max_pages):
            # Check quota before making API call
            if not self.quota.can_make_search():
                print(f"\n🛑 QUOTA LIMIT REACHED: Cannot perform more text searches (limit: {self.quota.MAX_TEXT_SEARCHES})")
                print(f"   Stopping pagination for query: '{query}'")
                break

            for attempt in range(max_retries):
                self._rate_limit()
                try:
                    if next_page_token:
                        # Google requires a short delay before using page token
                        time.sleep(2)
                        results = self.client.places(
                            query=query,
                            location=location,
                            radius=radius,
                            page_token=next_page_token
                        )
                    else:
                        results = self.client.places(
                            query=query,
                            location=location,
                            radius=radius
                        )

                    # Increment quota counter on successful request
                    self.quota.increment_searches()

                    places = results.get('results', [])
                    all_places.extend(places)

                    # Check for next page token
                    next_page_token = results.get('next_page_token')
                    if not next_page_token:
                        # No more pages available
                        break

                    break  # Success, exit retry loop

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check if it's a rate limit error
                    if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"  Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            continue

                    print(f"  ERROR searching '{query}' (page {page + 1}): {e}")
                    next_page_token = None  # Stop pagination on error
                    break

            # Exit pagination loop if no more pages
            if not next_page_token:
                break

        # Cache all results
        if all_places:
            self.cache.set(cache_key, all_places)

        return all_places

    def get_place_details(self, place_id: str) -> Optional[Dict]:
        """Get detailed information about a place with exponential backoff retry"""
        cache_key = f"details:{place_id}"

        # Check cache first (doesn't count against quota)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Check quota before making API call
        if not self.quota.can_make_details():
            print(f"\n🛑 QUOTA LIMIT REACHED: Cannot perform more place details requests (limit: {self.quota.MAX_PLACE_DETAILS})")
            print(f"   Skipping place_id: {place_id}")
            return None

        # Retry with exponential backoff
        max_retries = 5
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                result = self.client.place(
                    place_id=place_id,
                    fields=['name', 'formatted_address', 'geometry', 'formatted_phone_number',
                           'website', 'type', 'business_status', 'rating', 'user_ratings_total']
                )

                # Increment quota counter on successful request
                self.quota.increment_details()

                details = result.get('result', {})
                self.cache.set(cache_key, details)
                return details

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"  Rate limit hit, waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(delay)
                        continue

                # For non-rate-limit errors, just log and return None (don't retry)
                if attempt == 0:  # Only print error once
                    print(f"  ERROR fetching details for {place_id}: {e}")
                return None

        print(f"  Max retries exceeded for place_id: {place_id}")
        return None


# ============================================================================
# QUERY BUILDER
# ============================================================================

class QueryBuilder:
    """Builds search queries combining roles and CE keywords"""

    def __init__(self, city_config: Dict, use_expanded_terms: bool = True,
                 use_reduced_ce_keywords: bool = False, city_key: str = None):
        self.city = city_config['name']
        self.city_key = city_key

        # Use reduced CE keywords if requested (budget-conscious mode)
        if use_reduced_ce_keywords and city_key and city_key in CE_KEYWORDS_REDUCED:
            self.ce_keywords = CE_KEYWORDS_REDUCED[city_key]
        else:
            self.ce_keywords = city_config['ce_keywords']

        self.use_expanded_terms = use_expanded_terms
        # Use expanded terms if available and requested
        if use_expanded_terms and EXPANDED_ROLE_TERMS:
            self.roles = EXPANDED_ROLE_TERMS
        else:
            self.roles = ECOSYSTEM_ROLES

    def generate_queries(self, include_broad: bool = True,
                          include_ce_specific: bool = True) -> List[Dict[str, str]]:
        """Generate search queries with configurable strategies.

        Args:
            include_broad: Include role-only queries without CE keywords
            include_ce_specific: Include CE keyword + role combinations

        Returns:
            List of query dictionaries
        """
        queries = []

        # Phase A: Broad role-based searches (without CE keywords)
        # These capture all entities of a type, not just CE-related
        if include_broad:
            for role_name, role_terms in self.roles.items():
                for role_term in role_terms:
                    query = f"{role_term} {self.city}"
                    queries.append({
                        'query': query,
                        'ecosystem_role': role_name,
                        'role_term': role_term,
                        'ce_keyword': '',  # No CE keyword
                        'query_type': 'broad'
                    })

        # Phase B: CE-specific searches (CE keyword + role + city)
        # More targeted, finds CE-focused entities
        if include_ce_specific:
            for role_name, role_terms in self.roles.items():
                for role_term in role_terms:
                    for ce_keyword in self.ce_keywords:
                        query = f"{ce_keyword} {role_term} {self.city}"
                        queries.append({
                            'query': query,
                            'ecosystem_role': role_name,
                            'role_term': role_term,
                            'ce_keyword': ce_keyword,
                            'query_type': 'ce_specific'
                        })

        return queries

    def generate_type_based_queries(self) -> List[Dict[str, str]]:
        """Generate queries using Google Places built-in types.

        These queries search by place type within the location radius,
        which can find entities that don't match keyword searches.
        """
        queries = []

        for role_name, place_types in ROLE_TO_PLACE_TYPES.items():
            for place_type in place_types:
                # Simple type-based query
                query = f"{place_type} {self.city}"
                queries.append({
                    'query': query,
                    'ecosystem_role': role_name,
                    'role_term': place_type,
                    'ce_keyword': '',
                    'query_type': 'type_based'
                })

        return queries

    def get_query_count_estimate(self, include_broad: bool = True,
                                   include_ce_specific: bool = True,
                                   include_type_based: bool = True) -> Dict[str, int]:
        """Estimate the number of queries that will be generated."""
        counts = {
            'broad': 0,
            'ce_specific': 0,
            'type_based': 0,
            'total': 0
        }

        if include_broad:
            counts['broad'] = sum(len(terms) for terms in self.roles.values())

        if include_ce_specific:
            counts['ce_specific'] = sum(len(terms) for terms in self.roles.values()) * len(self.ce_keywords)

        if include_type_based:
            counts['type_based'] = sum(len(types) for types in ROLE_TO_PLACE_TYPES.values())

        counts['total'] = counts['broad'] + counts['ce_specific'] + counts['type_based']

        return counts


# ============================================================================
# ENTITY MAPPER
# ============================================================================

class EntityMapper:
    """Maps Google Places results to CSV format"""

    @staticmethod
    def map_to_csv_row(place: Dict, ecosystem_role: str, search_query: str) -> Dict:
        """Convert a Google Place to a CSV row"""
        geometry = place.get('geometry', {})
        location = geometry.get('location', {})

        # Handle both 'types' (list from search) and 'type' (string from details)
        types = place.get('types', [])
        if not types:
            type_value = place.get('type', '')
            types = [type_value] if type_value else []

        return {
            'name': place.get('name', ''),
            'address': place.get('formatted_address', place.get('vicinity', '')),
            'phone': place.get('formatted_phone_number', ''),
            'website': place.get('website', ''),
            'latitude': location.get('lat', ''),
            'longitude': location.get('lng', ''),
            'google_place_id': place.get('place_id', ''),
            'place_types': ', '.join(types),
            'business_status': place.get('business_status', ''),
            'rating': place.get('rating', ''),
            'user_ratings_total': place.get('user_ratings_total', ''),
            'ecosystem_role': ecosystem_role,
            'search_query': search_query,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# MAIN EXTRACTOR
# ============================================================================

class CEActorExtractor:
    """Main extraction orchestrator"""

    def __init__(self, api_key: str, requests_per_second: float = 0.5):
        # Conservative rate: 1 request every 2 seconds (0.5 QPS = 30 QPM)
        # Well below Google's 60 QPM limit, safe for long-running extraction
        self.client = GooglePlacesClient(api_key, requests_per_second=requests_per_second)
        self.seen_place_ids: Set[str] = set()
        self.results: List[Dict] = []

    def extract_city(self, city_key: str,
                      limit_queries: Optional[int] = None,
                      include_broad: bool = True,
                      include_ce_specific: bool = True,
                      include_type_based: bool = True,
                      use_expanded_terms: bool = True,
                      use_reduced_ce_keywords: bool = False,
                      max_pages: int = 3,
                      fetch_details: bool = False):
        """Extract CE actors for a specific city.

        Args:
            city_key: City identifier (e.g., 'novi_sad', 'cahul')
            limit_queries: Maximum number of queries to execute (for testing)
            include_broad: Include role-only searches (more results, less specific)
            include_ce_specific: Include CE keyword + role searches (targeted)
            include_type_based: Include Google Places type-based searches
            use_expanded_terms: Use expanded Serbian/local language terms
            use_reduced_ce_keywords: Use reduced set of 5 CE keywords (budget-conscious)
            max_pages: Maximum pages to fetch per query (1-3, each page ~20 results)
            fetch_details: Whether to fetch detailed info for each place (adds cost)
        """
        city_config = CITIES[city_key]
        print(f"\n{'='*80}")
        print(f"Extracting CE Actors for {city_config['name']}")
        print(f"{'='*80}\n")

        # Generate queries with selected strategies
        query_builder = QueryBuilder(
            city_config,
            use_expanded_terms=use_expanded_terms,
            use_reduced_ce_keywords=use_reduced_ce_keywords,
            city_key=city_key
        )

        # Get query count estimate
        estimates = query_builder.get_query_count_estimate(
            include_broad=include_broad,
            include_ce_specific=include_ce_specific,
            include_type_based=include_type_based
        )

        # Generate queries
        queries = query_builder.generate_queries(
            include_broad=include_broad,
            include_ce_specific=include_ce_specific
        )

        # Add type-based queries if requested
        if include_type_based:
            type_queries = query_builder.generate_type_based_queries()
            queries.extend(type_queries)

        if limit_queries:
            queries = queries[:limit_queries]
            print(f"LIMITED MODE: Processing only {limit_queries} queries\n")

        # Print configuration summary
        print(f"Query Strategy Configuration:")
        print(f"  - Broad role queries: {'Yes' if include_broad else 'No'} ({estimates['broad']} queries)")
        print(f"  - CE-specific queries: {'Yes' if include_ce_specific else 'No'} ({estimates['ce_specific']} queries)")
        print(f"  - Type-based queries: {'Yes' if include_type_based else 'No'} ({estimates['type_based']} queries)")
        print(f"  - Expanded terms: {'Yes' if use_expanded_terms else 'No'}")
        print(f"  - Max pages per query: {max_pages} (up to {max_pages * 20} results each)")
        print(f"  - Fetch place details: {'Yes' if fetch_details else 'No'}")
        print(f"\nTotal queries to execute: {len(queries)}")
        print(f"Estimated API calls: {len(queries)} - {len(queries) * max_pages} (with pagination)")
        print(f"Radius: {city_config['radius']/1000}km")
        print(f"Location: {city_config['location']}")
        print(f"Languages: {', '.join(city_config['languages'])}\n")

        # Estimate cost
        max_searches = len(queries) * max_pages
        estimated_cost = max_searches * 0.032  # $0.032 per search
        print(f"Estimated maximum cost: ${estimated_cost:.2f} (€{estimated_cost * 0.92:.2f})")
        if fetch_details:
            # Assume ~30% unique results that need details
            detail_cost = max_searches * 0.3 * 20 * 0.005  # ~30% unique × 20 results × $0.005
            print(f"  + Place details estimate: ${detail_cost:.2f} (€{detail_cost * 0.92:.2f})")
        print()

        # Execute searches
        for query_info in tqdm(queries, desc="Searching", unit="query"):
            places = self.client.search_places(
                query=query_info['query'],
                location=city_config['location'],
                radius=city_config['radius'],
                max_pages=max_pages
            )

            # Process each place
            for place in places:
                place_id = place.get('place_id')

                # Skip duplicates
                if place_id in self.seen_place_ids:
                    continue

                self.seen_place_ids.add(place_id)

                # Get detailed info if requested and place has potential
                if fetch_details and self._is_promising(place):
                    details = self.client.get_place_details(place_id)
                    if details:
                        place.update(details)

                # Map to CSV row
                row = EntityMapper.map_to_csv_row(
                    place,
                    query_info['ecosystem_role'],
                    query_info['query']
                )
                self.results.append(row)

        print(f"\nExtraction complete!")
        print(f"Total unique places found: {len(self.results)}")
        print(f"API requests made: {self.client.request_count}")

        # Print quota status
        status = self.client.quota.get_status()
        print(f"\nQuota usage:")
        print(f"  Text searches: {status['text_searches']['used']}/{status['text_searches']['limit']} ({status['text_searches']['percentage']:.1f}%)")
        print(f"  Place details: {status['place_details']['used']}/{status['place_details']['limit']} ({status['place_details']['percentage']:.1f}%)")

        # Save results
        self._save_results(city_config)

    def _is_promising(self, place: Dict) -> bool:
        """Determine if a place is worth fetching detailed info"""
        # Always fetch details for now (we want complete data)
        return True

    def _save_results(self, city_config: Dict):
        """Save results to CSV"""
        if not self.results:
            print("No results to save.")
            return

        # Create output directory
        output_dir = Path(city_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        output_file = output_dir / f"google_places_ce_actors_{timestamp}.csv"

        # Convert to DataFrame and save
        df = pd.DataFrame(self.results)

        # Sort by ecosystem role and name
        df = df.sort_values(['ecosystem_role', 'name'])

        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to: {output_file}")

        # Print summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY BY ECOSYSTEM ROLE")
        print(f"{'='*80}")
        role_counts = df['ecosystem_role'].value_counts()
        for role, count in role_counts.items():
            print(f"  {role}: {count}")
        print(f"{'='*80}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract Circular Economy actors from Google Places API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction with all strategies (recommended for best coverage)
  python google_places_ce_extractor.py novi_sad

  # Quick test with 10 queries
  python google_places_ce_extractor.py novi_sad --limit 10

  # Budget-conscious: broad searches only, single page
  python google_places_ce_extractor.py novi_sad --no-ce-specific --max-pages 1

  # CE-focused only (narrower, but more relevant)
  python google_places_ce_extractor.py novi_sad --no-broad

  # Estimate costs before running
  python google_places_ce_extractor.py novi_sad --dry-run
        """
    )
    parser.add_argument(
        'city',
        choices=['novi_sad', 'cahul', 'both'],
        help='City to extract data for'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of queries for testing (default: all)'
    )
    parser.add_argument(
        '--api-key',
        default=API_KEY,
        help='Google Places API key (default: from config)'
    )

    # Query strategy options
    parser.add_argument(
        '--no-broad',
        action='store_true',
        help='Disable broad role-only searches (reduces queries but may miss entities)'
    )
    parser.add_argument(
        '--no-ce-specific',
        action='store_true',
        help='Disable CE keyword + role searches'
    )
    parser.add_argument(
        '--no-type-based',
        action='store_true',
        help='Disable Google Places type-based searches'
    )
    parser.add_argument(
        '--no-expanded-terms',
        action='store_true',
        help='Use original English-only terms instead of expanded Serbian terms'
    )
    parser.add_argument(
        '--reduced-ce-keywords',
        action='store_true',
        help='Use reduced set of 5 CE keywords (budget-conscious, ~€40 for full extraction)'
    )

    # Pagination and detail options
    parser.add_argument(
        '--max-pages',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='Max pages per query (1-3, each page ~20 results). Default: 3'
    )
    parser.add_argument(
        '--fetch-details',
        action='store_true',
        help='Fetch detailed info for each place (adds cost, more complete data)'
    )

    # Rate limiting
    parser.add_argument(
        '--qps',
        type=float,
        default=0.5,
        help='Requests per second (default: 0.5, i.e., 1 request every 2 seconds)'
    )

    # Dry run
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show query count and cost estimate without executing'
    )

    args = parser.parse_args()

    # Handle dry run
    if args.dry_run:
        print("\n=== DRY RUN - Cost Estimation ===\n")
        for city_key in (['novi_sad', 'cahul'] if args.city == 'both' else [args.city]):
            city_config = CITIES[city_key]
            query_builder = QueryBuilder(
                city_config,
                use_expanded_terms=not args.no_expanded_terms,
                use_reduced_ce_keywords=args.reduced_ce_keywords,
                city_key=city_key
            )
            estimates = query_builder.get_query_count_estimate(
                include_broad=not args.no_broad,
                include_ce_specific=not args.no_ce_specific,
                include_type_based=not args.no_type_based
            )
            total_queries = estimates['total']
            max_api_calls = total_queries * args.max_pages
            cost_usd = max_api_calls * 0.032
            cost_eur = cost_usd * 0.92

            print(f"City: {city_config['name']}")
            print(f"  Reduced CE keywords: {'Yes' if args.reduced_ce_keywords else 'No'}")
            print(f"  Broad queries: {estimates['broad']}")
            print(f"  CE-specific queries: {estimates['ce_specific']}")
            print(f"  Type-based queries: {estimates['type_based']}")
            print(f"  Total queries: {total_queries}")
            print(f"  Max API calls (with pagination): {max_api_calls}")
            print(f"  Estimated cost: ${cost_usd:.2f} (€{cost_eur:.2f})")
            print()
        return

    # Initialize extractor
    extractor = CEActorExtractor(args.api_key, requests_per_second=args.qps)

    # Common extraction parameters
    extract_params = {
        'limit_queries': args.limit,
        'include_broad': not args.no_broad,
        'include_ce_specific': not args.no_ce_specific,
        'include_type_based': not args.no_type_based,
        'use_expanded_terms': not args.no_expanded_terms,
        'use_reduced_ce_keywords': args.reduced_ce_keywords,
        'max_pages': args.max_pages,
        'fetch_details': args.fetch_details
    }

    # Extract for selected city/cities
    if args.city == 'both':
        extractor.extract_city('novi_sad', **extract_params)
        # Reset for second city
        extractor.seen_place_ids.clear()
        extractor.results.clear()
        extractor.extract_city('cahul', **extract_params)
    else:
        extractor.extract_city(args.city, **extract_params)

    print("\n✓ Extraction complete!")


if __name__ == '__main__':
    main()
