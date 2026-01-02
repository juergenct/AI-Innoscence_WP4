#!/usr/bin/env python3
"""
Enrichment script to fetch website URLs for existing Google Places data.
Uses the Place Details API to get website information for places already extracted.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm

try:
    import googlemaps
except ImportError:
    print("ERROR: googlemaps library not installed. Installing...")
    os.system("pip install -U googlemaps")
    import googlemaps


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

    def get(self, key: str) -> Optional[Dict]:
        return self.cache.get(key)

    def set(self, key: str, value: Dict):
        self.cache[key] = value
        self._save_cache()


class QuotaManager:
    """Manages API quota limits"""
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

    def can_make_details(self) -> bool:
        return self.quotas['place_details'] < self.MAX_PLACE_DETAILS

    def increment_details(self):
        self.quotas['place_details'] += 1
        self._save_quotas()

    def get_remaining(self) -> int:
        return self.MAX_PLACE_DETAILS - self.quotas['place_details']


class WebsiteEnricher:
    """Fetches website URLs for existing places"""

    def __init__(self, api_key: str, qps: float = 0.5):
        self.client = googlemaps.Client(key=api_key)
        self.min_interval = 1.0 / qps
        self.last_request_time = 0
        self.cache = CacheManager()
        self.quota = QuotaManager()
        self.api_calls = 0
        self.cache_hits = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def get_website(self, place_id: str) -> Optional[str]:
        """Fetch website URL for a place ID"""
        cache_key = f"details:{place_id}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached.get('website', '')

        # Check quota
        if not self.quota.can_make_details():
            print(f"\nQuota limit reached ({self.quota.MAX_PLACE_DETAILS})")
            return None

        # Make API call with retry
        max_retries = 3
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                result = self.client.place(
                    place_id=place_id,
                    fields=['website', 'formatted_phone_number']
                )
                self.quota.increment_details()
                self.api_calls += 1

                details = result.get('result', {})
                self.cache.set(cache_key, details)
                return details.get('website', '')

            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'quota' in error_msg:
                    if attempt < max_retries - 1:
                        delay = 2 ** (attempt + 1)
                        print(f"\nRate limit, waiting {delay}s...")
                        time.sleep(delay)
                        continue
                return None

        return None

    def enrich_csv(self, input_path: str, output_path: str = None):
        """Enrich CSV with website URLs"""
        print(f"\nLoading: {input_path}")
        df = pd.read_csv(input_path)
        total = len(df)

        print(f"Total records: {total}")
        print(f"Records already with websites: {df['website'].notna().sum()}")
        print(f"Quota remaining: {self.quota.get_remaining()}")
        print()

        # Get unique place IDs that need websites
        needs_website = df[df['website'].isna() | (df['website'] == '')]['google_place_id'].unique()
        print(f"Unique place IDs needing websites: {len(needs_website)}")

        # Fetch websites
        website_map = {}
        for place_id in tqdm(needs_website, desc="Fetching websites"):
            website = self.get_website(place_id)
            if website:
                website_map[place_id] = website

        # Update DataFrame
        def update_website(row):
            if pd.isna(row['website']) or row['website'] == '':
                return website_map.get(row['google_place_id'], '')
            return row['website']

        df['website'] = df.apply(update_website, axis=1)

        # Save results
        if output_path is None:
            # Create output path with timestamp
            input_p = Path(input_path)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            output_path = input_p.parent / f"{input_p.stem}_enriched_{timestamp}.csv"

        df.to_csv(output_path, index=False, encoding='utf-8')

        # Print summary
        websites_found = len([w for w in website_map.values() if w])
        print(f"\n{'='*60}")
        print("ENRICHMENT COMPLETE")
        print(f"{'='*60}")
        print(f"API calls made: {self.api_calls}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"Websites found: {websites_found}")
        print(f"Total records with websites: {df['website'].notna().sum() - (df['website'] == '').sum()}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}\n")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Enrich Google Places CSV with website URLs'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file with google_place_id column'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV file (default: adds _enriched_timestamp suffix)'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='Google Places API key'
    )
    parser.add_argument(
        '--qps',
        type=float,
        default=0.5,
        help='Requests per second (default: 0.5)'
    )

    args = parser.parse_args()

    enricher = WebsiteEnricher(args.api_key, qps=args.qps)
    enricher.enrich_csv(args.input, args.output)


if __name__ == '__main__':
    main()
