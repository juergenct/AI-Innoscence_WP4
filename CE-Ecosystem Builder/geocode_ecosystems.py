#!/usr/bin/env python3
"""
Geocode entities for Novi Sad and Cahul ecosystems using Google Maps API.

Usage:
    python geocode_ecosystems.py --api-key YOUR_API_KEY [--ecosystem novi_sad|cahul|all]
"""

import sqlite3
import argparse
import time
import requests
from pathlib import Path
from typing import Optional, Tuple


class SimpleGeocoder:
    """Simple Google Maps geocoder for batch geocoding."""

    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.last_request_time = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def geocode(self, address: str, region: str = "rs") -> Tuple[Optional[float], Optional[float]]:
        """Geocode an address using Google Maps API."""
        if not address or not address.strip():
            return None, None

        self._rate_limit()

        try:
            params = {
                'address': address.strip(),
                'key': self.api_key,
                'region': region,
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'OK' and data.get('results'):
                location = data['results'][0]['geometry']['location']
                return float(location['lat']), float(location['lng'])
            elif data.get('status') == 'ZERO_RESULTS':
                return None, None
            elif data.get('status') == 'REQUEST_DENIED':
                print(f"  API Error: {data.get('error_message', 'Request denied')}")
                return None, None
            else:
                return None, None

        except Exception as e:
            print(f"  Error geocoding '{address[:50]}...': {e}")
            return None, None


ECOSYSTEMS = {
    "novi_sad": {
        "name": "Novi Sad",
        "db_path": "novi_sad_ce_ecosystem/data/final/ecosystem.db",
        "region": "rs",  # Serbia
        "city_suffix": ", Novi Sad, Serbia",
    },
    "cahul": {
        "name": "Cahul",
        "db_path": "cahul_ce_ecosystem/data/final/ecosystem.db",
        "region": "md",  # Moldova
        "city_suffix": ", Cahul, Moldova",
    },
}


def geocode_ecosystem(ecosystem_key: str, api_key: str, base_path: Path) -> dict:
    """Geocode all entities in an ecosystem database."""

    config = ECOSYSTEMS[ecosystem_key]
    db_path = base_path / config["db_path"]

    print(f"\n{'='*60}")
    print(f"Geocoding {config['name']} Ecosystem")
    print(f"{'='*60}")

    if not db_path.exists():
        print(f"  Database not found: {db_path}")
        return {"error": "Database not found"}

    geocoder = SimpleGeocoder(api_key)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get entities with addresses but no coordinates
    cursor.execute("""
        SELECT url, entity_name, address
        FROM entity_profiles
        WHERE address IS NOT NULL
          AND address != ''
          AND (latitude IS NULL OR longitude IS NULL)
    """)

    entities = cursor.fetchall()

    stats = {
        "total": len(entities),
        "geocoded": 0,
        "failed": 0,
    }

    print(f"  Found {len(entities)} entities to geocode")

    for i, (url, name, address) in enumerate(entities, 1):
        # Format address with city/country for better results
        if config["city_suffix"].lower() not in address.lower():
            full_address = f"{address}{config['city_suffix']}"
        else:
            full_address = address

        lat, lon = geocoder.geocode(full_address, region=config["region"])

        if lat is not None and lon is not None:
            cursor.execute("""
                UPDATE entity_profiles
                SET latitude = ?, longitude = ?
                WHERE url = ?
            """, (lat, lon, url))
            stats["geocoded"] += 1
            status = "OK"
        else:
            stats["failed"] += 1
            status = "FAILED"

        # Progress
        print(f"  [{i}/{len(entities)}] {status}: {name[:40] if name else 'Unknown'}...")

        # Commit every 10 entities
        if i % 10 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\n  Summary: {stats['geocoded']} geocoded, {stats['failed']} failed")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Geocode entities for CE ecosystems using Google Maps API'
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='Google Maps API key'
    )
    parser.add_argument(
        '--ecosystem',
        choices=['novi_sad', 'cahul', 'all'],
        default='all',
        help='Ecosystem to geocode (default: all)'
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent

    print("="*60)
    print("CE ECOSYSTEM GEOCODING")
    print("="*60)

    ecosystems_to_process = (
        list(ECOSYSTEMS.keys()) if args.ecosystem == 'all'
        else [args.ecosystem]
    )

    results = {}
    for ecosystem in ecosystems_to_process:
        results[ecosystem] = geocode_ecosystem(ecosystem, args.api_key, base_path)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for ecosystem, stats in results.items():
        name = ECOSYSTEMS[ecosystem]["name"]
        if "error" in stats:
            print(f"  {name}: {stats['error']}")
        else:
            print(f"  {name}: {stats['geocoded']}/{stats['total']} geocoded")


if __name__ == '__main__':
    main()
