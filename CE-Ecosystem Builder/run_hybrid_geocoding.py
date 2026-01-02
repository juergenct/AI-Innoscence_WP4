#!/usr/bin/env python3
"""
Hybrid Geocoding Script for Hamburg CE Ecosystem
================================================

This script performs hybrid geocoding using:
1. Nominatim (OpenStreetMap) - Free, for initial geocoding
2. Google Maps API - For fallback on entities with Hamburg center coordinates

Usage:
    python run_hybrid_geocoding.py [--ecosystem ECOSYSTEM] [--google-api-key KEY] [--skip-nominatim] [--skip-google]

Arguments:
    --ecosystem: Ecosystem to geocode (default: hamburg_ce_ecosystem)
    --google-api-key: Google Maps API key (default: from environment or config)
    --skip-nominatim: Skip Nominatim geocoding and go straight to Google Maps
    --skip-google: Only run Nominatim geocoding, skip Google Maps fallback
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run hybrid geocoding (Nominatim + Google Maps) on CE Ecosystem data'
    )
    parser.add_argument(
        '--ecosystem',
        default='hamburg_ce_ecosystem',
        choices=['hamburg_ce_ecosystem', 'novi_sad_ce_ecosystem', 'cahul_ce_ecosystem'],
        help='Ecosystem to geocode (default: hamburg_ce_ecosystem)'
    )
    parser.add_argument(
        '--google-api-key',
        default='',
        help='Google Maps API key'
    )
    parser.add_argument(
        '--skip-nominatim',
        action='store_true',
        help='Skip Nominatim geocoding (only run Google Maps fallback)'
    )
    parser.add_argument(
        '--skip-google',
        action='store_true',
        help='Skip Google Maps fallback (only run Nominatim)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("🌍 HYBRID GEOCODING WORKFLOW")
    logger.info("="*80)
    logger.info(f"Ecosystem: {args.ecosystem}")
    logger.info(f"Skip Nominatim: {args.skip_nominatim}")
    logger.info(f"Skip Google Maps: {args.skip_google}")
    logger.info("="*80)

    # Import ecosystem-specific modules
    if args.ecosystem == 'hamburg_ce_ecosystem':
        from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor
        from hamburg_ce_ecosystem.utils.data_manager import DataManager
        data_dir = project_root / "CE-Ecosystem Builder" / args.ecosystem / "data"
    elif args.ecosystem == 'novi_sad_ce_ecosystem':
        from novi_sad_ce_ecosystem.scrapers.batch_processor import BatchProcessor
        from novi_sad_ce_ecosystem.utils.data_manager import DataManager
        data_dir = project_root / "CE-Ecosystem Builder" / args.ecosystem / "data"
    elif args.ecosystem == 'cahul_ce_ecosystem':
        from cahul_ce_ecosystem.scrapers.batch_processor import BatchProcessor
        from cahul_ce_ecosystem.utils.data_manager import DataManager
        data_dir = project_root / "CE-Ecosystem Builder" / args.ecosystem / "data"
    else:
        logger.error(f"Unknown ecosystem: {args.ecosystem}")
        sys.exit(1)

    # Initialize data manager and batch processor
    logger.info(f"Initializing {args.ecosystem}...")
    dm = DataManager(data_dir=data_dir)
    bp = BatchProcessor(data_dir=data_dir)

    # Load entity profiles
    profiles_path = data_dir / 'extracted' / 'entity_profiles.json'
    if not profiles_path.exists():
        logger.error(f"Entity profiles not found at {profiles_path}")
        logger.error("Please run the full scraping pipeline first.")
        sys.exit(1)

    logger.info(f"Loading entity profiles from {profiles_path}...")
    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    logger.info(f"Loaded {len(profiles)} entity profiles")

    # Count entities with Hamburg center coordinates
    HAMBURG_CENTER_LAT = 53.550341
    HAMBURG_CENTER_LON = 10.000654

    hamburg_center_count = sum(
        1 for p in profiles
        if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
           abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001
    )

    with_coords_count = sum(
        1 for p in profiles
        if p.get('latitude') is not None and p.get('longitude') is not None
    )

    logger.info(f"Current state:")
    logger.info(f"  - Total entities: {len(profiles)}")
    logger.info(f"  - With coordinates: {with_coords_count}")
    logger.info(f"  - With Hamburg center coords: {hamburg_center_count}")
    logger.info(f"  - Specific coordinates: {with_coords_count - hamburg_center_count}")

    # Stage 1: Nominatim Geocoding (with fixed resume logic)
    if not args.skip_nominatim:
        logger.info("\n" + "="*80)
        logger.info("🗺️  STAGE 1: Nominatim Geocoding (with fixed resume logic)")
        logger.info("="*80)

        profiles = bp.geocode_entities(profiles)

        # Count after Nominatim
        hamburg_center_count_after = sum(
            1 for p in profiles
            if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
               abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001
        )

        logger.info(f"\nAfter Nominatim geocoding:")
        logger.info(f"  - Hamburg center coords: {hamburg_center_count_after}")
        logger.info(f"  - Improvement: {hamburg_center_count - hamburg_center_count_after} entities geocoded")
    else:
        logger.info("⏭️  Skipping Nominatim geocoding")

    # Stage 2: Google Maps Fallback
    if not args.skip_google:
        if not args.google_api_key:
            logger.warning("⚠️  Google Maps API key not provided. Skipping Google Maps fallback.")
            logger.warning("   Use --google-api-key to provide a key.")
        else:
            logger.info("\n" + "="*80)
            logger.info("🗺️  STAGE 2: Google Maps Fallback Geocoding")
            logger.info("="*80)

            profiles = bp.geocode_entities_google_maps(profiles, api_key=args.google_api_key)
    else:
        logger.info("⏭️  Skipping Google Maps fallback")

    # Final Report
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL GEOCODING REPORT")
    logger.info("="*80)

    hamburg_center_final = sum(
        1 for p in profiles
        if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
           abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001
    )

    with_coords_final = sum(
        1 for p in profiles
        if p.get('latitude') is not None and p.get('longitude') is not None
    )

    specific_coords = with_coords_final - hamburg_center_final

    logger.info(f"Total entities: {len(profiles)}")
    logger.info(f"  - With specific coordinates: {specific_coords} ({specific_coords/len(profiles)*100:.1f}%)")
    logger.info(f"  - With Hamburg center coords: {hamburg_center_final} ({hamburg_center_final/len(profiles)*100:.1f}%)")
    logger.info(f"  - Without coordinates: {len(profiles) - with_coords_final}")

    # Breakdown of Hamburg center entities
    hamburg_with_address = [
        p for p in profiles
        if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
           abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001 and
           p.get('address') and
           str(p.get('address')).strip() and
           str(p.get('address')).strip().lower() not in ['unknown', 'na', 'n/a', 'none']
    ]

    hamburg_without_address = hamburg_center_final - len(hamburg_with_address)

    logger.info(f"\nHamburg center breakdown:")
    logger.info(f"  - With valid addresses (⚠️ should be geocoded): {len(hamburg_with_address)}")
    logger.info(f"  - Without addresses (✓ expected): {hamburg_without_address}")

    if hamburg_with_address:
        logger.warning(f"\n⚠️  {len(hamburg_with_address)} entities still have Hamburg center coords despite having addresses!")
        logger.warning("Examples:")
        for p in hamburg_with_address[:5]:
            logger.warning(f"  - {p.get('entity_name')}: {p.get('address')}")
        if len(hamburg_with_address) > 5:
            logger.warning(f"  ... and {len(hamburg_with_address) - 5} more")

    logger.info("\n✅ Hybrid geocoding workflow complete!")


if __name__ == '__main__':
    main()
