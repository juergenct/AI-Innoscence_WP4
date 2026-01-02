#!/usr/bin/env python3
"""
Convenience script to run Novi Sad CE Ecosystem Builder.

This script runs the full pipeline for the Novi Sad circular economy ecosystem:
1. Verification (CE relevance + Novi Sad location)
2. Extraction (detailed entity profiles)
3. Geocoding (coordinates assignment)
4. Relationship Analysis (partnerships and synergies)
5. Graph Building (ecosystem map)

Expected runtime: 8-12 hours for ~1,256 URLs
Expected output: ~400-500 verified entities with full profiles
"""
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

from novi_sad_ce_ecosystem.main import main

if __name__ == '__main__':
    print("=" * 80)
    print("Starting Novi Sad CE Ecosystem Builder")
    print("=" * 80)
    print("\nThis will run the full 5-stage pipeline:")
    print("  1. Verification (CE relevance + Novi Sad location)")
    print("  2. Extraction (entity profiles)")
    print("  3. Geocoding (coordinates)")
    print("  4. Relationship Analysis (partnerships)")
    print("  5. Graph Building (ecosystem map)")
    print("\nExpected runtime: 8-12 hours")
    print("Progress can be monitored via checkpoint.json")
    print("\nPress Ctrl+C to interrupt (pipeline will resume from checkpoint)")
    print("=" * 80)
    print()

    main()
