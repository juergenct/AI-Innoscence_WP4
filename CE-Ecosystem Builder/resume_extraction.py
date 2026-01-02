#!/usr/bin/env python3
"""
Resume scraping from extraction stage (skips verification).
Loads already-verified entities and continues from Stage 2.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 70)
    print("RESUMING FROM EXTRACTION STAGE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load already-verified entities (Stage 1 - SKIPPED)")
    print("  2. Resume extraction (Stage 2)")
    print("  3. Geocode addresses (Stage 3)")
    print("  4. Analyze relationships (Stage 4)")
    print("  5. Build ecosystem graph (Stage 5)")
    print("=" * 70)

    # Check if Ollama is running
    try:
        subprocess.run(['ollama', 'list'], check=True, capture_output=True)
    except Exception:
        print("\nERROR: Ollama is not running!")
        print("Please start Ollama first with: ollama serve")
        sys.exit(1)

    print("\n✓ Ollama is running")

    # Load verified entities from CSV
    verified_csv = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'verified' / 'verification_results.csv'

    if not verified_csv.exists():
        print(f"\nERROR: Verification results not found at {verified_csv}")
        print("You need to run Stage 1 (verification) first.")
        sys.exit(1)

    print(f"\n✓ Loading verified entities from {verified_csv.name}")

    try:
        df = pd.read_csv(verified_csv)
        verified_entities = df.to_dict('records')
        print(f"✓ Loaded {len(verified_entities)} verified entities")
    except Exception as e:
        print(f"\nERROR: Failed to load verification results: {e}")
        sys.exit(1)

    # Import and run the scraper from Stage 2 onwards
    print("\n" + "=" * 70)
    print("STARTING FROM STAGE 2: EXTRACTION")
    print("=" * 70)

    try:
        from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor

        # Initialize processor
        config_path = base_dir / 'hamburg_ce_ecosystem' / 'config' / 'scrape_config.yaml'
        input_file = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'input' / 'entity_urls.json'

        processor = BatchProcessor(input_file, config_path=config_path)

        # Skip Stage 1, run from Stage 2 onwards
        print(f"\n📊 Stage 2: Extracting entity information...")
        processor.logger.info("Stage 2: Extracting entity information...")
        extracted = processor.run_extraction(verified_entities)

        print(f"\n✅ Stage 2 complete: {len(extracted)} entities extracted")
        print(f"🌍 Stage 3: Geocoding entities...")
        processor.logger.info("Stage 3: Geocoding entities...")
        geocoded = processor.geocode_entities(extracted)

        print(f"\n✅ Stage 3 complete: {len(geocoded)} entities geocoded")

        # Relationship and ecosystem analysis
        if len(geocoded) > 0:
            print(f"\n🔗 Stage 4: Analyzing relationships and ecosystem...")
            processor.logger.info("Stage 4: Analyzing relationships and ecosystem...")
            relationships, insights = processor.run_relationship_analysis(geocoded)

            print(f"\n✅ Stage 4 complete:")
            print(f"  - {len(relationships)} relationships identified")
            print(f"  - {len([i for i in insights if i.get('insight_type') == 'synergy'])} synergies discovered")
            print(f"  - {len([i for i in insights if i.get('insight_type') == 'gap'])} gaps identified")
            print(f"  - {len([i for i in insights if i.get('insight_type') == 'recommendation'])} recommendations generated")

            print(f"\n📈 Stage 5: Building enhanced ecosystem graph...")
            processor.logger.info("Stage 5: Building enhanced ecosystem graph...")
            graph = processor.build_ecosystem_graph(geocoded, relationships, insights)
            processor.save_results(geocoded, graph, relationships)
        else:
            # Fallback: build basic graph without relationships
            graph = processor.build_ecosystem_graph(geocoded)
            processor.save_results(geocoded, graph)

        print(f"\n✅ All stages complete!")

        # Success
        print("\n" + "=" * 70)
        print("✅ EXTRACTION PIPELINE COMPLETE!")
        print("=" * 70)
        print("\nResults are available in:")
        print("  - hamburg_ce_ecosystem/data/final/ecosystem_entities.csv")
        print("  - hamburg_ce_ecosystem/data/final/ecosystem_map.json")
        print("  - hamburg_ce_ecosystem/data/final/ecosystem.db")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
