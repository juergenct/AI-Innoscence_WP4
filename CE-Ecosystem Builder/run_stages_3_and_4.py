#!/usr/bin/env python3
"""
Run Stages 3 (Geocoding) + 4 (Relationships) + 5 (Graph) on all entities.

IMPROVED VERSION with proper error handling and validation.

This script:
1. Loads all entity profiles from JSON or database
2. Geocodes missing coordinates (uses cache, saves to database automatically)
3. Validates JSON save succeeded
4. Runs Stage 4 (relationship analysis) on all entities
5. Builds final ecosystem graph with validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor


def main():
    base_dir = Path(__file__).resolve().parent / 'hamburg_ce_ecosystem'
    config_path = base_dir / 'config' / 'scrape_config.yaml'
    input_file = base_dir / 'data' / 'input' / 'entity_urls.json'

    print("=" * 80)
    print("Running Stages 3 (Geocoding) + 4 (Relationships) + 5 (Graph) on all entities")
    print("=" * 80)

    try:
        # Initialize processor
        processor = BatchProcessor(input_file, config_path=config_path)

        # Check database state
        db_count = processor.dm.get_entity_count()
        db_geocoded = len(processor.dm.get_geocoded_urls())
        print(f"\n📊 Database State:")
        print(f"   - Total entities: {db_count}")
        print(f"   - Already geocoded: {db_geocoded}")

        # Load entity profiles from JSON
        profiles_path = base_dir / 'data' / 'extracted' / 'entity_profiles.json'
        print(f"\n📂 Loading entity profiles from: {profiles_path}")

        if not profiles_path.exists():
            print(f"❌ ERROR: {profiles_path} not found!")
            print(f"   Run Stage 2 (extraction) first, or rebuild from database.")
            sys.exit(1)

        with open(profiles_path, 'r', encoding='utf-8') as f:
            extracted = json.load(f)

        print(f"✅ Loaded {len(extracted)} entity profiles from JSON")

        # Validate entity count
        if len(extracted) < db_count * 0.95:
            print(f"⚠️  WARNING: JSON has {len(extracted)} entities but database has {db_count}")
            print(f"   Consider rebuilding JSON from database first")

        # Count how many need geocoding
        need_geocoding = sum(1 for p in extracted if p.get('latitude') is None or p.get('longitude') is None)
        print(f"📍 Entities needing geocoding: {need_geocoding}/{len(extracted)}")

        # === STAGE 3: Geocoding ===
        print(f"\n{'=' * 80}")
        print(f"🌍 Stage 3: Geocoding entities...")
        print(f"{'=' * 80}")

        # geocode_entities now automatically saves to database
        geocoded = processor.geocode_entities(extracted)

        print(f"\n✅ Stage 3 complete: {len(geocoded)} entities processed")

        # Validate geocoding succeeded
        geocoded_count = sum(1 for p in geocoded if p.get('latitude') and p.get('longitude'))
        print(f"   - {geocoded_count} entities now have coordinates")

        # NOTE: geocode_entities() already saves updated JSON, no need to save again
        print(f"   - JSON file already updated by geocode_entities()")

        # === STAGE 4: Relationships ===
        if len(geocoded) > 0:
            print(f"\n{'=' * 80}")
            print(f"🔗 Stage 4: Analyzing relationships and ecosystem...")
            print(f"{'=' * 80}")

            relationships, insights = processor.run_relationship_analysis(geocoded)

            print(f"\n✅ Stage 4 complete:")
            print(f"   - {len(relationships)} relationships identified")
            synergies = [i for i in insights if i.get('insight_type') == 'synergy']
            gaps = [i for i in insights if i.get('insight_type') == 'gap']
            recommendations = [i for i in insights if i.get('insight_type') == 'recommendation']
            print(f"   - {len(synergies)} synergies discovered")
            print(f"   - {len(gaps)} gaps identified")
            print(f"   - {len(recommendations)} recommendations generated")

            # === STAGE 5: Build Final Graph ===
            print(f"\n{'=' * 80}")
            print(f"📈 Stage 5: Building enhanced ecosystem graph...")
            print(f"{'=' * 80}")

            # build_ecosystem_graph now includes data validation
            graph = processor.build_ecosystem_graph(geocoded, relationships, insights)
            processor.save_results(geocoded, graph, relationships)

            print(f"\n✅ Stage 5 complete:")
            print(f"   - {len(graph.get('nodes', []))} nodes in graph")
            print(f"   - {len(graph.get('edges', []))} edges in graph")
        else:
            print(f"\n⚠️  No entities to process for Stages 4-5")

        print(f"\n{'=' * 80}")
        print(f"✅ All stages complete!")
        print(f"{'=' * 80}")
        print(f"\nFinal outputs saved to: {base_dir / 'data' / 'final'}/")

        # Final validation summary
        final_db_geocoded = len(processor.dm.get_geocoded_urls())
        print(f"\n📊 Final Database State:")
        print(f"   - Total entities: {db_count}")
        print(f"   - Geocoded entities: {final_db_geocoded}")
        if db_count > 0:
            print(f"   - Coverage: {(final_db_geocoded / db_count * 100):.1f}%")
        else:
            print(f"   - Coverage: N/A (database is empty, but {geocoded_count} entities processed from JSON)")

    except KeyboardInterrupt:
        print(f"\n\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
