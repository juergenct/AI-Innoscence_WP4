#!/usr/bin/env python3
"""
Recovery script to identify entities that passed verification but failed extraction.

This script:
1. Loads verification_results.csv
2. Filters for Hamburg-based AND CE-related entities (should_extract=True)
3. Loads entity_profiles.json
4. Identifies failed extractions by set difference
5. Saves results to data/recovery/failed_entities_to_retry.csv
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def load_verification_results(csv_path: Path) -> pd.DataFrame:
    """Load and filter verification results for entities that should be extracted."""
    df = pd.read_csv(csv_path)

    # Filter for Hamburg-based AND CE-related entities
    filtered = df[
        (df['is_hamburg_based'] == True) &
        (df['is_ce_related'] == True)
    ].copy()

    print(f"Total verified entities: {len(df)}")
    print(f"Hamburg + CE entities (should extract): {len(filtered)}")

    return filtered


def load_extracted_entities(json_path: Path) -> set:
    """Load successfully extracted entity URLs from entity_profiles.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    extracted_urls = {profile['url'] for profile in profiles}
    print(f"Successfully extracted entities: {len(extracted_urls)}")

    return extracted_urls


def identify_failed_entities(verified_df: pd.DataFrame, extracted_urls: set) -> pd.DataFrame:
    """Identify entities that passed verification but failed extraction."""
    verified_urls = set(verified_df['url'])
    failed_urls = verified_urls - extracted_urls

    # Create dataframe of failed entities with full metadata
    failed_df = verified_df[verified_df['url'].isin(failed_urls)].copy()

    print(f"\nFailed extractions identified: {len(failed_df)}")
    print(f"Success rate: {len(extracted_urls) / len(verified_urls) * 100:.1f}%")

    # Add metadata
    failed_df['recovery_timestamp'] = datetime.now().isoformat()
    failed_df['retry_status'] = 'pending'

    return failed_df


def save_results(failed_df: pd.DataFrame, output_path: Path):
    """Save failed entities to CSV for retry processing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failed_df.to_csv(output_path, index=False)
    print(f"\nSaved failed entities to: {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total failed entities: {len(failed_df)}")

    if 'input_category' in failed_df.columns:
        print("\nBreakdown by category:")
        print(failed_df['input_category'].value_counts())

    if 'hamburg_confidence' in failed_df.columns:
        print(f"\nAverage Hamburg confidence: {failed_df['hamburg_confidence'].mean():.2f}")

    if 'ce_confidence' in failed_df.columns:
        print(f"Average CE confidence: {failed_df['ce_confidence'].mean():.2f}")


def main():
    """Main recovery identification workflow."""
    base_path = Path(__file__).parent / "hamburg_ce_ecosystem"

    # Define paths
    verification_csv = base_path / "data" / "verified" / "verification_results.csv"
    entity_profiles_json = base_path / "data" / "extracted" / "entity_profiles.json"
    output_csv = base_path / "data" / "recovery" / "failed_entities_to_retry.csv"

    # Check files exist
    if not verification_csv.exists():
        raise FileNotFoundError(f"Verification results not found: {verification_csv}")

    if not entity_profiles_json.exists():
        raise FileNotFoundError(f"Entity profiles not found: {entity_profiles_json}")

    print("=" * 60)
    print("Failed Extraction Recovery - Identification Phase")
    print("=" * 60)
    print()

    # Load data
    verified_df = load_verification_results(verification_csv)
    extracted_urls = load_extracted_entities(entity_profiles_json)

    # Identify failed entities
    failed_df = identify_failed_entities(verified_df, extracted_urls)

    # Save results
    save_results(failed_df, output_csv)

    print("\n" + "=" * 60)
    print("Identification complete! Next steps:")
    print("1. Review failed_entities_to_retry.csv")
    print("2. Run prompt improvements and validation updates")
    print("3. Execute retry_failed_extractions.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
