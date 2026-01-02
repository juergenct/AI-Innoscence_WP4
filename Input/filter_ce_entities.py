#!/usr/bin/env python3
"""
Post-processing filter for Google Places CE entity extraction results.

This script filters out irrelevant entities (hotels, restaurants, churches, etc.)
and optionally scores entities for CE-relevance based on their names and types.

Usage:
    python filter_ce_entities.py input.csv output.csv
    python filter_ce_entities.py input.csv output.csv --keep-all --add-scores
"""

import argparse
import re
from pathlib import Path
from typing import Set, Dict, List
import pandas as pd


# Place types that are typically NOT relevant for CE ecosystem mapping
IRRELEVANT_TYPES = {
    # Hospitality & Food
    'lodging', 'hotel', 'motel', 'hostel', 'resort',
    'restaurant', 'cafe', 'bar', 'night_club', 'bakery',
    'meal_delivery', 'meal_takeaway', 'food',

    # Religious
    'church', 'mosque', 'synagogue', 'hindu_temple', 'place_of_worship',

    # Health & Fitness (unless specifically CE-related)
    'gym', 'spa', 'beauty_salon', 'hair_care',

    # Entertainment
    'movie_theater', 'casino', 'bowling_alley', 'amusement_park',

    # Transportation hubs (generic)
    'airport', 'train_station', 'bus_station', 'transit_station',

    # Personal services
    'laundry', 'funeral_home', 'cemetery',

    # Generic retail (too broad)
    'clothing_store', 'shoe_store', 'jewelry_store',
}

# Place types that are ALWAYS relevant for CE ecosystem
ALWAYS_RELEVANT_TYPES = {
    'university', 'school', 'library',
    'local_government_office', 'city_hall', 'courthouse',
    'bank', 'finance', 'accounting',
}

# Keywords that indicate CE relevance (in name or address)
CE_POSITIVE_KEYWORDS = [
    # English
    'recycl', 'recycle', 'recycling',
    'circular', 'sustainability', 'sustainable',
    'environment', 'environmental', 'eco',
    'green', 'renewable', 'solar', 'wind', 'energy',
    'waste', 'reuse', 'repair', 'refurbish',
    'innovation', 'research', 'institute', 'university',
    'startup', 'incubator', 'accelerator',
    'ngo', 'foundation', 'association',

    # Serbian
    'recikla', 'cirkularn', 'održiv',
    'životna sredina', 'ekološk', 'zelena',
    'otpad', 'ponovna upotreba', 'popravka',
    'inovaci', 'istraživanj', 'institut', 'univerzitet', 'fakultet',
    'startup', 'inkubator', 'akcelerator',
    'udruženje', 'fondacija', 'organizacija',
]

# Keywords that indicate the entity is likely NOT CE-related
CE_NEGATIVE_KEYWORDS = [
    'hotel', 'hostel', 'motel', 'resort',
    'restaurant', 'restoran', 'cafe', 'kafić', 'bar', 'pub',
    'church', 'crkva', 'mosque', 'džamija', 'synagogue',
    'salon', 'spa', 'fitness', 'gym', 'teretana',
    'casino', 'kladionica',
]


def parse_place_types(types_str: str) -> Set[str]:
    """Parse comma-separated place types string into a set."""
    if pd.isna(types_str) or not types_str:
        return set()
    return {t.strip().lower() for t in str(types_str).split(',')}


def is_relevant_by_type(place_types: Set[str]) -> bool:
    """Check if entity is relevant based on place types."""
    # If it has any always-relevant type, keep it
    if place_types.intersection(ALWAYS_RELEVANT_TYPES):
        return True

    # If it has any irrelevant type, filter it out
    if place_types.intersection(IRRELEVANT_TYPES):
        return False

    # Default: keep it (we err on the side of inclusion)
    return True


def calculate_ce_score(row: pd.Series) -> float:
    """Calculate a CE-relevance score for an entity (0-100)."""
    score = 50.0  # Base score

    name = str(row.get('name', '')).lower()
    address = str(row.get('address', '')).lower()
    place_types = parse_place_types(row.get('place_types', ''))
    query = str(row.get('search_query', '')).lower()

    # Boost for positive keywords in name
    for keyword in CE_POSITIVE_KEYWORDS:
        if keyword in name:
            score += 10
            break  # Only count once per category

    # Boost for positive keywords in query (indicates targeted search found it)
    for keyword in CE_POSITIVE_KEYWORDS:
        if keyword in query:
            score += 5
            break

    # Boost for always-relevant types
    if place_types.intersection(ALWAYS_RELEVANT_TYPES):
        score += 15

    # Penalty for negative keywords
    for keyword in CE_NEGATIVE_KEYWORDS:
        if keyword in name:
            score -= 20
            break

    # Penalty for irrelevant types
    if place_types.intersection(IRRELEVANT_TYPES):
        score -= 25

    # Boost for having a website (indicates established entity)
    if row.get('website') and not pd.isna(row.get('website')):
        score += 5

    # Boost for higher ratings (indicates active business)
    rating = row.get('rating')
    if rating and not pd.isna(rating):
        try:
            rating = float(rating)
            if rating >= 4.5:
                score += 5
            elif rating >= 4.0:
                score += 3
        except (ValueError, TypeError):
            pass

    # Clamp to 0-100
    return max(0, min(100, score))


def filter_entities(df: pd.DataFrame,
                    remove_irrelevant: bool = True,
                    min_score: float = 0) -> pd.DataFrame:
    """Filter entities based on relevance criteria.

    Args:
        df: DataFrame with entity data
        remove_irrelevant: Remove entities with irrelevant place types
        min_score: Minimum CE score to keep (0-100)

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)

    # Calculate CE scores
    df['ce_score'] = df.apply(calculate_ce_score, axis=1)

    # Filter by type relevance
    if remove_irrelevant:
        df['_type_relevant'] = df['place_types'].apply(
            lambda x: is_relevant_by_type(parse_place_types(x))
        )
        df = df[df['_type_relevant']].drop(columns=['_type_relevant'])

    # Filter by minimum score
    if min_score > 0:
        df = df[df['ce_score'] >= min_score]

    filtered_count = len(df)
    print(f"Filtered: {original_count} -> {filtered_count} entities ({original_count - filtered_count} removed)")

    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics for the filtered data."""
    print(f"\n{'='*60}")
    print("FILTERED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total entities: {len(df)}")

    if 'ce_score' in df.columns:
        print(f"\nCE Score distribution:")
        print(f"  High relevance (>70): {len(df[df['ce_score'] > 70])}")
        print(f"  Medium relevance (50-70): {len(df[(df['ce_score'] >= 50) & (df['ce_score'] <= 70)])}")
        print(f"  Low relevance (<50): {len(df[df['ce_score'] < 50])}")

    print(f"\nBy Ecosystem Role:")
    role_counts = df['ecosystem_role'].value_counts()
    for role, count in role_counts.items():
        print(f"  {role}: {count}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Filter CE entity extraction results for relevance'
    )
    parser.add_argument(
        'input_file',
        help='Input CSV file from Google Places extraction'
    )
    parser.add_argument(
        'output_file',
        nargs='?',
        help='Output CSV file (default: input_filtered.csv)'
    )
    parser.add_argument(
        '--keep-all',
        action='store_true',
        help='Keep all entities, only add CE scores (no filtering)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0,
        help='Minimum CE score to keep (0-100, default: 0)'
    )
    parser.add_argument(
        '--add-scores',
        action='store_true',
        help='Add CE relevance scores to output'
    )

    args = parser.parse_args()

    # Determine output file
    input_path = Path(args.input_file)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"

    # Read input
    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} entities")

    # Filter
    df = filter_entities(
        df,
        remove_irrelevant=not args.keep_all,
        min_score=args.min_score
    )

    # Remove CE score column if not requested
    if not args.add_scores and 'ce_score' in df.columns:
        df = df.drop(columns=['ce_score'])

    # Sort by role and score (if available)
    sort_cols = ['ecosystem_role', 'name']
    if 'ce_score' in df.columns:
        sort_cols = ['ecosystem_role', 'ce_score', 'name']
        df = df.sort_values(sort_cols, ascending=[True, False, True])
    else:
        df = df.sort_values(sort_cols)

    # Save output
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")

    # Print summary
    print_summary(df)


if __name__ == '__main__':
    main()
