#!/usr/bin/env python3
"""
Preprocessing script for Novi Sad CE Ecosystem input data.
Combines multiple CSV files into a single JSON format for the CE-Ecosystem Builder pipeline.
"""

import csv
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Set, Tuple

# Configuration for each CSV file
CSV_CONFIGS = {
    'amcham_members.csv': {
        'url_col': 'website',
        'name_col': 'company_name',
        'category_col': None,
        'default_category': 'AmCham Members',
        'add_protocol': True
    },
    'crunchbase_companies_SRB_Novi_Sad.csv': {
        'url_col': 'homepage_url',
        'name_col': 'name',
        'category_col': 'type',
        'default_category': 'Crunchbase Companies',
        'add_protocol': False
    },
    'crunchbase_investors_SRB_Novi_Sad.csv': {
        'url_col': 'domain',
        'name_col': 'name',
        'category_col': None,
        'default_category': 'Crunchbase Investors',
        'add_protocol': True
    },
    'digital_serbia_members_friends.csv': {
        'url_col': 'website',
        'name_col': 'name',
        'category_col': 'group',
        'default_category': 'Digital Serbia',
        'add_protocol': False
    },
    'google_places_ce_actors_2025-12-05_141957_enriched_2025-12-15_102841.csv': {
        'url_col': 'website',
        'name_col': 'name',
        'category_col': 'ecosystem_role',
        'default_category': 'Google Places CE Actors',
        'add_protocol': False
    },
    'ntpns_members.csv': {
        'url_col': 'website',
        'name_col': 'name',
        'category_col': 'category',
        'default_category': 'NTPNS Members',
        'add_protocol': False
    },
    'startap_portal_serbia_organizations.csv': {
        'url_col': 'website',
        'name_col': 'name',
        'category_col': 'group',
        'default_category': 'Startap Portal Organizations',
        'add_protocol': False
    },
    'university_novisad_faculties.csv': {
        'url_col': 'website',
        'name_col': 'faculty_name',
        'category_col': None,
        'default_category': 'University of Novi Sad - Faculties',
        'add_protocol': False
    },
    'university_novisad_laboratories.csv': {
        'url_col': 'website',
        'name_col': 'entity_name',
        'category_col': None,
        'default_category': 'University of Novi Sad - Laboratories',
        'add_protocol': False
    },
    'university_novisad_scientific_centers.csv': {
        'url_col': 'website',
        'name_col': 'center_name',
        'category_col': None,
        'default_category': 'University of Novi Sad - Scientific Centers',
        'add_protocol': False
    }
}

def normalize_url(url: str, add_protocol: bool = False) -> str:
    """
    Normalize URL by adding protocol if missing and cleaning up.

    Args:
        url: Raw URL string
        add_protocol: Whether to add https:// if protocol is missing

    Returns:
        Normalized URL or empty string if invalid
    """
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()

    # Skip empty or placeholder URLs
    if not url or url.lower() in ['n/a', 'none', 'null', '-']:
        return ""

    # Add protocol if missing
    if add_protocol and not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Parse and validate
    try:
        parsed = urlparse(url)
        # Must have a valid scheme and netloc
        if parsed.scheme in ['http', 'https'] and parsed.netloc:
            # Reconstruct URL without trailing slash
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            # Remove trailing slash unless it's the root
            if normalized.endswith('/') and parsed.path != '/':
                normalized = normalized[:-1]
            return normalized
    except Exception:
        pass

    return ""

def process_csv_file(
    file_path: Path,
    config: Dict,
    seen_urls: Set[str]
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, int]]:
    """
    Process a single CSV file and extract URLs with metadata.

    Args:
        file_path: Path to CSV file
        config: Configuration for this CSV file
        seen_urls: Set of URLs already processed (for deduplication)

    Returns:
        Tuple of (category_urls, url_to_name, stats)
    """
    category_urls = {}
    url_to_name = {}
    stats = {
        'total_rows': 0,
        'valid_urls': 0,
        'skipped_no_url': 0,
        'skipped_duplicate': 0
    }

    url_col = config['url_col']
    name_col = config['name_col']
    category_col = config['category_col']
    default_category = config['default_category']
    add_protocol = config.get('add_protocol', False)

    print(f"Processing: {file_path.name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                stats['total_rows'] += 1

                # Extract and normalize URL
                raw_url = row.get(url_col, '').strip()
                url = normalize_url(raw_url, add_protocol)

                if not url:
                    stats['skipped_no_url'] += 1
                    continue

                # Skip duplicates
                if url in seen_urls:
                    stats['skipped_duplicate'] += 1
                    continue

                seen_urls.add(url)
                stats['valid_urls'] += 1

                # Extract entity name
                entity_name = row.get(name_col, '').strip()
                if entity_name:
                    url_to_name[url] = entity_name

                # Determine category
                category = default_category
                if category_col and category_col in row:
                    custom_category = row[category_col].strip()
                    if custom_category:
                        # Create more readable category names
                        category = f"{default_category} - {custom_category.title()}"

                # Add to category
                if category not in category_urls:
                    category_urls[category] = []
                category_urls[category].append(url)

    except FileNotFoundError:
        print(f"  Warning: File not found - {file_path.name}")
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")

    print(f"  Valid URLs: {stats['valid_urls']}, Skipped (no URL): {stats['skipped_no_url']}, Skipped (duplicate): {stats['skipped_duplicate']}")

    return category_urls, url_to_name, stats

def main():
    """Main preprocessing function."""
    # Paths
    script_dir = Path(__file__).parent
    input_dir = script_dir
    output_dir = script_dir.parent.parent / "CE-Ecosystem Builder" / "novi_sad_ce_ecosystem" / "data" / "input"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track all URLs and metadata
    all_category_urls = {}
    all_url_to_name = {}
    seen_urls = set()
    total_stats = {
        'files_processed': 0,
        'total_rows': 0,
        'valid_urls': 0,
        'skipped_no_url': 0,
        'skipped_duplicate': 0
    }

    # Process each CSV file
    for csv_file, config in CSV_CONFIGS.items():
        file_path = input_dir / csv_file

        if not file_path.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue

        category_urls, url_to_name, stats = process_csv_file(file_path, config, seen_urls)

        # Merge results
        for category, urls in category_urls.items():
            if category not in all_category_urls:
                all_category_urls[category] = []
            all_category_urls[category].extend(urls)

        all_url_to_name.update(url_to_name)

        # Update stats
        total_stats['files_processed'] += 1
        total_stats['total_rows'] += stats['total_rows']
        total_stats['valid_urls'] += stats['valid_urls']
        total_stats['skipped_no_url'] += stats['skipped_no_url']
        total_stats['skipped_duplicate'] += stats['skipped_duplicate']

        print()

    # Generate category statistics
    category_stats = {category: len(urls) for category, urls in all_category_urls.items()}

    # Save entity_urls.json
    entity_urls_path = output_dir / "entity_urls.json"
    with open(entity_urls_path, 'w', encoding='utf-8') as f:
        json.dump(all_category_urls, f, indent=2, ensure_ascii=False)

    print(f"Saved entity URLs to: {entity_urls_path}")

    # Save entity_metadata.json
    metadata = {
        'url_to_name': all_url_to_name,
        'total_entities': len(seen_urls),
        'categories': category_stats,
        'processing_stats': {
            'files_processed': total_stats['files_processed'],
            'total_input_rows': total_stats['total_rows'],
            'valid_urls_extracted': total_stats['valid_urls'],
            'skipped_no_url': total_stats['skipped_no_url'],
            'skipped_duplicate': total_stats['skipped_duplicate']
        }
    }

    entity_metadata_path = output_dir / "entity_metadata.json"
    with open(entity_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved entity metadata to: {entity_metadata_path}")
    print("=" * 80)

    # Print summary
    print("\nSUMMARY:")
    print(f"  Files processed: {total_stats['files_processed']}")
    print(f"  Total input rows: {total_stats['total_rows']}")
    print(f"  Valid unique URLs: {total_stats['valid_urls']}")
    print(f"  Skipped (no URL): {total_stats['skipped_no_url']}")
    print(f"  Skipped (duplicate): {total_stats['skipped_duplicate']}")
    print(f"\n  Categories: {len(category_stats)}")
    for category, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        print(f"    - {category}: {count} entities")

    print(f"\n✓ Preprocessing complete! Data ready for CE-Ecosystem Builder pipeline.")
    print(f"\nNext step: Run the pipeline with:")
    print(f"  cd '{script_dir.parent.parent / 'CE-Ecosystem Builder'}'")
    print(f"  python -m novi_sad_ce_ecosystem.main")

if __name__ == "__main__":
    main()
