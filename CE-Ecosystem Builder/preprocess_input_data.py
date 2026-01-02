#!/usr/bin/env python3
"""
Preprocessing script to consolidate all CSV files from Input Data directory
into the format expected by the Hamburg CE Ecosystem scraper.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


# Define CSV file configurations with their URL and name columns
CSV_CONFIGS = {
    'umweltpartner_enriched.csv': {
        'url_col': 'website_url',
        'name_col': 'name',
        'category_col': None,
        'default_category': 'Umweltpartner'
    },
    'hamburg_branchenbuch_filtered_categories.csv': {
        'url_col': 'website',
        'name_col': 'name',
        'category_col': 'category',
        'default_category': 'Hamburg Branchenbuch'
    },
    'crunchbase_companies_DEU_Hamburg.csv': {
        'url_col': 'homepage_url',
        'name_col': 'name',
        'category_col': 'type',
        'default_category': 'Crunchbase Hamburg Companies'
    },
    'tuhh_institutes_20250928_170519.csv': {
        'url_col': 'url',
        'name_col': 'name',
        'category_col': None,
        'default_category': 'TUHH Institutes'
    },
    'additional_ce_entities.csv': {
        'url_col': 'URL',
        'name_col': 'Name',
        'category_col': None,
        'default_category': 'Additional CE Entities'
    },
    'bde_foerdermitglieder.csv': {
        'url_col': 'website_url',
        'name_col': 'name',
        'category_col': 'category',
        'default_category': 'BDE Fördermitglieder'
    },
    'bde_netzwerk.csv': {
        'url_col': 'url',
        'name_col': 'name',
        'category_col': 'section',
        'default_category': 'BDE Netzwerk'
    },
    'bde_scrape.csv': {
        'url_col': 'url',
        'name_col': 'name',
        'category_col': 'section',
        'default_category': 'BDE'
    },
    'claude_deep_research_entities.csv': {
        'url_col': 'website_url',
        'name_col': 'organization',
        'category_col': 'section',
        'default_category': 'Claude Research Entities'
    },
    'eeh_mitglieder.csv': {
        'url_col': 'website_url',
        'name_col': 'name',
        'category_col': 'category',
        'default_category': 'EEH Mitglieder'
    },
    'gemini_deep_research_entities.csv': {
        'url_col': 'website_url',
        'name_col': 'organization',
        'category_col': 'section',
        'default_category': 'Gemini Research Entities'
    },
    'openalex_institution_directory_with_websites.csv': {
        'url_col': 'inst_homepage_url',
        'name_col': 'institution_display_name',
        'category_col': None,
        'default_category': 'OpenAlex Institutions'
    },
    'retech_scrape.csv': {
        'url_col': 'url',
        'name_col': 'name',
        'category_col': 'section',
        'default_category': 'RETech'
    }
}


def is_valid_url(url: str) -> bool:
    """Check if a URL is valid and not empty."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not url:
        return False
    # Basic URL validation
    return url.startswith(('http://', 'https://'))


def clean_url(url: str) -> str:
    """Clean and normalize URLs."""
    url = url.strip()
    # Remove trailing slashes for consistency
    if url.endswith('/'):
        url = url[:-1]
    return url


def clean_category(category: str, default: str) -> str:
    """Clean and normalize category names."""
    if not category or not isinstance(category, str):
        return default
    category = category.strip()
    if not category:
        return default
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in category.split())


def read_csv_file(file_path: Path, config: Dict) -> tuple[List[tuple[str, str]], Dict[str, str]]:
    """
    Read a CSV file and extract URLs and metadata.
    
    Returns:
        - List of (url, category) tuples
        - Dictionary mapping URLs to entity names
    """
    urls_with_categories = []
    url_to_name = {}
    
    url_col = config['url_col']
    name_col = config['name_col']
    category_col = config.get('category_col')
    default_category = config['default_category']
    
    print(f"  Reading {file_path.name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Validate that required columns exist
            if not reader.fieldnames:
                print(f"    WARNING: No columns found in {file_path.name}")
                return urls_with_categories, url_to_name
            
            if url_col not in reader.fieldnames:
                print(f"    WARNING: Column '{url_col}' not found in {file_path.name}")
                return urls_with_categories, url_to_name
            
            if name_col not in reader.fieldnames:
                print(f"    WARNING: Column '{name_col}' not found in {file_path.name}")
                return urls_with_categories, url_to_name
            
            count = 0
            skipped = 0
            
            for row in reader:
                url = row.get(url_col, '').strip()
                
                if not is_valid_url(url):
                    skipped += 1
                    continue
                
                url = clean_url(url)
                name = row.get(name_col, '').strip()
                
                # Get category
                if category_col and category_col in reader.fieldnames:
                    category = clean_category(row.get(category_col, ''), default_category)
                else:
                    category = default_category
                
                urls_with_categories.append((url, category))
                if name:
                    url_to_name[url] = name
                
                count += 1
            
            print(f"    ✓ Extracted {count} URLs, skipped {skipped} invalid entries")
    
    except Exception as e:
        print(f"    ERROR reading {file_path.name}: {e}")
    
    return urls_with_categories, url_to_name


def consolidate_all_csvs(input_dir: Path) -> tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Read all CSV files and consolidate them into category-based structure.
    
    Returns:
        - Dictionary mapping categories to lists of URLs
        - Dictionary mapping URLs to entity names
    """
    category_to_urls = defaultdict(list)
    url_to_name = {}
    all_urls: Set[str] = set()
    
    print("\n" + "=" * 70)
    print("Processing CSV files from Input Data directory")
    print("=" * 70)
    
    for csv_filename, config in CSV_CONFIGS.items():
        csv_path = input_dir / csv_filename
        
        if not csv_path.exists():
            print(f"  ⚠ Skipping {csv_filename} (file not found)")
            continue
        
        urls_with_categories, names = read_csv_file(csv_path, config)
        
        # Add to consolidated data
        for url, category in urls_with_categories:
            if url not in all_urls:
                category_to_urls[category].append(url)
                all_urls.add(url)
            # Update name mapping (later entries override earlier ones)
            if url in names:
                url_to_name[url] = names[url]
    
    # Sort URLs within each category for consistency
    for category in category_to_urls:
        category_to_urls[category].sort()
    
    return dict(category_to_urls), url_to_name


def save_outputs(
    base_dir: Path,
    category_to_urls: Dict[str, List[str]],
    url_to_name: Dict[str, str]
) -> None:
    """Save the consolidated data to JSON files."""
    
    output_dir = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'input'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save entity_urls.json (required by the scraper)
    urls_file = output_dir / 'entity_urls.json'
    with open(urls_file, 'w', encoding='utf-8') as f:
        json.dump(category_to_urls, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {urls_file}")
    
    # Save entity_metadata.json (for reference and future use)
    metadata_file = output_dir / 'entity_metadata.json'
    metadata = {
        'url_to_name': url_to_name,
        'total_entities': len(url_to_name),
        'categories': {cat: len(urls) for cat, urls in category_to_urls.items()}
    }
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {metadata_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("CONSOLIDATION SUMMARY")
    print("=" * 70)
    print(f"Total unique URLs: {len(url_to_name)}")
    print(f"Total categories: {len(category_to_urls)}")
    print("\nURLs per category:")
    for category, urls in sorted(category_to_urls.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {category:50s} {len(urls):5d} URLs")
    print("=" * 70)


def main() -> None:
    """Main preprocessing function."""
    # Determine paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    input_data_dir = script_dir.parent / 'Input Data'
    
    if not input_data_dir.exists():
        print(f"ERROR: Input Data directory not found at {input_data_dir}")
        return
    
    # Process all CSV files
    category_to_urls, url_to_name = consolidate_all_csvs(input_data_dir)
    
    if not category_to_urls:
        print("\nERROR: No URLs were extracted from any CSV files!")
        return
    
    # Save outputs
    save_outputs(base_dir, category_to_urls, url_to_name)
    
    print("\n✓ Preprocessing complete! You can now run main.py to start scraping.\n")


if __name__ == '__main__':
    main()

