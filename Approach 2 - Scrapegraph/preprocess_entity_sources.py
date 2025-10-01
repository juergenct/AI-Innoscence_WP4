#!/usr/bin/env python3
"""
Preprocessing script to unify multiple CSV sources into entity_urls.json format.
This script reads various CSV files with entity names and websites, deduplicates them,
and creates a unified input file for the Hamburg CE Ecosystem Scraper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from urllib.parse import urlparse


def normalize_url(url: str | None) -> str | None:
    """Normalize URL and validate format."""
    if not url or pd.isna(url):
        return None
    
    url = str(url).strip()
    if not url:
        return None
    
    # Add https:// if no scheme present
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Basic validation
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return url
    except Exception:
        return None


def load_crunchbase(base_dir: Path) -> List[tuple[str, str, str]]:
    """Load Crunchbase companies. Returns list of (name, url, source)."""
    csv_path = base_dir / 'crunchbase_companies_DEU_Hamburg.csv'
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping...")
        return []
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        name = row.get('name', '')
        url = normalize_url(row.get('homepage_url'))
        if url and name:
            results.append((str(name), url, 'crunchbase'))
    
    print(f"✓ Loaded {len(results)} entities from Crunchbase")
    return results


def load_stakeholders_hamburg(base_dir: Path) -> List[tuple[str, str, str]]:
    """Load stakeholders_hamburg.csv. Returns list of (name, url, source)."""
    csv_path = base_dir / 'Approach 2 - Scrapegraph' / 'stakeholders_hamburg.csv'
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping...")
        return []
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        name = row.get('institution', '')
        url = normalize_url(row.get('website'))
        if url and name:
            results.append((str(name), url, 'stakeholders_hamburg'))
    
    print(f"✓ Loaded {len(results)} entities from stakeholders_hamburg.csv")
    return results


def load_branchenbuch(base_dir: Path) -> List[tuple[str, str, str]]:
    """Load Hamburg Branchenbuch companies. Returns list of (name, url, source)."""
    csv_path = base_dir / 'hamburg_branchenbuch_companies_details_from_map_20250930_162445_with_websites.csv'
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping...")
        return []
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        name = row.get('name', '')
        url = normalize_url(row.get('website'))
        if url and name:
            results.append((str(name), url, 'branchenbuch'))
    
    print(f"✓ Loaded {len(results)} entities from Branchenbuch")
    return results


def load_tuhh_institutes(base_dir: Path) -> List[tuple[str, str, str]]:
    """Load TUHH institutes. Returns list of (name, url, source)."""
    csv_path = base_dir / 'tuhh_institutes_20250928_170519.csv'
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping...")
        return []
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        name = row.get('name', '')
        url = normalize_url(row.get('url'))
        if url and name:
            results.append((str(name), url, 'tuhh_institutes'))
    
    print(f"✓ Loaded {len(results)} entities from TUHH institutes")
    return results


def load_openalex_institutions(base_dir: Path) -> List[tuple[str, str, str]]:
    """Load OpenAlex institutions. Returns list of (name, url, source)."""
    csv_path = base_dir / 'openalex_institution_directory_with_websites.csv'
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping...")
        return []
    
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        name = row.get('institution_display_name', '')
        url = normalize_url(row.get('inst_homepage_url'))
        if url and name:
            results.append((str(name), url, 'openalex'))
    
    print(f"✓ Loaded {len(results)} entities from OpenAlex")
    return results


def deduplicate_entities(entities: List[tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Deduplicate entities by URL and create a dictionary with metadata.
    Returns dict[url] = {name, source, sources_list}
    """
    url_to_entity: Dict[str, Dict] = {}
    
    for name, url, source in entities:
        if url in url_to_entity:
            # URL already exists, merge sources
            existing = url_to_entity[url]
            if source not in existing['sources']:
                existing['sources'].append(source)
        else:
            url_to_entity[url] = {
                'name': name,
                'url': url,
                'source': source,
                'sources': [source]
            }
    
    return url_to_entity


def create_entity_urls_json(
    deduplicated: Dict[str, Dict[str, str]],
    output_path: Path,
    category: str = "Entities to Classify"
) -> None:
    """
    Create entity_urls.json in the format expected by the scraper.
    Groups all entities under a single category since classification happens during scraping.
    """
    # Extract just the URLs for the main input
    urls = [entity['url'] for entity in deduplicated.values()]
    
    # Create the JSON structure
    entity_urls = {
        category: urls
    }
    
    # Save main input file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entity_urls, f, ensure_ascii=False, indent=2)
    
    # Also save detailed metadata for reference
    metadata_path = output_path.parent / 'entity_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(list(deduplicated.values()), f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Created {output_path}")
    print(f"✓ Created {metadata_path} (detailed metadata)")
    print(f"✓ Total unique URLs: {len(urls)}")


def generate_statistics(deduplicated: Dict[str, Dict[str, str]]) -> None:
    """Print statistics about the loaded data."""
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    # Count by source
    source_counts: Dict[str, int] = {}
    multi_source_count = 0
    
    for entity in deduplicated.values():
        sources = entity['sources']
        if len(sources) > 1:
            multi_source_count += 1
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\nEntities by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source:25s}: {count:6,d}")
    
    print(f"\nTotal unique URLs: {len(deduplicated):,}")
    print(f"Entities from multiple sources: {multi_source_count:,}")
    print("="*60 + "\n")


def main():
    """Main preprocessing workflow."""
    print("Hamburg CE Ecosystem - Entity Source Preprocessor")
    print("="*60 + "\n")
    
    # Determine base directory
    base_dir = Path(__file__).resolve().parent.parent
    print(f"Base directory: {base_dir}\n")
    
    # Load all sources
    all_entities: List[tuple[str, str, str]] = []
    
    all_entities.extend(load_crunchbase(base_dir))
    all_entities.extend(load_stakeholders_hamburg(base_dir))
    all_entities.extend(load_branchenbuch(base_dir))
    all_entities.extend(load_tuhh_institutes(base_dir))
    all_entities.extend(load_openalex_institutions(base_dir))
    
    print(f"\n✓ Total entities loaded (before deduplication): {len(all_entities):,}")
    
    # Deduplicate
    print("\nDeduplicating entities by URL...")
    deduplicated = deduplicate_entities(all_entities)
    
    # Generate statistics
    generate_statistics(deduplicated)
    
    # Create output files
    output_dir = base_dir / 'Approach 2 - Scrapegraph' / 'hamburg_ce_ecosystem' / 'data' / 'input'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'entity_urls.json'
    
    create_entity_urls_json(deduplicated, output_path)
    
    print("\n✅ Preprocessing complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the generated files in: {output_dir}")
    print(f"  2. Ensure Ollama is running: ollama serve")
    print(f"  3. Run the scraper: python run_large_scale_scraping.py")


if __name__ == '__main__':
    main()

