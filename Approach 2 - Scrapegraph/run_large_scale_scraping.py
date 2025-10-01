#!/usr/bin/env python3
"""
Large-scale scraping launcher for Hamburg CE Ecosystem Mapper.
This script runs the full pipeline with optimized settings for processing
tens of thousands of entities.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            check=True,
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False


def check_required_files(base_dir: Path) -> bool:
    """Check if required input files exist."""
    input_file = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'input' / 'entity_urls.json'
    config_file = base_dir / 'hamburg_ce_ecosystem' / 'config' / 'scrape_config.yaml'
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Please run: python preprocess_entity_sources.py")
        return False
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    return True


def estimate_runtime(num_entities: int, max_workers: int = 5) -> str:
    """Estimate approximate runtime."""
    # Rough estimate: ~30 seconds per entity with verification + extraction
    # Parallel processing divides by workers
    avg_time_per_entity = 30  # seconds
    total_seconds = (num_entities * avg_time_per_entity) / max_workers
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"{hours}h {minutes}m"


def print_banner():
    """Print a fancy banner."""
    print("=" * 70)
    print(" " * 10 + "HAMBURG CE ECOSYSTEM - LARGE SCALE SCRAPER")
    print("=" * 70)
    print()


def print_configuration(max_workers: int, input_file: Path):
    """Print current configuration."""
    import json
    
    # Load entity count
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_entities = sum(len(urls) for urls in data.values())
    
    print("Configuration:")
    print(f"  - Input file: {input_file}")
    print(f"  - Total entities: {total_entities:,}")
    print(f"  - Max workers: {max_workers}")
    print(f"  - Estimated runtime: {estimate_runtime(total_entities, max_workers)}")
    print(f"  - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def run_pipeline(base_dir: Path, max_workers: int = 5) -> int:
    """Run the scraping pipeline."""
    from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor
    
    input_file = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'input' / 'entity_urls.json'
    config_path = base_dir / 'hamburg_ce_ecosystem' / 'config' / 'scrape_config.yaml'
    
    print("Starting pipeline...\n")
    
    try:
        processor = BatchProcessor(
            input_file=input_file,
            config_path=config_path,
            max_workers=max_workers
        )
        processor.run_pipeline()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("Results saved to:")
        print(f"  - CSV: {base_dir / 'hamburg_ce_ecosystem' / 'data' / 'final' / 'ecosystem_entities.csv'}")
        print(f"  - JSON: {base_dir / 'hamburg_ce_ecosystem' / 'data' / 'final' / 'ecosystem_map.json'}")
        print(f"  - Database: {base_dir / 'hamburg_ce_ecosystem' / 'data' / 'final' / 'ecosystem.db'}")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("   Progress has been saved. You can resume by running this script again.")
        return 130
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Hamburg CE Ecosystem large-scale scraper')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    print_banner()
    
    base_dir = Path(__file__).resolve().parent
    
    # Pre-flight checks
    print("Running pre-flight checks...")
    print()
    
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("   Please start it with: ollama serve")
        print("   Then run this script again.")
        return 1
    
    print("✓ Ollama is running")
    
    if not check_required_files(base_dir):
        return 1
    
    print("✓ Required files found")
    print()
    
    # Configuration
    max_workers = 20  # Optimized for 24-core system (adjust based on your needs)
    input_file = base_dir / 'hamburg_ce_ecosystem' / 'data' / 'input' / 'entity_urls.json'
    
    print_configuration(max_workers, input_file)
    
    # Confirmation prompt
    if not args.yes:
        print("⚠️  This will process a large number of entities and may take several hours.")
        try:
            response = input("Do you want to proceed? [y/N]: ").strip().lower()
        except EOFError:
            print("\nRunning in non-interactive mode. Use --yes flag to proceed automatically.")
            return 1
        
        if response not in ['y', 'yes']:
            print("\nAborted by user.")
            return 0
    else:
        print("✓ Auto-confirmed with --yes flag")
    
    print()
    
    # Run the pipeline
    return run_pipeline(base_dir, max_workers)


if __name__ == '__main__':
    sys.exit(main())

