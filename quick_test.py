#!/usr/bin/env python
"""
Quick 30-minute test for the Circular Economy Hamburg Scraper
Tests all major features with controlled scope
"""

import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def create_test_seeds():
    """Create a focused test seed file"""
    seeds = [
        "https://www.tuhh.de/crem/en/welcome-1",  # TUHH Circular Economy
        "https://www.stadtreinigung.hamburg",      # Hamburg Waste Management
        "https://www.veolia.de/standorte/hamburg", # Veolia Hamburg
        "https://www.haw-hamburg.de",              # HAW Hamburg
        "https://cirplus.com",                     # Circular Economy Platform
    ]
    
    # Write seeds to CSV
    test_file = Path("test_seeds_quick.csv")
    with open(test_file, 'w') as f:
        f.write("website\n")
        for url in seeds:
            f.write(f"{url}\n")
    
    return test_file

def run_test():
    """Run the test crawl"""
    print("=" * 60)
    print("CIRCULAR ECONOMY HAMBURG SCRAPER - 30 MINUTE TEST")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create test seeds
    seed_file = create_test_seeds()
    print(f"✅ Created test seeds: {seed_file}")
    print()
    
    # Test parameters
    params = {
        "seed": str(seed_file),
        "depth": 2,
        "iterate": 2,
        "iter_limit": 8,
        "spider": "static",
        "export": "excel",
        "classify": True,
        "llm_samples": 2,
        "llm_max_entities": 5
    }
    
    print("📋 Test Configuration:")
    print(f"  • Seed URLs: 5 Hamburg CE organizations")
    print(f"  • Max depth: {params['depth']} levels")
    print(f"  • Iterations: {params['iterate']} rounds")
    print(f"  • Entities per iteration: {params['iter_limit']}")
    print(f"  • LLM classification: Top {params['llm_max_entities']} entities")
    print()
    
    # Build command
    cmd = ["python", "run_scraper.py"]
    for key, value in params.items():
        if key == "classify" and value:
            cmd.append("--classify")
        elif key != "classify":
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print("🚀 Starting crawl...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    # Run the scraper
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr[:500])
    except Exception as e:
        print(f"Error running scraper: {e}")
        return
    
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"✅ Crawl completed in {elapsed/60:.1f} minutes")
    print()
    
    # Analyze results
    print("📊 RESULTS ANALYSIS")
    print("=" * 60)
    
    # Check entities discovered
    entities_file = Path("data/exports/entities_aggregated.csv")
    if entities_file.exists():
        df = pd.read_csv(entities_file)
        print(f"📍 Entities discovered: {len(df)}")
        print(f"📍 Hamburg-related: {df['hh_pages'].sum() if 'hh_pages' in df else 'N/A'}")
        print(f"📍 CE-related: {df['ce_pages'].sum() if 'ce_pages' in df else 'N/A'}")
        print()
        
        print("Top 5 entities by pages:")
        top_entities = df.nlargest(5, 'pages')[['entity_name', 'pages', 'domain']]
        for _, row in top_entities.iterrows():
            print(f"  • {row['entity_name']}: {row['pages']} pages ({row['domain']})")
        print()
    
    # Check link discovery
    all_links = list(Path("data/exports").glob("discovered_links_all_*.txt"))
    if all_links:
        latest = max(all_links, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            lines = [l for l in f if not l.startswith('#')]
            print(f"🔗 Total links discovered: {len(lines)}")
    
    iter_seeds = list(Path("data/exports").glob("iteration_seeds_*.csv"))
    if iter_seeds:
        latest = max(iter_seeds, key=lambda p: p.stat().st_mtime)
        df_seeds = pd.read_csv(latest)
        print(f"🔗 Filtered seeds (Hamburg+CE): {len(df_seeds)}")
    print()
    
    # Check LLM classification
    llm_file = Path("data/exports/entities_aggregated_with_llm.csv")
    if llm_file.exists():
        df_llm = pd.read_csv(llm_file)
        print("🤖 LLM Classification Results:")
        if 'role_llm' in df_llm:
            roles = df_llm['role_llm'].value_counts()
            for role, count in roles.items():
                print(f"  • {role}: {count}")
    print()
    
    # Check entity storage
    raw_entities = len(list(Path("data/raw").glob("*/entities/*")))
    proc_entities = len(list(Path("data/processed").glob("*/entities/*")))
    print(f"💾 Storage Structure:")
    print(f"  • Raw HTML entities: {raw_entities}")
    print(f"  • Processed text entities: {proc_entities}")
    print()
    
    # Summary statistics
    stats_files = list(Path("data/exports").glob("spider_stats_*.json"))
    if stats_files:
        latest = max(stats_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            stats = json.load(f)
            print("📈 Spider Statistics:")
            print(f"  • Total URLs visited: {stats.get('total_urls', 'N/A')}")
            print(f"  • Domains crawled: {len(stats.get('domain_counts', {}))}")
            if 'stats' in stats:
                print(f"  • Items scraped: {stats['stats'].get('items_scraped', 'N/A')}")
                print(f"  • Successful responses: {stats['stats'].get('successful_responses', 'N/A')}")
    
    print()
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print()
    print("✅ Test complete! Check data/exports/ for detailed results.")
    print("📁 Excel report: data/exports/circular_economy_data_*.xlsx")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
