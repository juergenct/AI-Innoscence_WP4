#!/usr/bin/env python3
"""
Convenience script to run the full pipeline:
1. Preprocess all input CSV files
2. Run the scraping pipeline
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_preprocessing() -> bool:
    """Run the preprocessing script."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING INPUT DATA")
    print("=" * 70)
    
    script_path = Path(__file__).resolve().parent / 'preprocess_input_data.py'
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Preprocessing failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to run preprocessing: {e}")
        return False


def run_scraper() -> bool:
    """Run the main scraping pipeline."""
    print("\n" + "=" * 70)
    print("STEP 2: RUNNING SCRAPER PIPELINE")
    print("=" * 70)
    
    # Check if Ollama is running
    try:
        subprocess.run(['ollama', 'list'], check=True, capture_output=True)
    except Exception:
        print("\nERROR: Ollama is not running!")
        print("Please start Ollama first with: ollama serve")
        return False
    
    print("✓ Ollama is running")
    
    # Import and run the main scraper
    try:
        from hamburg_ce_ecosystem.main import main
        main()
        return True
    except Exception as e:
        print(f"\nERROR: Scraper failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Run the full pipeline."""
    print("\n" + "=" * 70)
    print("HAMBURG CE ECOSYSTEM - FULL PIPELINE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Preprocess all CSV files from Input Data directory")
    print("  2. Run the scraping pipeline to verify Hamburg + CE relevance")
    print("  3. Extract detailed entity profiles")
    print("  4. Geocode addresses and create final outputs")
    print("=" * 70)
    
    # Step 1: Preprocessing
    if not run_preprocessing():
        print("\n❌ Pipeline failed at preprocessing stage")
        sys.exit(1)
    
    # Step 2: Scraping
    if not run_scraper():
        print("\n❌ Pipeline failed at scraping stage")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nResults are available in:")
    print("  - hamburg_ce_ecosystem/data/final/ecosystem_entities.csv")
    print("  - hamburg_ce_ecosystem/data/final/ecosystem_map.json")
    print("  - hamburg_ce_ecosystem/data/final/ecosystem.db")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

