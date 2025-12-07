#!/usr/bin/env python3
"""
Retry script for entities that failed initial extraction.

This script:
1. Loads failed_entities_to_retry.csv (Hamburg + CE verified entities)
2. Retries extraction using improved prompts and validation
3. Tracks success/failure outcomes
4. Updates verification results with retry status
5. Saves detailed results for manual review
"""

import asyncio
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hamburg_ce_ecosystem.scrapers.batch_processor import BatchProcessor
from hamburg_ce_ecosystem.utils.data_manager import DataManager


class RetryProcessor:
    """Handles retry logic for failed extractions."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.recovery_dir = base_path / "data" / "recovery"
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

        # Initialize batch processor for extraction
        config_path = base_path / "config" / "scrape_config.yaml"
        self.batch_processor = BatchProcessor(config_path=config_path)
        self.data_manager = DataManager()

    def load_failed_entities(self) -> pd.DataFrame:
        """Load list of failed entities to retry."""
        failed_csv = self.recovery_dir / "failed_entities_to_retry.csv"

        if not failed_csv.exists():
            raise FileNotFoundError(
                f"Failed entities file not found: {failed_csv}\n"
                f"Run recover_failed_extractions.py first!"
            )

        df = pd.read_csv(failed_csv)
        print(f"Loaded {len(df)} failed entities to retry")
        return df

    async def retry_extractions(self, failed_df: pd.DataFrame, batch_size: int = 50):
        """Retry extractions for failed entities.

        Args:
            failed_df: DataFrame with failed entities
            batch_size: Number of entities to process at once
        """
        total = len(failed_df)
        urls = failed_df['url'].tolist()

        print(f"\n{'=' * 60}")
        print(f"Starting retry extraction for {total} entities")
        print(f"Batch size: {batch_size}")
        print(f"{'=' * 60}\n")

        # Track results
        retry_results = []
        start_time = datetime.now()

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_urls = urls[batch_start:batch_end]

            print(f"\n--- Batch {batch_start // batch_size + 1} "
                  f"(entities {batch_start + 1}-{batch_end}) ---")

            # Run extraction batch
            await self.batch_processor.process_extraction_batch(
                urls=batch_urls,
                max_concurrent=5  # Limit concurrency to avoid overwhelming
            )

            # Check which entities now have profiles
            for url in batch_urls:
                profile = self.data_manager.get_entity_profile(url)
                success = profile is not None

                # Check if it's a minimal profile (extraction_confidence = 0.0)
                is_minimal = False
                if success and profile.get('extraction_confidence', 1.0) == 0.0:
                    is_minimal = True
                    status = 'minimal_profile'
                elif success:
                    status = 'success'
                else:
                    status = 'still_failed'

                retry_results.append({
                    'url': url,
                    'status': status,
                    'is_minimal': is_minimal,
                    'retry_timestamp': datetime.now().isoformat()
                })

            print(f"Batch complete. Processed {batch_end}/{total} entities.")

        elapsed = datetime.now() - start_time
        print(f"\n{'=' * 60}")
        print(f"Retry extraction complete!")
        print(f"Total time: {elapsed}")
        print(f"{'=' * 60}\n")

        return retry_results

    def analyze_results(self, retry_results: List[dict], failed_df: pd.DataFrame):
        """Analyze and save retry results.

        Args:
            retry_results: List of retry outcome records
            failed_df: Original failed entities DataFrame
        """
        results_df = pd.DataFrame(retry_results)

        # Calculate statistics
        total = len(results_df)
        successful = len(results_df[results_df['status'] == 'success'])
        minimal = len(results_df[results_df['status'] == 'minimal_profile'])
        still_failed = len(results_df[results_df['status'] == 'still_failed'])

        print("\n=== Retry Results Summary ===")
        print(f"Total retried: {total}")
        print(f"Successful extractions: {successful} ({successful / total * 100:.1f}%)")
        print(f"Minimal profiles (empty fields): {minimal} ({minimal / total * 100:.1f}%)")
        print(f"Still failing: {still_failed} ({still_failed / total * 100:.1f}%)")
        print()

        # Save results
        results_file = self.recovery_dir / "retry_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Saved retry results to: {results_file}")

        # Save still-failing entities for further investigation
        still_failing = results_df[results_df['status'] == 'still_failed']
        if len(still_failing) > 0:
            # Merge with original data to preserve metadata
            still_failing_full = failed_df[failed_df['url'].isin(still_failing['url'])]
            still_failing_file = self.recovery_dir / "still_failing.csv"
            still_failing_full.to_csv(still_failing_file, index=False)
            print(f"Saved {len(still_failing)} still-failing entities to: {still_failing_file}")

        # Save successful recoveries
        successful_recoveries = results_df[results_df['status'] == 'success']
        if len(successful_recoveries) > 0:
            recoveries_file = self.recovery_dir / "successful_recoveries.csv"
            successful_recoveries.to_csv(recoveries_file, index=False)
            print(f"Saved {len(successful_recoveries)} successful recoveries to: {recoveries_file}")

        # Save comprehensive report
        report = {
            'retry_timestamp': datetime.now().isoformat(),
            'total_retried': total,
            'successful': successful,
            'minimal_profiles': minimal,
            'still_failing': still_failed,
            'success_rate': successful / total if total > 0 else 0,
            'recovery_rate': (successful + minimal) / total if total > 0 else 0,
            'results': retry_results
        }

        report_file = self.recovery_dir / "retry_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed report to: {report_file}")

        print("\n=== Next Steps ===")
        if still_failed > 0:
            print(f"1. Review still_failing.csv ({still_failed} entities)")
            print("2. Check partial_extractions.jsonl for error details")
            print("3. Consider manual data entry for critical entities")
        if minimal > 0:
            print(f"4. Review {minimal} minimal profiles in entity_profiles.json")
            print("   (search for extraction_confidence: 0.0)")
            print("5. These entities need manual enrichment")


async def main():
    """Main retry workflow."""
    base_path = Path(__file__).parent / "hamburg_ce_ecosystem"

    print("=" * 60)
    print("Failed Extraction Recovery - Retry Phase")
    print("=" * 60)
    print()

    # Initialize processor
    processor = RetryProcessor(base_path)

    # Load failed entities
    try:
        failed_df = processor.load_failed_entities()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Confirm retry
    print(f"\nAbout to retry extraction for {len(failed_df)} entities.")
    print("This will use improved prompts and validation logic.")
    print("Entities that still fail will be saved with empty fields.\n")

    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Retry cancelled.")
        return

    # Retry extractions
    retry_results = await processor.retry_extractions(failed_df, batch_size=50)

    # Analyze and save results
    processor.analyze_results(retry_results, failed_df)

    print("\n" + "=" * 60)
    print("Retry process complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
