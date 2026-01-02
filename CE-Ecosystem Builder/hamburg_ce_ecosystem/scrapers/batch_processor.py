from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import httpx
import pandas as pd
import yaml
from tqdm import tqdm

from hamburg_ce_ecosystem.models.schemas import (
    VerificationResultDict,
    EntityProfileDict,
    RelationshipDict,
    EcosystemInsightDict,
)
from hamburg_ce_ecosystem.scrapers.verification_scraper import VerificationScraper
from hamburg_ce_ecosystem.scrapers.extraction_scraper import ExtractionScraper
from hamburg_ce_ecosystem.scrapers.relationship_analyzer import RelationshipAnalyzer
from hamburg_ce_ecosystem.utils.logging_setup import setup_logging
from hamburg_ce_ecosystem.utils.geocoder import Geocoder
from hamburg_ce_ecosystem.utils.url_utils import extract_domain
from hamburg_ce_ecosystem.utils.rate_limiter import DomainRateLimiter
from hamburg_ce_ecosystem.utils.data_manager import DataManager
from hamburg_ce_ecosystem.utils.suppress_warnings import suppress_verbose_output

# Suppress verbose third-party library output
suppress_verbose_output()


def _json_safe(obj: Any) -> Any:
    try:
        import pydantic
        if isinstance(obj, pydantic.BaseModel):
            return obj.model_dump()
    except Exception:
        pass
    if hasattr(obj, 'value'):
        return getattr(obj, 'value')
    if hasattr(obj, '__str__') and not isinstance(obj, (dict, list)):
        return str(obj)
    return obj


class BatchProcessor:
    def __init__(self, input_file: str | Path, config_path: str | Path | None = None, max_workers: int = 8):
        self.input_file = Path(input_file)
        self.config_path = Path(config_path) if config_path else (self.input_file.parent.parent.parent / 'config' / 'scrape_config.yaml')
        self.config = self._load_config(self.config_path)
        # Only show ERROR and CRITICAL on console, log everything to file
        self.logger = setup_logging(self.input_file.parent.parent.parent / 'logs' / 'scraping_errors_hamburg.log', console_level=logging.ERROR)
        self.input_urls = self.load_input_urls(self.input_file)
        self.verification_scraper = VerificationScraper(self.config_path)
        self.extraction_scraper = ExtractionScraper(self.config_path)
        self.relationship_analyzer = RelationshipAnalyzer(self.config_path)
        self.rate_limiter = DomainRateLimiter(max_per_second=3.0)
        self.max_workers = max_workers
        self.data_dir = self.input_file.parent.parent.parent / 'data'
        self.dm = DataManager(self.data_dir / 'final' / 'ecosystem.db')

    @staticmethod
    def load_input_urls(file_path: Path) -> Dict[str, List[str]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def save_checkpoint(self, stage: str, total: int, completed: int, failed: int) -> None:
        """Save checkpoint file for progress tracking and resume visibility."""
        checkpoint_path = self.data_dir / 'checkpoint.json'
        checkpoint = {
            'stage': stage,
            'total_entities': total,
            'completed': completed,
            'failed': failed,
            'remaining': total - completed - failed,
            'last_updated': datetime.now().isoformat(),
            'progress_percentage': round((completed / total * 100), 2) if total > 0 else 0
        }
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding='utf-8')

    def _verify_single(self, category: str, url: str) -> VerificationResultDict | None:
        try:
            self.rate_limiter.acquire(extract_domain(url))
            res = self.verification_scraper.verify_entity(url)
            d = res.model_dump()
            d['input_category'] = category
            d['url'] = str(url)  # Ensure URL is included
            return d  # type: ignore[return-value]
        except Exception as e:
            # Log but continue - skip entities where LLM completely fails
            self.logger.info(f"Skipping {url} - could not verify: {e}")
            return None

    async def _verify_single_async(self, category: str, url: str, semaphore: asyncio.Semaphore) -> VerificationResultDict | None:
        """Async version of _verify_single for concurrent processing."""
        async with semaphore:  # Limit concurrent requests
            try:
                self.rate_limiter.acquire(extract_domain(url))
                # Note: verification_scraper.verify_entity is still sync (uses ScrapegraphAI)
                # Run in executor to not block event loop
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None,
                    self.verification_scraper.verify_entity,
                    url
                )
                # Add delay to prevent overwhelming Ollama
                await asyncio.sleep(1.0)  # 1 second delay between Ollama calls
                d = res.model_dump()
                d['input_category'] = category
                d['url'] = str(url)
                return d  # type: ignore[return-value]
            except Exception as e:
                self.logger.info(f"Skipping {url} - could not verify: {e}")
                return None

    def run_verification(self) -> List[VerificationResultDict]:
        """Run verification with async I/O for 3-5x speedup."""
        all_tasks: List[tuple[str, str]] = []
        for category, urls in self.input_urls.items():
            for url in urls:
                all_tasks.append((category, url))

        results: List[VerificationResultDict] = []
        save_path = self.data_dir / 'verified' / 'verification_results.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Run async event loop
        results = asyncio.run(self._run_verification_async(all_tasks, save_path))

        pd.DataFrame(results).to_csv(save_path, index=False)
        try:
            self.dm.save_verification(results)
        except Exception as e:
            self.logger.warning(f"DB save_verification failed: {e}")
        return results

    async def _run_verification_async(
        self,
        all_tasks: List[tuple[str, str]],
        save_path: Path
    ) -> List[VerificationResultDict]:
        """Async implementation of verification for concurrent processing."""
        # Semaphore to limit concurrent requests (avoid overwhelming Ollama with large 32B model)
        semaphore = asyncio.Semaphore(max(2, self.max_workers // 2))  # Conservative limit for local Ollama

        results: List[VerificationResultDict] = []
        skipped = 0

        # Create all tasks
        tasks = [
            self._verify_single_async(category, url, semaphore)
            for category, url in all_tasks
        ]

        # Process with progress bar
        with tqdm(total=len(all_tasks), desc="Verification", unit="entity") as pbar:
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                res = await coro
                if res is not None:
                    results.append(res)
                else:
                    skipped += 1
                pbar.set_postfix({"✓ LLM": len(results), "⊘ Skipped": skipped})
                pbar.update(1)

                # Periodic save
                if idx % 10 == 0:
                    pd.DataFrame(results).to_csv(save_path, index=False)
                    try:
                        self.dm.save_verification(results[-10:])
                    except Exception as e:
                        self.logger.warning(f"DB save_verification partial failed: {e}")

        return results

    def _extract_single(self, url: str, input_category: str) -> EntityProfileDict | None:
        try:
            source_url = str(url)
            self.rate_limiter.acquire(extract_domain(source_url))
            profile = self.extraction_scraper.extract_entity_info(source_url)
            d = profile.model_dump()
            d['input_category'] = input_category
            return d  # type: ignore[return-value]
        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {e}")
            return None

    async def _extract_single_async(self, url: str, input_category: str, semaphore: asyncio.Semaphore) -> EntityProfileDict | None:
        """Async version of _extract_single for concurrent processing."""
        async with semaphore:
            try:
                source_url = str(url)
                self.rate_limiter.acquire(extract_domain(source_url))
                # extract_entity_info is now async with parallel Instructor calls
                profile = await self.extraction_scraper.extract_entity_info(source_url)
                # Add delay to prevent overwhelming Ollama (though parallel extraction has internal delays)
                await asyncio.sleep(1.5)  # 1.5 second delay between entity extractions
                d = profile.model_dump()
                d['input_category'] = input_category
                return d  # type: ignore[return-value]
            except Exception as e:
                self.logger.error(f"Extraction failed for {url}: {e}")
                return None

    def run_extraction(self, verified_entities: List[VerificationResultDict]) -> List[EntityProfileDict]:
        """Run extraction with async I/O for 3-5x speedup. Supports resume from checkpoint."""
        # LLM-gated extraction: only extract where should_extract is True
        to_extract = [e for e in verified_entities if e.get('should_extract')]
        save_path = self.data_dir / 'extracted' / 'entity_profiles.json'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # CHECKPOINT/RESUME LOGIC: Skip already-extracted entities
        print(f"\n🔍 Checking for already-extracted entities...")
        already_extracted_db = self.dm.get_extracted_urls()

        # Get URLs we need to extract
        urls_to_extract = {e['url'] for e in to_extract}

        # Calculate ACTUAL overlap (URLs that are both in to_extract AND already done)
        db_overlap = urls_to_extract & already_extracted_db

        # Also check cache for additional completed extractions
        cache_overlap = set()
        for entity in to_extract:
            cache_key = f"extraction::{entity['url']}"
            if self.extraction_scraper.cache.get(cache_key):
                cache_overlap.add(entity['url'])

        # Combine overlaps (URLs we can skip)
        all_skippable = db_overlap | cache_overlap

        # Validate JSON file integrity if we have DB matches
        if save_path.exists() and len(db_overlap) > 0:
            try:
                existing_json = json.loads(save_path.read_text(encoding='utf-8'))
                json_count = len(existing_json)
                db_count = len(db_overlap)

                if json_count < db_count * 0.95:  # More than 5% discrepancy
                    print(f"⚠️  JSON/Database mismatch detected:")
                    print(f"   - Database (matching): {db_count} entities")
                    print(f"   - JSON file: {json_count} entities")
                    print(f"🔧 Rebuilding entity_profiles.json from database...")

                    # Rebuild JSON from database
                    self._rebuild_json_from_database(save_path)
                    print(f"✅ Successfully rebuilt entity_profiles.json")
            except Exception as e:
                self.logger.warning(f"Failed to validate JSON integrity: {e}")

        # Filter to only process URLs not yet extracted
        original_count = len(to_extract)
        to_extract = [e for e in to_extract if e['url'] not in all_skippable]

        # Report accurate numbers
        actual_skipped = original_count - len(to_extract)
        print(f"✅ Resume checkpoint: {actual_skipped} already extracted (skipping)")
        print(f"   - {len(db_overlap)} found in database")
        print(f"   - {len(cache_overlap - db_overlap)} found in cache only")
        if len(already_extracted_db) != len(db_overlap):
            print(f"   - (Note: database has {len(already_extracted_db)} total, but only {len(db_overlap)} match current input)")
        print(f"📊 Extracting {len(to_extract)} remaining entities (of {original_count} total)")

        if len(to_extract) == 0:
            print("✨ All entities already extracted! Loading existing results from file...")
            # Load existing results from JSON file
            if save_path.exists():
                try:
                    existing_data = json.loads(save_path.read_text(encoding='utf-8'))
                    print(f"   Loaded {len(existing_data)} existing entity profiles")
                    return existing_data  # type: ignore[return-value]
                except Exception as e:
                    self.logger.warning(f"Failed to load existing results: {e}")
                    return []
            else:
                print("   Warning: No existing results file found")
                return []

        # Run async event loop for remaining entities
        new_results = asyncio.run(
            self._run_extraction_async(
                to_extract,
                save_path,
                already_completed=actual_skipped,
                overall_total=original_count
            )
        )

        # Combine with existing results before saving
        all_results = new_results
        if save_path.exists() and len(db_overlap) > 0:
            try:
                existing_data = json.loads(save_path.read_text(encoding='utf-8'))
                # Combine existing + new results
                all_results = existing_data + new_results
                print(f"   Combined {len(existing_data)} existing + {len(new_results)} new = {len(all_results)} total profiles")
            except Exception as e:
                self.logger.warning(f"Failed to load existing results for combining: {e}")
                all_results = new_results

        # Save combined results to JSON
        save_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, default=_json_safe), encoding='utf-8')

        # Save only new results to database (existing ones already there)
        try:
            self.dm.save_profiles(new_results)
        except Exception as e:
            self.logger.warning(f"DB save_profiles failed: {e}")

        # Return combined results for next pipeline stages
        return all_results

    async def _run_extraction_async(
        self,
        to_extract: List[VerificationResultDict],
        save_path: Path,
        already_completed: int = 0,
        overall_total: int = None
    ) -> List[EntityProfileDict]:
        """Async implementation of extraction for concurrent processing."""
        # Conservative limit for local Ollama with large 32B model
        semaphore = asyncio.Semaphore(max(2, self.max_workers // 2))

        results: List[EntityProfileDict] = []
        total_count = len(to_extract)
        # Track overall progress if provided
        overall_total = overall_total or total_count

        # Create all tasks
        tasks = [
            self._extract_single_async(e['url'], e.get('input_category', ''), semaphore)
            for e in to_extract
        ]

        # Process with progress bar
        with tqdm(total=total_count, desc="Extraction", unit="entity") as pbar:
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                res = await coro
                if res is not None:
                    results.append(res)
                failed_count = idx - len(results)
                pbar.set_postfix({"✓": len(results), "✗": failed_count})
                pbar.update(1)

                # Periodic save + checkpoint
                if idx % 10 == 0:
                    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_safe), encoding='utf-8')
                    try:
                        self.dm.save_profiles(results[-10:])
                    except Exception as e:
                        self.logger.warning(f"DB save_profiles partial failed: {e}")

                    # Save checkpoint for resume visibility (overall progress)
                    try:
                        overall_completed = already_completed + len(results)
                        overall_failed = already_completed + failed_count  # Approximate
                        self.save_checkpoint(
                            stage='extraction',
                            total=overall_total,
                            completed=overall_completed,
                            failed=idx - len(results)  # Failed in current run only
                        )
                    except Exception as e:
                        self.logger.warning(f"Checkpoint save failed: {e}")

        # Final checkpoint save (overall progress)
        try:
            overall_completed = already_completed + len(results)
            overall_failed = (overall_total - overall_completed)
            self.save_checkpoint(
                stage='extraction',
                total=overall_total,
                completed=overall_completed,
                failed=total_count - len(results)  # Failed in current run
            )
        except Exception as e:
            self.logger.warning(f"Final checkpoint save failed: {e}")

        return results

    def geocode_entities(self, profiles: List[EntityProfileDict]) -> List[EntityProfileDict]:
        """
        Geocode entities using robust geocoder with fallback strategies.

        IMPROVED: Now includes resume logic and database persistence.
        - Checks database for already-geocoded entities
        - Saves coordinates to database immediately after each entity
        - Updates entity_profiles.json file after completion
        - Supports resume from interruption
        """
        cache_file = self.data_dir / 'final' / 'geocode_cache.csv'
        geocoder = Geocoder(
            cache_file=cache_file,
            timeout=15,
            max_retries=3
        )

        # RESUME LOGIC: Check database for already-geocoded entities
        print(f"\n🔍 Checking for already-geocoded entities...")
        already_geocoded_urls = self.dm.get_geocoded_urls()

        if len(already_geocoded_urls) > 0:
            print(f"✅ Resume checkpoint: {len(already_geocoded_urls)} already geocoded")

            # Update profiles with coordinates from database
            db_profiles = {p['url']: p for p in self.dm.get_all_entity_profiles() if p.get('latitude')}
            for p in profiles:
                if p['url'] in already_geocoded_urls and p['url'] in db_profiles:
                    db_profile = db_profiles[p['url']]
                    p['latitude'] = db_profile.get('latitude')
                    p['longitude'] = db_profile.get('longitude')

        # Define Hamburg city center coordinates (default fallback)
        HAMBURG_CENTER_LAT = 53.550341
        HAMBURG_CENTER_LON = 10.000654

        # Count entities still needing geocoding
        # Entities need geocoding if:
        # 1. They have no coordinates (None), OR
        # 2. They have Hamburg center coordinates (will retry with name/URL fallback)
        to_geocode = [
            p for p in profiles
            if (p.get('latitude') is None or p.get('longitude') is None) or
               (abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
                abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001)
        ]

        print(f"🌍 Geocoding {len(to_geocode)} remaining entities (of {len(profiles)} total)")
        if len(to_geocode) == 0:
            print("✨ All entities already geocoded!")
            return profiles

        geocoded_count = 0
        failed_count = 0
        batch_save_count = 0

        for idx, p in enumerate(tqdm(profiles, desc="Geocoding", unit="entity"), 1):
            # Check if entity needs geocoding (None coords OR Hamburg center)
            needs_geocoding = (
                (p.get('latitude') is None or p.get('longitude') is None) or
                (abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
                 abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001)
            )

            if needs_geocoding:
                # Use improved geocoder with fallback to entity name and URL domain
                lat, lon = geocoder.geocode_hamburg(
                    address=p.get('address'),
                    entity_name=p.get('entity_name'),
                    url=p.get('url'),
                    use_fallback=True
                )

                if lat is not None and lon is not None:
                    p['latitude'] = lat
                    p['longitude'] = lon
                    geocoded_count += 1

                    # DATABASE PERSISTENCE: Save coordinates immediately
                    try:
                        self.dm.update_coordinates(p['url'], lat, lon)
                        batch_save_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to save coordinates for {p['url']}: {e}")
                else:
                    failed_count += 1

                # Progress logging every 100 entities
                if idx % 100 == 0:
                    print(f"   Progress: {geocoded_count} geocoded, {failed_count} failed, {batch_save_count} saved to DB")

        print(f"✅ Geocoding complete: {geocoded_count}/{len(to_geocode)} successfully geocoded")
        print(f"   - {batch_save_count} coordinates saved to database")
        print(f"   - {failed_count} entities could not be geocoded")

        # VALIDATION: Report on entities still using Hamburg center coordinates
        hamburg_center_entities = [
            p for p in profiles
            if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
               abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001
        ]

        hamburg_with_address = [
            p for p in hamburg_center_entities
            if p.get('address') and
               str(p.get('address')).strip() and
               str(p.get('address')).strip().lower() not in ['unknown', 'na', 'n/a', 'none']
        ]

        hamburg_without_address = [
            p for p in hamburg_center_entities
            if not p.get('address') or
               not str(p.get('address')).strip() or
               str(p.get('address')).strip().lower() in ['unknown', 'na', 'n/a', 'none']
        ]

        print(f"\n📍 Hamburg Center Coordinate Analysis:")
        print(f"   - Total entities with Hamburg center coords: {len(hamburg_center_entities)}")
        print(f"   - With valid addresses: {len(hamburg_with_address)}")
        print(f"   - Without addresses (will use company name): {len(hamburg_without_address)}")

        # STAGE 2: Google Maps Fallback (if enabled and needed)
        if hamburg_center_entities:
            print(f"\n⚠️  Found {len(hamburg_center_entities)} entities with Hamburg center coordinates")
            print(f"   - Will retry with Google Maps using address OR company name")

            # Check if Google Maps fallback is enabled in config
            google_api_key = self.config.get('geocoding', {}).get('google_maps_api_key')
            enable_google_fallback = self.config.get('geocoding', {}).get('enable_google_maps_fallback', True)

            if enable_google_fallback and google_api_key:
                print(f"\n{'='*80}")
                print(f"🗺️  STAGE 2: Google Maps Fallback Geocoding")
                print(f"{'='*80}")
                print(f"💰 Estimated cost: ${len(hamburg_center_entities) * 0.005:.2f} USD")

                # Initialize Google Maps geocoder
                from hamburg_ce_ecosystem.utils.google_maps_geocoder import GoogleMapsGeocoder

                cache_file = self.data_dir / 'final' / 'google_maps_cache.csv'
                google_geocoder = GoogleMapsGeocoder(
                    api_key=google_api_key,
                    cache_file=cache_file,
                    timeout=10,
                    max_retries=3
                )

                google_geocoded_count = 0
                google_failed_count = 0
                used_address_count = 0
                used_company_name_count = 0

                for p in tqdm(hamburg_center_entities, desc="Google Maps Geocoding", unit="entity"):
                    # Determine what to use for geocoding query
                    address = p.get('address')
                    has_valid_address = (
                        address and
                        str(address).strip() and
                        str(address).strip().lower() not in ['unknown', 'na', 'n/a', 'none']
                    )

                    if has_valid_address:
                        # Use address if available
                        query = str(address).strip()
                        used_address_count += 1
                    else:
                        # Fallback to company name (geocode_hamburg will add ", Hamburg, Germany")
                        entity_name = p.get('entity_name', '')
                        if entity_name:
                            query = entity_name  # Don't add Hamburg here - geocode_hamburg adds it
                            used_company_name_count += 1
                        else:
                            # Skip if no name available
                            google_failed_count += 1
                            continue

                    lat, lon = google_geocoder.geocode_hamburg(query)

                    if lat is not None and lon is not None:
                        # Verify it's not the Hamburg center
                        if not (abs(lat - HAMBURG_CENTER_LAT) < 0.000001 and abs(lon - HAMBURG_CENTER_LON) < 0.000001):
                            p['latitude'] = lat
                            p['longitude'] = lon
                            google_geocoded_count += 1

                            # Save to database immediately
                            try:
                                self.dm.update_coordinates(p['url'], lat, lon)
                            except Exception as e:
                                self.logger.warning(f"Failed to save Google Maps coordinates for {p['url']}: {e}")
                        else:
                            google_failed_count += 1
                    else:
                        google_failed_count += 1

                print(f"\n✅ Google Maps geocoding complete:")
                print(f"   - {google_geocoded_count}/{len(hamburg_center_entities)} successfully geocoded")
                print(f"   - {used_address_count} queries used address")
                print(f"   - {used_company_name_count} queries used company name + Hamburg")
                print(f"   - {google_failed_count} entities could not be geocoded")
                print(f"   - Actual cost: ${google_geocoded_count * 0.005:.2f} USD")

                # Update overall counts
                geocoded_count += google_geocoded_count
                failed_count = failed_count - google_geocoded_count + google_failed_count
            else:
                if not enable_google_fallback:
                    print(f"   ℹ️  Google Maps fallback is disabled in config")
                elif not google_api_key:
                    print(f"   ℹ️  Google Maps API key not configured")
                    print(f"   Add 'geocoding.google_maps_api_key' to scrape_config.yaml to enable fallback")

        # UPDATE JSON FILE: Save updated profiles to entity_profiles.json
        profiles_path = self.data_dir / 'extracted' / 'entity_profiles.json'
        try:
            print(f"💾 Updating entity_profiles.json with geocoded coordinates...")
            profiles_path.write_text(
                json.dumps(profiles, ensure_ascii=False, indent=2, default=_json_safe),
                encoding='utf-8'
            )
            print(f"✅ Successfully updated {profiles_path}")
        except Exception as e:
            self.logger.error(f"Failed to update entity_profiles.json: {e}")

        # Save checkpoint
        try:
            self.save_checkpoint(
                stage='geocoding',
                total=len(profiles),
                completed=len(profiles) - failed_count,
                failed=failed_count
            )
        except Exception as e:
            self.logger.warning(f"Checkpoint save failed: {e}")

        return profiles

    def geocode_entities_google_maps(
        self,
        profiles: List[EntityProfileDict],
        api_key: str
    ) -> List[EntityProfileDict]:
        """
        Fallback geocoding using Google Maps API for entities that still have
        Hamburg center coordinates despite having valid addresses.

        This is a precision geocoding step that runs after Nominatim geocoding
        to ensure all entities with addresses have specific coordinates.

        Args:
            profiles: List of entity profiles
            api_key: Google Maps API key

        Returns:
            Updated list of profiles with improved geocoding
        """
        from hamburg_ce_ecosystem.utils.google_maps_geocoder import GoogleMapsGeocoder

        print(f"\n{'='*80}")
        print(f"🗺️  GOOGLE MAPS FALLBACK GEOCODING")
        print(f"{'='*80}")

        # Define Hamburg city center coordinates
        HAMBURG_CENTER_LAT = 53.550341
        HAMBURG_CENTER_LON = 10.000654

        # Identify entities that need Google Maps geocoding
        # (Hamburg center coords + valid address)
        to_geocode = [
            p for p in profiles
            if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
               abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001 and
               p.get('address') and
               str(p.get('address')).strip() and
               str(p.get('address')).strip().lower() not in ['unknown', 'na', 'n/a', 'none']
        ]

        if not to_geocode:
            print("✨ No entities need Google Maps geocoding - all addresses already geocoded!")
            return profiles

        print(f"📍 Found {len(to_geocode)} entities with Hamburg center coords but valid addresses")
        print(f"💰 Estimated cost: ${len(to_geocode) * 0.005:.2f} USD")

        # Initialize Google Maps geocoder
        cache_file = self.data_dir / 'final' / 'google_maps_cache.csv'
        geocoder = GoogleMapsGeocoder(
            api_key=api_key,
            cache_file=cache_file,
            timeout=10,
            max_retries=3
        )

        geocoded_count = 0
        failed_count = 0
        batch_save_count = 0

        from tqdm import tqdm

        def fix_utf8_encoding(text: str) -> str:
            """Fix double-encoded UTF-8 text (common with German umlauts)."""
            if not text or not isinstance(text, str):
                return text
            try:
                # Try to detect and fix UTF-8 double-encoding
                # Example: "PoppenbÃ¼ttler" (wrong) -> "Poppenbüttler" (correct)
                return text.encode('latin1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
                # Not double-encoded or can't fix - return original
                return text

        for idx, p in enumerate(tqdm(to_geocode, desc="Google Maps Geocoding", unit="entity"), 1):
            # Fix potential UTF-8 double-encoding in address
            address = p.get('address')
            if address:
                address = fix_utf8_encoding(str(address))

            # Geocode using Google Maps
            lat, lon = geocoder.geocode_hamburg(address)

            if lat is not None and lon is not None:
                # Verify it's not the Hamburg center (Google also might return center for bad addresses)
                if not (abs(lat - HAMBURG_CENTER_LAT) < 0.000001 and abs(lon - HAMBURG_CENTER_LON) < 0.000001):
                    p['latitude'] = lat
                    p['longitude'] = lon
                    geocoded_count += 1

                    # Save to database immediately
                    try:
                        self.dm.update_coordinates(p['url'], lat, lon)
                        batch_save_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to save coordinates for {p['url']}: {e}")
                else:
                    self.logger.debug(f"Google Maps returned Hamburg center for: {p.get('entity_name')}")
                    failed_count += 1
            else:
                failed_count += 1

            # Progress logging every 50 entities
            if idx % 50 == 0:
                print(f"   Progress: {geocoded_count} geocoded, {failed_count} failed, {batch_save_count} saved to DB")

        print(f"\n✅ Google Maps geocoding complete:")
        print(f"   - {geocoded_count}/{len(to_geocode)} successfully geocoded")
        print(f"   - {batch_save_count} coordinates saved to database")
        print(f"   - {failed_count} entities could not be geocoded")
        print(f"   - Actual cost: ${geocoded_count * 0.005:.2f} USD")

        # Get cache stats
        stats = geocoder.get_stats()
        print(f"   - Cache size: {stats['cache_size']} entries")

        # Update JSON file
        profiles_path = self.data_dir / 'extracted' / 'entity_profiles.json'
        try:
            print(f"💾 Updating entity_profiles.json with Google Maps coordinates...")
            profiles_path.write_text(
                json.dumps(profiles, ensure_ascii=False, indent=2, default=_json_safe),
                encoding='utf-8'
            )
            print(f"✅ Successfully updated {profiles_path}")
        except Exception as e:
            self.logger.error(f"Failed to update entity_profiles.json: {e}")

        # Final validation
        still_hamburg_center = [
            p for p in profiles
            if abs(p.get('latitude', 0) - HAMBURG_CENTER_LAT) < 0.000001 and
               abs(p.get('longitude', 0) - HAMBURG_CENTER_LON) < 0.000001 and
               p.get('address') and
               str(p.get('address')).strip() and
               str(p.get('address')).strip().lower() not in ['unknown', 'na', 'n/a', 'none']
        ]

        if still_hamburg_center:
            print(f"\n⚠️  Warning: {len(still_hamburg_center)} entities still have Hamburg center coords despite having addresses:")
            for p in still_hamburg_center[:5]:  # Show first 5
                print(f"   - {p.get('entity_name')}: {p.get('address')}")
            if len(still_hamburg_center) > 5:
                print(f"   ... and {len(still_hamburg_center) - 5} more")

        return profiles

    def run_relationship_analysis(
        self, profiles: List[EntityProfileDict]
    ) -> tuple[List[RelationshipDict], List[EcosystemInsightDict]]:
        """Run relationship and ecosystem analysis on extracted entities."""
        relationships, clusters, insights = self.relationship_analyzer.analyze_all_relationships(profiles)

        # Save to database
        try:
            self.dm.save_relationships(relationships)
            self.dm.save_clusters(clusters)
            self.dm.save_insights(insights)
        except Exception as e:
            self.logger.warning(f"DB save for relationships/insights failed: {e}")

        # Save to JSON files
        rel_path = self.data_dir / 'relationships' / 'relationships.json'
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        rel_path.write_text(
            json.dumps(relationships, ensure_ascii=False, indent=2, default=_json_safe),
            encoding='utf-8'
        )

        clusters_path = self.data_dir / 'relationships' / 'clusters.json'
        clusters_path.write_text(
            json.dumps(clusters, ensure_ascii=False, indent=2, default=_json_safe),
            encoding='utf-8'
        )

        insights_path = self.data_dir / 'relationships' / 'ecosystem_insights.json'
        insights_path.write_text(
            json.dumps(insights, ensure_ascii=False, indent=2, default=_json_safe),
            encoding='utf-8'
        )

        return relationships, insights

    def build_ecosystem_graph(
        self,
        profiles: List[EntityProfileDict],
        relationships: List[RelationshipDict] = None,
        insights: List[EcosystemInsightDict] = None,
    ) -> Dict[str, Any]:
        """
        Build enhanced ecosystem graph with relationships and insights.

        IMPROVED: Now includes data validation to detect incomplete datasets.
        """
        # DATA VALIDATION: Check if processing all entities from database
        try:
            db_entity_count = self.dm.get_entity_count()
            db_geocoded_count = len(self.dm.get_geocoded_urls())

            print(f"\n📊 Stage 5 Data Validation:")
            print(f"   - Database has {db_entity_count} total entities")
            print(f"   - Database has {db_geocoded_count} geocoded entities")
            print(f"   - Processing {len(profiles)} entities for graph construction")

            # Warn if processing significantly fewer entities than exist in database
            if db_entity_count > 0 and len(profiles) < db_entity_count * 0.95:
                coverage_pct = (len(profiles) / db_entity_count) * 100
                self.logger.warning(
                    f"⚠️  INCOMPLETE DATASET: Processing only {len(profiles)}/{db_entity_count} "
                    f"entities ({coverage_pct:.1f}% coverage)"
                )
                print(f"   ⚠️  WARNING: Only {coverage_pct:.1f}% of database entities are being processed!")
                print(f"   ⚠️  Missing {db_entity_count - len(profiles)} entities from final outputs")

            # Check geocoding coverage
            missing_coords = sum(1 for p in profiles if not p.get('latitude') or not p.get('longitude'))
            if missing_coords > 0:
                geocoded_pct = ((len(profiles) - missing_coords) / len(profiles)) * 100 if len(profiles) > 0 else 0
                self.logger.warning(
                    f"⚠️  MISSING COORDINATES: {missing_coords}/{len(profiles)} entities lack geocoded coordinates ({geocoded_pct:.1f}% coverage)"
                )
                print(f"   ⚠️  WARNING: {missing_coords} entities missing geocoded coordinates")

            # Log success if dataset is complete
            if len(profiles) >= db_entity_count * 0.95 and missing_coords == 0:
                print(f"   ✅ Dataset is complete: {len(profiles)}/{db_entity_count} entities with full geocoding")

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")

        graph: Dict[str, Any] = {"nodes": [], "edges": [], "insights": {}}

        # Build nodes
        for p in profiles:
            graph["nodes"].append({
                "id": p.get('entity_name'),
                "url": p.get('url'),
                "role": p.get('ecosystem_role'),
                "lat": p.get('latitude'),
                "lon": p.get('longitude'),
                "ce_activities": p.get('ce_activities', []),
                "description": p.get('brief_description', ''),
            })

        # Build edges from relationships
        if relationships:
            for rel in relationships:
                graph["edges"].append({
                    "source": rel.get('source_entity'),
                    "target": rel.get('target_entity'),
                    "type": rel.get('relationship_type'),
                    "confidence": rel.get('confidence'),
                    "evidence": rel.get('evidence'),
                    "bidirectional": rel.get('bidirectional', False),
                })

        # Add insights
        if insights:
            # Group insights by type
            insights_by_type: Dict[str, List[Dict[str, Any]]] = {
                "synergies": [],
                "gaps": [],
                "recommendations": [],
            }

            for insight in insights:
                insight_type = insight.get('insight_type', '')
                if insight_type == 'synergy':
                    insights_by_type['synergies'].append({
                        'title': insight.get('title'),
                        'description': insight.get('description'),
                        'entities': insight.get('entities_involved', []),
                        'confidence': insight.get('confidence'),
                        'priority': insight.get('priority'),
                    })
                elif insight_type == 'gap':
                    insights_by_type['gaps'].append({
                        'title': insight.get('title'),
                        'description': insight.get('description'),
                        'confidence': insight.get('confidence'),
                        'priority': insight.get('priority'),
                    })
                elif insight_type == 'recommendation':
                    insights_by_type['recommendations'].append({
                        'title': insight.get('title'),
                        'description': insight.get('description'),
                        'confidence': insight.get('confidence'),
                        'priority': insight.get('priority'),
                    })

            graph['insights'] = insights_by_type

        return graph

    def _rebuild_json_from_database(self, output_path: Path) -> None:
        """
        Rebuild entity_profiles.json from the database when mismatch detected.

        Args:
            output_path: Path to the entity_profiles.json file to rebuild
        """
        import sqlite3

        db_path = self.data_dir / 'final' / 'ecosystem.db'
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Fetch all entity profiles
        cur.execute("SELECT * FROM entity_profiles;")
        rows = cur.fetchall()

        # Convert to list of dicts
        print(f"📊 Fetched {len(rows)} entities from database")

        entities = []
        parse_errors = 0

        for row in rows:
            entity = dict(row)

            # Parse JSON fields
            json_fields = [
                'contact_persons', 'emails', 'phone_numbers', 'ce_activities',
                'capabilities_offered', 'needs_requirements', 'capability_categories',
                'partners', 'partner_urls', 'ce_activities_structured',
                'ce_capabilities_offered', 'ce_needs_requirements',
                'mentioned_partners', 'discovered_entities'
            ]

            for field in json_fields:
                if entity.get(field):
                    try:
                        entity[field] = json.loads(entity[field])
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"Failed to parse {field} for {entity.get('url', 'unknown')}: {e}")
                        entity[field] = []
                        parse_errors += 1
                else:
                    entity[field] = []

            # Remove database-specific fields
            entity.pop('id', None)
            entities.append(entity)

        conn.close()

        # VALIDATION: Check for data loss
        print(f"📊 Processed {len(entities)} entities (fetched {len(rows)})")
        if len(entities) != len(rows):
            self.logger.error(f"❌ DATA LOSS: Lost {len(rows) - len(entities)} entities during rebuild!")
            raise RuntimeError(f"JSON rebuild incomplete: {len(rows)} fetched, {len(entities)} processed")

        if parse_errors > 0:
            self.logger.warning(f"⚠️  Encountered {parse_errors} JSON parse errors during rebuild")
            print(f"   ⚠️  {parse_errors} JSON field parsing errors (see logs)")

        # Backup existing file
        if output_path.exists():
            backup_path = output_path.with_suffix('.json.backup')
            output_path.rename(backup_path)
            print(f"📦 Backed up existing file to: {backup_path}")

        # Save rebuilt JSON
        output_path.write_text(
            json.dumps(entities, ensure_ascii=False, indent=2, default=_json_safe),
            encoding='utf-8'
        )

        print(f"✅ Successfully rebuilt JSON with {len(entities)} entities")

    def save_results(
        self,
        profiles: List[EntityProfileDict],
        graph: Dict[str, Any],
        relationships: List[RelationshipDict] = None,
    ) -> None:
        """Save all results to final output directory."""
        out_dir = self.data_dir / 'final'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save ecosystem map (graph with relationships and insights)
        (out_dir / 'ecosystem_map.json').write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, default=_json_safe),
            encoding='utf-8'
        )

        # Save entity profiles
        df = pd.DataFrame(profiles)
        df.to_csv(out_dir / 'ecosystem_entities.csv', index=False)

        # Save relationships as CSV for easy analysis
        if relationships:
            rel_df = pd.DataFrame(relationships)
            rel_df.to_csv(out_dir / 'ecosystem_relationships.csv', index=False)

    def run_pipeline(self) -> None:
        """
        Run the complete pipeline from verification to final graph construction.

        IMPROVED: Now includes comprehensive logging and entity count tracking at each stage.
        """
        print("=" * 80)
        print("Hamburg CE Ecosystem Builder - Full Pipeline")
        print("=" * 80)

        # Log database state at start
        try:
            db_count = self.dm.get_entity_count()
            db_geocoded = len(self.dm.get_geocoded_urls())
            print(f"\n📊 Database State at Start:")
            print(f"   - Total entities: {db_count}")
            print(f"   - Geocoded entities: {db_geocoded}")
        except Exception as e:
            self.logger.warning(f"Could not fetch database state: {e}")

        # Stage 1: Verification
        print("\n" + "=" * 80)
        print("🔍 Stage 1: Verifying Hamburg + CE relevance...")
        print("=" * 80)
        self.logger.info("Stage 1: Verifying Hamburg + CE relevance...")
        verified = self.run_verification()

        print(f"\n✅ Stage 1 complete: {len(verified)} entities verified")
        should_extract = [e for e in verified if e.get('should_extract')]
        print(f"   - {len(should_extract)} entities marked for extraction")

        # Stage 2: Extraction
        print("\n" + "=" * 80)
        print(f"📊 Stage 2: Extracting entity information...")
        print("=" * 80)
        self.logger.info("Stage 2: Extracting entity information...")
        extracted = self.run_extraction(verified)

        print(f"\n✅ Stage 2 complete: {len(extracted)} entities extracted")
        self.logger.info(f"Stage 2 complete: {len(extracted)} entities")

        # Stage 3: Geocoding
        print("\n" + "=" * 80)
        print(f"🌍 Stage 3: Geocoding entities...")
        print("=" * 80)
        self.logger.info("Stage 3: Geocoding entities...")
        geocoded = self.geocode_entities(extracted)

        geocoded_with_coords = sum(1 for p in geocoded if p.get('latitude') and p.get('longitude'))
        print(f"\n✅ Stage 3 complete: {len(geocoded)} entities processed")
        print(f"   - {geocoded_with_coords} entities have geocoded coordinates")
        self.logger.info(f"Stage 3 complete: {geocoded_with_coords}/{len(geocoded)} geocoded")

        # Stage 4: Relationship and ecosystem analysis
        if len(geocoded) > 0:
            print("\n" + "=" * 80)
            print(f"🔗 Stage 4: Analyzing relationships and ecosystem...")
            print("=" * 80)
            self.logger.info("Stage 4: Analyzing relationships and ecosystem...")
            relationships, insights = self.run_relationship_analysis(geocoded)

            print(f"\n✅ Stage 4 complete:")
            print(f"   - {len(relationships)} relationships identified")
            synergies = [i for i in insights if i.get('insight_type') == 'synergy']
            gaps = [i for i in insights if i.get('insight_type') == 'gap']
            recommendations = [i for i in insights if i.get('insight_type') == 'recommendation']
            print(f"   - {len(synergies)} synergies discovered")
            print(f"   - {len(gaps)} gaps identified")
            print(f"   - {len(recommendations)} recommendations generated")
            self.logger.info(f"Stage 4 complete: {len(relationships)} relationships, {len(insights)} insights")

            # Stage 5: Build final graph
            print("\n" + "=" * 80)
            print(f"📈 Stage 5: Building enhanced ecosystem graph...")
            print("=" * 80)
            self.logger.info("Stage 5: Building enhanced ecosystem graph...")
            graph = self.build_ecosystem_graph(geocoded, relationships, insights)
            self.save_results(geocoded, graph, relationships)

            print(f"\n✅ Stage 5 complete:")
            print(f"   - {len(graph.get('nodes', []))} nodes in graph")
            print(f"   - {len(graph.get('edges', []))} edges in graph")
            self.logger.info(f"Stage 5 complete: {len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges")
        else:
            # Fallback: build basic graph without relationships
            print(f"\n⚠️  No entities to process for Stages 4-5")
            graph = self.build_ecosystem_graph(geocoded)
            self.save_results(geocoded, graph)

        print("\n" + "=" * 80)
        print(f"✅ All stages complete!")
        print("=" * 80)
        print(f"\nFinal outputs saved to: {self.data_dir / 'final'}/")
        self.logger.info("Pipeline completed successfully")
