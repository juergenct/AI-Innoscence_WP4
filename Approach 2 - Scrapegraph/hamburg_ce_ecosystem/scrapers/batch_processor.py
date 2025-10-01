from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

from hamburg_ce_ecosystem.models.schemas import VerificationResultDict, EntityProfileDict
from hamburg_ce_ecosystem.scrapers.verification_scraper import VerificationScraper
from hamburg_ce_ecosystem.scrapers.extraction_scraper import ExtractionScraper
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
    def __init__(self, input_file: str | Path, config_path: str | Path | None = None, max_workers: int = 5):
        self.input_file = Path(input_file)
        self.config_path = Path(config_path) if config_path else (self.input_file.parent.parent.parent / 'config' / 'scrape_config.yaml')
        # Only show ERROR and CRITICAL on console, log everything to file
        self.logger = setup_logging(self.input_file.parent.parent.parent / 'logs' / 'scraping_errors.log', console_level=logging.ERROR)
        self.input_urls = self.load_input_urls(self.input_file)
        self.verification_scraper = VerificationScraper(self.config_path)
        self.extraction_scraper = ExtractionScraper(self.config_path)
        self.rate_limiter = DomainRateLimiter(max_per_second=3.0)
        self.max_workers = max_workers
        self.data_dir = self.input_file.parent.parent.parent / 'data'
        self.dm = DataManager(self.data_dir / 'final' / 'ecosystem.db')

    @staticmethod
    def load_input_urls(file_path: Path) -> Dict[str, List[str]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _verify_single(self, category: str, url: str) -> VerificationResultDict | None:
        try:
            self.rate_limiter.acquire(extract_domain(url))
            res = self.verification_scraper.verify_entity(url)
            d = res.dict()
            d['input_category'] = category
            d['url'] = str(url)  # Ensure URL is included
            return d  # type: ignore[return-value]
        except Exception as e:
            # Log but continue - skip entities where LLM completely fails
            self.logger.info(f"Skipping {url} - could not verify: {e}")
            return None

    def run_verification(self) -> List[VerificationResultDict]:
        all_tasks: List[tuple[str, str]] = []
        for category, urls in self.input_urls.items():
            for url in urls:
                all_tasks.append((category, url))

        results: List[VerificationResultDict] = []
        save_path = self.data_dir / 'verified' / 'verification_results.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._verify_single, category, url): (category, url)
                for category, url in all_tasks
            }
            
            skipped = 0
            with tqdm(total=len(all_tasks), desc="Verification", unit="entity") as pbar:
                for idx, future in enumerate(as_completed(futures), 1):
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    else:
                        skipped += 1
                    pbar.set_postfix({"✓ LLM": len(results), "⊘ Skipped": skipped})
                    pbar.update(1)
                    
                    if idx % 10 == 0:
                        pd.DataFrame(results).to_csv(save_path, index=False)
                        try:
                            self.dm.save_verification(results[-10:])
                        except Exception as e:
                            self.logger.warning(f"DB save_verification partial failed: {e}")

        pd.DataFrame(results).to_csv(save_path, index=False)
        try:
            self.dm.save_verification(results)
        except Exception as e:
            self.logger.warning(f"DB save_verification failed: {e}")
        return results

    def _extract_single(self, url: str, input_category: str) -> EntityProfileDict | None:
        try:
            source_url = str(url)
            self.rate_limiter.acquire(extract_domain(source_url))
            profile = self.extraction_scraper.extract_entity_info(source_url)
            d = profile.dict()
            d['input_category'] = input_category
            return d  # type: ignore[return-value]
        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {e}")
            return None

    def run_extraction(self, verified_entities: List[VerificationResultDict]) -> List[EntityProfileDict]:
        # LLM-gated extraction: only extract where should_extract is True
        to_extract = [e for e in verified_entities if e.get('should_extract')]
        results: List[EntityProfileDict] = []
        save_path = self.data_dir / 'extracted' / 'entity_profiles.json'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Extracting {len(to_extract)} entities (filtered from {len(verified_entities)} verified)")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_single, e['url'], e.get('input_category', '')): e['url']
                for e in to_extract
            }
            
            with tqdm(total=len(to_extract), desc="Extraction", unit="entity") as pbar:
                for idx, future in enumerate(as_completed(futures), 1):
                    res = future.result()
                    if res is not None:
                        results.append(res)
                        pbar.set_postfix({"✓": len(results), "✗": idx - len(results)})
                    pbar.update(1)
                    
                    if idx % 10 == 0:
                        save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_safe), encoding='utf-8')
                        try:
                            self.dm.save_profiles(results[-10:])
                        except Exception as e:
                            self.logger.warning(f"DB save_profiles partial failed: {e}")

        save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_safe), encoding='utf-8')
        try:
            self.dm.save_profiles(results)
        except Exception as e:
            self.logger.warning(f"DB save_profiles failed: {e}")
        return results

    def geocode_entities(self, profiles: List[EntityProfileDict]) -> List[EntityProfileDict]:
        geocoder = Geocoder()
        to_geocode = [p for p in profiles if p.get('address') and (p.get('latitude') is None or p.get('longitude') is None)]
        
        print(f"\n🌍 Geocoding {len(to_geocode)} addresses")
        
        for p in tqdm(profiles, desc="Geocoding", unit="entity"):
            if p.get('address') and (p.get('latitude') is None or p.get('longitude') is None):
                lat, lon = geocoder.geocode_hamburg(p['address'])
                p['latitude'] = lat
                p['longitude'] = lon
        return profiles

    @staticmethod
    def build_ecosystem_graph(profiles: List[EntityProfileDict]) -> Dict[str, Any]:
        graph: Dict[str, Any] = {"nodes": [], "edges": []}
        for p in profiles:
            graph["nodes"].append({
                "id": p.get('entity_name'),
                "url": p.get('url'),
                "role": p.get('ecosystem_role'),
                "lat": p.get('latitude'),
                "lon": p.get('longitude'),
            })
        known_names = {p.get('entity_name') for p in profiles if p.get('entity_name')}
        for p in profiles:
            for partner in p.get('partners', []) or []:
                if partner in known_names:
                    graph["edges"].append({
                        "source": p.get('entity_name'),
                        "target": partner,
                        "type": "partnership"
                    })
        return graph

    def save_results(self, profiles: List[EntityProfileDict], graph: Dict[str, Any]) -> None:
        out_dir = self.data_dir / 'final'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'ecosystem_map.json').write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, default=_json_safe), encoding='utf-8'
        )
        df = pd.DataFrame(profiles)
        df.to_csv(out_dir / 'ecosystem_entities.csv', index=False)

    def run_pipeline(self) -> None:
        print("\n🔍 Stage 1: Verifying Hamburg + CE relevance...")
        self.logger.info("Stage 1: Verifying Hamburg + CE relevance...")
        verified = self.run_verification()
        
        print(f"\n✅ Stage 1 complete: {len(verified)} entities verified")
        print(f"📊 Stage 2: Extracting entity information...")
        self.logger.info("Stage 2: Extracting entity information...")
        extracted = self.run_extraction(verified)
        
        print(f"\n✅ Stage 2 complete: {len(extracted)} entities extracted")
        print(f"🌍 Stage 3: Geocoding entities...")
        self.logger.info("Stage 3: Geocoding entities...")
        geocoded = self.geocode_entities(extracted)
        
        print(f"\n✅ Stage 3 complete: {len(geocoded)} entities geocoded")
        print(f"🔗 Stage 4: Building relationship graph...")
        self.logger.info("Stage 4: Building relationship graph...")
        graph = self.build_ecosystem_graph(geocoded)
        self.save_results(geocoded, graph)
        print(f"\n✅ All stages complete!")
