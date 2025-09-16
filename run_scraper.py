#!/usr/bin/env python
"""
Main script to run the Circular Economy Hamburg Web Scraper
Provides a simple interface to start scraping with different configurations
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import json
import csv
from typing import List, Set

from circular_scraper.utils.entity_resolver import EntityResolver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scraper_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScraperRunner:
    """Main scraper execution class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
    def run_static_spider(self, seed_file: str = None, start_url: str = None, 
                         max_depth: int = 3):
        """
        Run the static spider for simple HTML sites
        
        Args:
            seed_file: CSV file with URLs
            start_url: Single URL to start
            max_depth: Maximum crawl depth
        """
        logger.info("Starting Static Spider...")
        
        cmd = [sys.executable, '-m', 'scrapy', 'crawl', 'static_spider']
        
        if seed_file:
            cmd.extend(['-a', f'seed_file={seed_file}'])
        elif start_url:
            cmd.extend(['-a', f'start_url={start_url}'])
        else:
            cmd.extend(['-a', 'seed_file=/home/thiesen/Documents/AI-Innoscence_Ecosystem/data/processed/stakeholders_hamburg.csv'])
        
        cmd.extend(['-a', f'max_depth={max_depth}'])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Static spider completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Static spider failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def run_dynamic_spider(self, start_url: str, max_depth: int = 2):
        """
        Run the dynamic spider for JavaScript-heavy sites
        
        Args:
            start_url: URL to start crawling
            max_depth: Maximum crawl depth
        """
        logger.info("Starting Dynamic Spider...")
        
        cmd = [
            sys.executable, '-m', 'scrapy', 'crawl', 'dynamic_spider',
            '-a', f'start_url={start_url}',
            '-a', f'max_depth={max_depth}'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Dynamic spider completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dynamic spider failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def run_full_crawl(self, seed_file: str = '/home/thiesen/Documents/AI-Innoscence_Ecosystem/data/processed/stakeholders_hamburg.csv', max_depth: int = 3):
        """
        Run a full crawl with intelligent spider selection
        
        Args:
            seed_file: CSV file with seed URLs
            max_depth: Maximum crawl depth
        """
        logger.info("Starting Full Crawl...")
        
        # First, run static spider on all URLs
        logger.info("Phase 1: Running static spider on seed URLs...")
        self.run_static_spider(seed_file=seed_file, max_depth=max_depth)
        
        # Check if any sites need dynamic rendering
        logger.info("Phase 2: Checking for JavaScript-heavy sites...")
        js_sites = self._identify_javascript_sites()
        
        if js_sites:
            logger.info(f"Found {len(js_sites)} sites that may need JavaScript rendering")
            for site in js_sites:
                logger.info(f"Running dynamic spider on: {site}")
                self.run_dynamic_spider(start_url=site, max_depth=2)
        
        logger.info("Full crawl completed")
        return True
    
    def _identify_javascript_sites(self):
        """
        Identify sites that might need JavaScript rendering
        based on initial crawl results
        """
        # This would analyze the scraped data to find sites with
        # incomplete content that might need JS rendering
        # For now, return known JS-heavy sites
        return [
            'https://cirplus.com/',
            # Add more as identified
        ]
    
    def export_data(self, format: str = 'excel'):
        """
        Export scraped data to desired format
        
        Args:
            format: Export format (excel, parquet, llm, all)
        """
        logger.info(f"Exporting data to {format} format...")
        
        cmd = [
            sys.executable, '-m', 'circular_scraper.utils.export_manager',
            '--format', format,
            '--summary'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Data export completed")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'scrapy', 'scrapy_playwright', 'beautifulsoup4',
            'trafilatura', 'pandas', 'pyarrow', 'fake_useragent'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
        
        # Check Playwright browsers
        try:
            result = subprocess.run(
                ['playwright', 'install', '--help'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("Playwright browsers might not be installed")
                logger.info("Install with: playwright install chromium")
        except FileNotFoundError:
            logger.warning("Playwright command not found")
        
        logger.info("All dependencies are installed")
        return True

    def _latest_iteration_seeds(self) -> Path | None:
        """Return the latest iteration_seeds_*.csv file path if exists."""
        export_dir = self.data_dir / 'exports'
        # Look for CSV files (easier to load than txt)
        candidates = sorted(export_dir.glob('iteration_seeds_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        # Fallback to old discovered_links files for backwards compatibility
        candidates = sorted(export_dir.glob('discovered_links_*.txt'), key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _write_seed_csv(self, urls: List[str], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['website'])
            writer.writeheader()
            for u in urls:
                writer.writerow({'website': u})
        return out_path

    def _entity_dedup(self, urls: List[str], limit: int = 50) -> List[str]:
        """Group URLs by entity root and return one root per entity up to limit."""
        resolver = EntityResolver()
        seen: Set[str] = set()
        seeds: List[str] = []
        for u in urls:
            if not u.startswith(('http://', 'https://')):
                u = f'https://{u}'
            entity = resolver.resolve(u)
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            seeds.append(entity.entity_root_url)
            if len(seeds) >= limit:
                break
        return seeds

    def run_iterative_crawl(self, start_seed_file: str, max_depth: int = 2, iterations: int = 1, per_iter_limit: int = 50):
        """Run N iterations; next iterations use discovered links grouped by entity.

        Args:
            start_seed_file: initial CSV with seeds (website/url column)
            max_depth: spider depth per iteration
            iterations: number of iterations to run
            per_iter_limit: limit of entity roots to seed per iteration
        """
        logger.info(f"Starting iterative crawl for {iterations} iteration(s)")

        seeds_csv = Path(start_seed_file)
        iter_dir = self.data_dir / 'iter_seeds'

        for i in range(1, iterations + 1):
            logger.info(f"Iteration {i}: using seeds from {seeds_csv}")
            self.run_static_spider(seed_file=str(seeds_csv), max_depth=max_depth)

            # After spider run, export data and aggregated entities
            self.export_data('excel')
            try:
                # Also export aggregated entities table
                from circular_scraper.utils.export_manager import ExportManager
                ExportManager(str(self.data_dir)).consolidate_session_data()
                ExportManager(str(self.data_dir)).export_entities_table()
            except Exception as e:
                logger.warning(f"Could not export aggregated entities table: {e}")

            # Prepare next iteration seeds from discovered links
            latest_seeds = self._latest_iteration_seeds()
            if not latest_seeds:
                logger.info("No iteration seeds found; stopping iterations.")
                break
            
            # Handle CSV or TXT format
            try:
                if latest_seeds.suffix == '.csv':
                    # Read from CSV
                    import pandas as pd
                    df = pd.read_csv(latest_seeds)
                    links = df['website'].tolist() if 'website' in df else []
                else:
                    # Read from TXT (backwards compatibility)
                    with open(latest_seeds, 'r', encoding='utf-8') as f:
                        links = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                logger.warning(f"Failed to read iteration seeds: {e}")
                break

            next_seeds = self._entity_dedup(links, limit=per_iter_limit)
            if not next_seeds:
                logger.info("No new seeds derived; stopping iterations.")
                break

            seeds_csv = iter_dir / f"iter_{datetime.now():%Y%m%d_%H%M%S}_{i:02d}.csv"
            self._write_seed_csv(next_seeds, seeds_csv)
        
        logger.info("Iterative crawl finished")
        return True

    def classify_entities_with_llm(self, sample_per_entity: int = 3, model: str = 'qwen3-4b-instruct', max_entities: int | None = None, debug_dir: str | None = None):
        """Classify entities via Ollama LLM and write annotated aggregated CSV."""
        from circular_scraper.utils.export_manager import ExportManager
        from circular_scraper.utils.llm_classifier import LLMClassifier

        manager = ExportManager(str(self.data_dir))
        data = manager.consolidate_session_data()
        entities_df = data.get('entities')
        if entities_df is None or entities_df.empty:
            logger.error("No entities data available for classification")
            return False

        # Ensure aggregated table exists
        agg_df = manager.export_entities_table()
        aggregated_csv = str(self.data_dir / 'exports' / 'entities_aggregated.csv')

        # Run LLM classifier
        dbg = str(self.data_dir / 'exports' / 'llm_debug') if debug_dir is None else debug_dir
        clf = LLMClassifier(model=model, debug_dir=dbg)
        classified = clf.classify_from_entities_df(entities_df, sample_per_entity=sample_per_entity, max_entities=max_entities)
        out_path = clf.merge_into_aggregated(aggregated_csv, classified)
        print(f"LLM classification written to: {out_path}")
        return True
    
    def show_statistics(self):
        """Display scraping statistics"""
        stats_files = list(Path('data/exports').glob('spider_stats_*.json'))
        
        if not stats_files:
            logger.info("No statistics files found")
            return
        
        # Get the latest stats file
        latest_stats = max(stats_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_stats, 'r') as f:
            stats = json.load(f)
        
        print("\n" + "=" * 60)
        print("SCRAPING STATISTICS")
        print("=" * 60)
        print(f"Spider: {stats.get('spider')}")
        print(f"Timestamp: {stats.get('timestamp')}")
        print(f"Total URLs visited: {stats.get('total_urls')}")
        print(f"Domains crawled: {len(stats.get('domain_counts', {}))}")
        
        if 'stats' in stats:
            print("\nDetailed Statistics:")
            for key, value in stats['stats'].items():
                print(f"  {key}: {value}")
        
        if 'domain_counts' in stats:
            print("\nTop Domains:")
            sorted_domains = sorted(
                stats['domain_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for domain, count in sorted_domains:
                print(f"  {domain}: {count} pages")
        
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Circular Economy Hamburg Web Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full crawl with default settings
  python run_scraper.py --full
  
  # Scrape a single URL
  python run_scraper.py --url https://www.tuhh.de/crem/en/welcome-1
  
  # Use custom seed file
  python run_scraper.py --seed my_urls.csv
  
  # Export data after scraping
  python run_scraper.py --export excel
  
  # Check dependencies
  python run_scraper.py --check
        """
    )
    
    # Crawling options
    parser.add_argument('--full', action='store_true',
                       help='Run full crawl with seed URLs')
    parser.add_argument('--url', type=str,
                       help='Single URL to scrape')
    parser.add_argument('--seed', type=str, default='/home/thiesen/Documents/AI-Innoscence_Ecosystem/data/processed/stakeholders_hamburg.csv',
                       help='CSV file with seed URLs')
    parser.add_argument('--depth', type=int, default=3,
                       help='Maximum crawl depth (default: 3)')
    parser.add_argument('--spider', choices=['static', 'dynamic'],
                       default='static',
                       help='Spider type to use')
    parser.add_argument('--iterate', type=int, default=0,
                       help='Run N iterative rounds using discovered links (default: 0)')
    parser.add_argument('--iter-limit', type=int, default=50,
                       help='Max entity seeds per iteration (default: 50)')
    
    # Export options
    parser.add_argument('--export', choices=['excel', 'parquet', 'llm', 'all'],
                       help='Export data after scraping')
    parser.add_argument('--classify', action='store_true',
                       help='Classify entities via local LLM (Ollama)')
    parser.add_argument('--llm-samples', type=int, default=3,
                       help='Samples per entity for LLM classification (default: 3)')
    parser.add_argument('--llm-model', type=str, default='qwen3-4b-instruct',
                       help='Ollama model name (default: qwen3-4b-instruct)')
    parser.add_argument('--llm-max-entities', type=int, default=0,
                       help='Limit number of entities to classify (0=all)')
    parser.add_argument('--llm-debug-dir', type=str, default=None,
                       help='Directory to dump LLM request/response for debugging')
    
    # Utility options
    parser.add_argument('--check', action='store_true',
                       help='Check dependencies')
    parser.add_argument('--stats', action='store_true',
                       help='Show scraping statistics')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ScraperRunner()
    
    # Handle different commands
    if args.check:
        runner.check_dependencies()
        return
    
    if args.stats:
        runner.show_statistics()
        return
    
    if args.export:
        runner.export_data(args.export)
        return
    
    # Run scraping
    if args.full:
        runner.run_full_crawl(seed_file=args.seed, max_depth=args.depth)
        # Auto-export after full crawl
        runner.export_data('excel')
        runner.show_statistics()
    
    elif args.url:
        if args.spider == 'dynamic':
            runner.run_dynamic_spider(start_url=args.url, max_depth=args.depth)
        else:
            runner.run_static_spider(start_url=args.url, max_depth=args.depth)
        runner.export_data('excel')
    
    elif args.seed and args.iterate > 0:
        runner.run_iterative_crawl(start_seed_file=args.seed, max_depth=args.depth, iterations=args.iterate, per_iter_limit=args.iter_limit)
        runner.show_statistics()
    
    elif args.seed:
        runner.run_static_spider(seed_file=args.seed, max_depth=args.depth)
        runner.export_data('excel')

    if args.classify:
        max_e = args.llm_max_entities if args.llm_max_entities and args.llm_max_entities > 0 else None
        runner.classify_entities_with_llm(sample_per_entity=args.llm_samples, model=args.llm_model, max_entities=max_e, debug_dir=args.llm_debug_dir)
    
    else:
        # Default: show help
        parser.print_help()


if __name__ == '__main__':
    main()