# circular_scraper/utils/url_manager.py
"""
URL management utilities for tracking and filtering URLs
Handles deduplication, domain management, and URL validation
"""

import re
import logging
from typing import Set, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class URLManager:
    """
    Manages URLs for crawling
    Handles deduplication, normalization, and domain tracking
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize URL manager
        
        Args:
            data_dir: Directory for storing URL data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URL tracking
        self.seen_urls: Set[str] = set()
        self.normalized_urls: Dict[str, str] = {}  # normalized -> original
        self.domain_urls: Dict[str, Set[str]] = defaultdict(set)
        
        # Domain statistics
        self.domain_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_urls': 0,
            'crawled_urls': 0,
            'failed_urls': 0,
            'last_crawled': None,
            'avg_response_time': 0,
            'is_relevant': None,  # Will be determined by content
        })
        
        # Load existing data if available
        self.load_state()
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for deduplication
        
        Args:
            url: URL to normalize
        
        Returns:
            Normalized URL
        """
        try:
            # Parse URL
            parsed = urlparse(url.lower())
            
            # Remove default ports
            netloc = parsed.netloc
            if netloc.endswith(':80') and parsed.scheme == 'http':
                netloc = netloc[:-3]
            elif netloc.endswith(':443') and parsed.scheme == 'https':
                netloc = netloc[:-4]
            
            # Remove trailing slash from path
            path = parsed.path.rstrip('/')
            if not path:
                path = '/'
            
            # Sort query parameters
            query_params = parse_qs(parsed.query)
            # Remove common tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
                'utm_content', 'fbclid', 'gclid', 'ref', 'source'
            }
            query_params = {
                k: v for k, v in query_params.items() 
                if k not in tracking_params
            }
            sorted_query = urlencode(sorted(query_params.items()), doseq=True)
            
            # Remove fragment
            normalized = urlunparse((
                parsed.scheme,
                netloc,
                path,
                parsed.params,
                sorted_query,
                ''  # No fragment
            ))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {e}")
            return url
    
    def add_url(self, url: str, domain: str = None) -> bool:
        """
        Add URL to tracking
        
        Args:
            url: URL to add
            domain: Domain (will be extracted if not provided)
        
        Returns:
            True if URL is new, False if already seen
        """
        normalized = self.normalize_url(url)
        
        if normalized in self.seen_urls:
            return False
        
        self.seen_urls.add(normalized)
        self.normalized_urls[normalized] = url
        
        if not domain:
            domain = urlparse(url).netloc
        
        self.domain_urls[domain].add(normalized)
        self.domain_stats[domain]['total_urls'] += 1
        
        return True
    
    def mark_crawled(self, url: str, success: bool = True, 
                     response_time: float = None):
        """
        Mark URL as crawled
        
        Args:
            url: URL that was crawled
            success: Whether crawl was successful
            response_time: Response time in seconds
        """
        normalized = self.normalize_url(url)
        domain = urlparse(url).netloc
        
        if success:
            self.domain_stats[domain]['crawled_urls'] += 1
        else:
            self.domain_stats[domain]['failed_urls'] += 1
        
        self.domain_stats[domain]['last_crawled'] = datetime.now().isoformat()
        
        if response_time:
            # Update average response time
            stats = self.domain_stats[domain]
            current_avg = stats['avg_response_time']
            crawled = stats['crawled_urls']
            
            if crawled > 0:
                stats['avg_response_time'] = (
                    (current_avg * (crawled - 1) + response_time) / crawled
                )
    
    def is_url_allowed(self, url: str) -> bool:
        """
        Check if URL should be crawled
        
        Args:
            url: URL to check
        
        Returns:
            True if URL should be crawled
        """
        # Check if already seen
        normalized = self.normalize_url(url)
        if normalized in self.seen_urls:
            return False
        
        # Check URL patterns to skip
        skip_patterns = [
            r'/login', r'/logout', r'/signin', r'/signout',
            r'/register', r'/download/', r'/api/',
            r'\.pdf$', r'\.doc', r'\.xls', r'\.ppt',
            r'\.zip$', r'\.rar$', r'\.tar',
            r'/wp-admin', r'/admin',
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if re.search(pattern, url_lower):
                logger.debug(f"Skipping URL due to pattern {pattern}: {url}")
                return False
        
        # Check file extensions to skip
        skip_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.tar', '.gz', '.7z',
            '.exe', '.dmg', '.iso', '.apk',
        }
        
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        
        for ext in skip_extensions:
            if path_lower.endswith(ext):
                logger.debug(f"Skipping URL due to extension {ext}: {url}")
                return False
        
        return True
    
    def get_domain_priority(self, domain: str) -> int:
        """
        Get crawl priority for a domain
        
        Args:
            domain: Domain to check
        
        Returns:
            Priority score (higher = more important)
        """
        priority = 0
        
        # Prioritize Hamburg domains
        if 'hamburg' in domain.lower():
            priority += 20
        
        # Prioritize German domains
        if domain.endswith('.de'):
            priority += 10
        
        # Prioritize university domains
        uni_patterns = ['uni-', 'tuhh', 'haw-', 'hcu-', '.edu']
        if any(pattern in domain.lower() for pattern in uni_patterns):
            priority += 15
        
        # Prioritize known relevant domains
        relevant_keywords = [
            'recycl', 'sustainab', 'circular', 'umwelt',
            'waste', 'klima', 'energie', 'bioeconom'
        ]
        if any(kw in domain.lower() for kw in relevant_keywords):
            priority += 10
        
        # Deprioritize large sites that might have too much content
        large_sites = ['wikipedia', 'amazon', 'ebay', 'linkedin', 'facebook']
        if any(site in domain.lower() for site in large_sites):
            priority -= 20
        
        # Use domain statistics if available
        stats = self.domain_stats.get(domain, {})
        if stats.get('is_relevant') is True:
            priority += 25
        elif stats.get('is_relevant') is False:
            priority -= 15
        
        # Penalize slow domains
        avg_response = stats.get('avg_response_time', 0)
        if avg_response > 10:
            priority -= 10
        elif avg_response > 5:
            priority -= 5
        
        return priority
    
    def get_next_urls(self, count: int = 10, 
                     prioritize: bool = True) -> List[str]:
        """
        Get next URLs to crawl
        
        Args:
            count: Number of URLs to return
            prioritize: Whether to prioritize by domain
        
        Returns:
            List of URLs to crawl
        """
        # Get uncrawled URLs
        uncrawled = []
        
        for domain, urls in self.domain_urls.items():
            stats = self.domain_stats[domain]
            crawled_count = stats['crawled_urls'] + stats['failed_urls']
            
            for url_norm in urls:
                if crawled_count < len(urls):
                    url = self.normalized_urls[url_norm]
                    if prioritize:
                        priority = self.get_domain_priority(domain)
                        uncrawled.append((priority, url))
                    else:
                        uncrawled.append((0, url))
        
        # Sort by priority
        if prioritize:
            uncrawled.sort(key=lambda x: x[0], reverse=True)
        
        return [url for _, url in uncrawled[:count]]
    
    def mark_domain_relevant(self, domain: str, is_relevant: bool):
        """
        Mark a domain as relevant or not for circular economy
        
        Args:
            domain: Domain to mark
            is_relevant: Whether domain is relevant
        """
        self.domain_stats[domain]['is_relevant'] = is_relevant
        logger.info(f"Marked domain {domain} as {'relevant' if is_relevant else 'not relevant'}")
    
    def get_statistics(self) -> Dict:
        """
        Get crawling statistics
        
        Returns:
            Dictionary with statistics
        """
        total_urls = len(self.seen_urls)
        total_domains = len(self.domain_urls)
        
        crawled_urls = sum(
            stats['crawled_urls'] 
            for stats in self.domain_stats.values()
        )
        
        failed_urls = sum(
            stats['failed_urls'] 
            for stats in self.domain_stats.values()
        )
        
        relevant_domains = sum(
            1 for stats in self.domain_stats.values() 
            if stats.get('is_relevant') is True
        )
        
        return {
            'total_urls_seen': total_urls,
            'total_domains': total_domains,
            'crawled_urls': crawled_urls,
            'failed_urls': failed_urls,
            'pending_urls': total_urls - crawled_urls - failed_urls,
            'relevant_domains': relevant_domains,
            'success_rate': crawled_urls / (crawled_urls + failed_urls) if crawled_urls + failed_urls > 0 else 0,
        }
    
    def save_state(self):
        """Save current state to disk"""
        state_file = self.data_dir / 'url_manager_state.json'
        
        state = {
            'seen_urls': list(self.seen_urls),
            'normalized_urls': self.normalized_urls,
            'domain_urls': {
                domain: list(urls) 
                for domain, urls in self.domain_urls.items()
            },
            'domain_stats': dict(self.domain_stats),
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved URL manager state to {state_file}")
    
    def load_state(self):
        """Load state from disk if available"""
        state_file = self.data_dir / 'url_manager_state.json'
        
        if not state_file.exists():
            logger.info("No previous URL manager state found")
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.seen_urls = set(state.get('seen_urls', []))
            self.normalized_urls = state.get('normalized_urls', {})
            
            domain_urls = state.get('domain_urls', {})
            self.domain_urls = defaultdict(set)
            for domain, urls in domain_urls.items():
                self.domain_urls[domain] = set(urls)
            
            self.domain_stats = defaultdict(
                lambda: {
                    'total_urls': 0,
                    'crawled_urls': 0,
                    'failed_urls': 0,
                    'last_crawled': None,
                    'avg_response_time': 0,
                    'is_relevant': None,
                },
                state.get('domain_stats', {})
            )
            
            logger.info(f"Loaded URL manager state: {len(self.seen_urls)} URLs, "
                       f"{len(self.domain_urls)} domains")
            
        except Exception as e:
            logger.error(f"Error loading URL manager state: {e}")
    
    def export_urls(self, output_file: str = None):
        """
        Export URLs to CSV file
        
        Args:
            output_file: Output file path
        """
        if not output_file:
            output_file = self.data_dir / f'urls_export_{datetime.now():%Y%m%d_%H%M%S}.csv'
        
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'url', 'domain', 'crawled', 'failed', 'relevant',
                'last_crawled', 'avg_response_time'
            ])
            writer.writeheader()
            
            for domain, urls in self.domain_urls.items():
                stats = self.domain_stats[domain]
                
                for url_norm in urls:
                    url = self.normalized_urls[url_norm]
                    writer.writerow({
                        'url': url,
                        'domain': domain,
                        'crawled': stats['crawled_urls'] > 0,
                        'failed': stats['failed_urls'] > 0,
                        'relevant': stats.get('is_relevant'),
                        'last_crawled': stats.get('last_crawled'),
                        'avg_response_time': stats.get('avg_response_time', 0),
                    })
        
        logger.info(f"Exported URLs to {output_file}")