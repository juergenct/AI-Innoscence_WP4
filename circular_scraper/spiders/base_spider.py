# circular_scraper/spiders/base_spider.py
"""
Base spider class with common functionality
All specific spiders inherit from this base class
"""

import csv
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator
from urllib.parse import urlparse, urljoin
from datetime import datetime

import scrapy
from scrapy import Request, Spider
from scrapy.http import Response

from circular_scraper.items import CircularEconomyItem, ErrorItem
from circular_scraper.utils.entity_resolver import EntityResolver


logger = logging.getLogger(__name__)


class BaseCircularEconomySpider(Spider):
    """
    Base spider with common functionality for all circular economy spiders
    Handles URL management, depth tracking, and basic parsing
    """
    
    name = 'base_spider'
    
    # Custom settings that can be overridden
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'DEPTH_LIMIT': 3,
    }
    
    def __init__(self, *args, **kwargs):
        """
        Initialize spider with configuration
        
        Args:
            start_url: Single URL to start crawling
            seed_file: CSV file with multiple URLs
            max_depth: Maximum crawl depth (default: 3)
            follow_external: Follow external links (default: True)
        """
        super().__init__(*args, **kwargs)
        
        # Configuration
        self.max_depth = int(kwargs.get('max_depth', 3))
        self.follow_external = kwargs.get('follow_external', 'true').lower() == 'true'
        
        # URL management
        self.visited_urls = set()
        self.domain_counts = {}  # Track requests per domain
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'items_scraped': 0,
        }
        # Entity resolver
        self.entity_resolver = EntityResolver()
        # Track entity relevance (Hamburg gating)
        self.entity_relevance = {}
        
        # Load start URLs
        self.start_urls = self._load_start_urls(kwargs)
        
        logger.info(f"Initialized {self.name} with {len(self.start_urls)} start URLs")
        logger.info(f"Max depth: {self.max_depth}, Follow external: {self.follow_external}")
    
    def _load_start_urls(self, kwargs) -> List[str]:
        """Load starting URLs from arguments or file"""
        urls: List[str] = []
        
        # Single URL provided
        if 'start_url' in kwargs:
            candidate = (kwargs['start_url'] or '').strip()
            if candidate:
                urls.append(candidate)
        
        # CSV file with URLs
        elif 'seed_file' in kwargs:
            seed_path = Path(kwargs['seed_file'])
            if seed_path.exists():
                with open(seed_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Support multiple common column names
                        raw_url = (
                            (row.get('url') or '').strip() or
                            (row.get('URL') or '').strip() or
                            (row.get('website') or '').strip()
                        )
                        if not raw_url:
                            continue
                        # Ensure scheme
                        if not raw_url.startswith(('http://', 'https://')):
                            raw_url = f"https://{raw_url}"
                        # Basic cleanup: strip whitespace and trailing slashes normalization
                        cleaned = raw_url.strip()
                        urls.append(cleaned)
            else:
                logger.error(f"Seed file not found: {seed_path}")
        
        # Normalize and deduplicate while preserving order
        seen = set()
        normalized_urls: List[str] = []
        for u in urls:
            if not u:
                continue
            # Lowercase scheme+host only
            try:
                parsed = urlparse(u)
                scheme = (parsed.scheme or 'https').lower()
                netloc = parsed.netloc.lower()
                path = parsed.path or '/'
                rebuilt = f"{scheme}://{netloc}{path}"
            except Exception:
                rebuilt = u
            if rebuilt not in seen:
                seen.add(rebuilt)
                normalized_urls.append(rebuilt)
        
        # Default URLs if none provided
        if not normalized_urls:
            normalized_urls = [
                'https://www.tuhh.de/unuhub/capacity-building/campuslab-circular-economy',
                'https://www.tuhh.de/crem/en/welcome-1',
            ]
            logger.warning("No URLs provided, using default TUHH URLs")
        
        return normalized_urls
    
    def start_requests(self) -> Generator[Request, None, None]:
        """Generate initial requests"""
        for url in self.start_urls:
            yield self.make_request(
                url=url,
                callback=self.parse,
                meta={'depth': 0, 'parent_url': None}
            )
    
    def make_request(self, url: str, callback=None, meta: Dict = None, 
                    priority: int = 0, dont_filter: bool = False) -> Request:
        """
        Create a request with proper configuration
        
        Args:
            url: URL to request
            callback: Callback function for response
            meta: Request metadata
            priority: Request priority
            dont_filter: Skip duplicate filtering
        
        Returns:
            Configured Request object
        """
        if meta is None:
            meta = {}
        
        # Add tracking metadata
        meta.setdefault('depth', 0)
        meta.setdefault('handle_httpstatus_all', True)
        meta['request_time'] = datetime.now().isoformat()
        
        # Track domain
        domain = urlparse(url).netloc
        self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
        
        self.stats['total_requests'] += 1
        
        return Request(
            url=url,
            callback=callback or self.parse,
            meta=meta,
            priority=priority,
            dont_filter=dont_filter,
            errback=self.handle_error
        )
    
    def parse(self, response: Response) -> Generator:
        """
        Main parsing method - must be implemented by subclasses
        
        Args:
            response: Scrapy Response object
        
        Yields:
            Items and new Requests
        """
        raise NotImplementedError("Subclasses must implement parse method")
    
    def parse_page(self, response: Response) -> Generator:
        """
        Parse a single page and extract data
        
        Args:
            response: Scrapy Response object
        
        Yields:
            CircularEconomyItem and follow-up Requests
        """
        self.stats['successful_responses'] += 1
        
        # Create item
        item = CircularEconomyItem()
        
        # Basic metadata
        item['url'] = response.url
        item['domain'] = urlparse(response.url).netloc
        item['scraped_at'] = datetime.now().isoformat()
        item['crawl_depth'] = response.meta.get('depth', 0)
        item['parent_url'] = response.meta.get('parent_url')
        
        # Response metadata
        item['response_status'] = response.status
        item['response_time'] = (
            datetime.now() - datetime.fromisoformat(response.meta.get('request_time', datetime.now().isoformat()))
        ).total_seconds()
        item['encoding'] = response.encoding
        
        # Store raw HTML
        item['raw_html'] = response.text
        item['content_length'] = len(response.body)
        
        # Page type (static or dynamic)
        item['page_type'] = 'dynamic' if response.meta.get('playwright') else 'static'
        
        # Extract basic info using Scrapy selectors when response is HTML
        title_text = ''
        try:
            title_text = response.css('title::text').get(default='').strip()
        except Exception:
            title_text = ''
        item['title'] = title_text
        
        # Meta description (guard if non-HTML)
        meta_desc = ''
        try:
            meta_desc = response.css('meta[name="description"]::attr(content)').get()
            if not meta_desc:
                meta_desc = response.css('meta[property="og:description"]::attr(content)').get()
        except Exception:
            meta_desc = ''
        item['meta_description'] = (meta_desc or '').strip()
        
        # Language hint from HTML attributes (best-effort)
        lang = None
        try:
            lang = response.css('html::attr(lang)').get()
            if not lang:
                lang = response.css('meta[http-equiv="content-language"]::attr(content)').get()
        except Exception:
            lang = None
        item['language'] = (lang or 'unknown').lower()[:2]
        
        # Extract structured data if present
        try:
            json_ld = response.css('script[type="application/ld+json"]::text').getall()
            if json_ld:
                try:
                    item['structured_data'] = [json.loads(ld) for ld in json_ld]
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        
        # Resolve entity grouping
        resolved = self.entity_resolver.resolve(item['url'], page_title=item.get('title'))
        item['entity_id'] = resolved.entity_id
        item['entity_name'] = resolved.entity_name
        item['entity_root_url'] = resolved.entity_root_url

        # Quick Hamburg relevance detection to gate deeper crawling
        if self._quick_hamburg_check(response):
            self.entity_relevance[resolved.entity_id] = True
        else:
            self.entity_relevance.setdefault(resolved.entity_id, False)

        self.stats['items_scraped'] += 1
        yield item
        
        # Follow links if depth allows
        if item['crawl_depth'] < self.max_depth:
            yield from self.follow_links(response)
    
    def follow_links(self, response: Response) -> Generator[Request, None, None]:
        """
        Extract and follow links from the response
        
        Args:
            response: Scrapy Response object
        
        Yields:
            New Request objects for discovered links
        """
        current_depth = response.meta.get('depth', 0)
        current_domain = urlparse(response.url).netloc
        
        # Extract all links (if HTML)
        try:
            links = response.css('a::attr(href)').getall()
        except Exception:
            links = []
        
        for link in links:
            # Convert to absolute URL
            absolute_url = urljoin(response.url, link)
            
            # Skip non-HTTP URLs
            if not absolute_url.startswith(('http://', 'https://')):
                continue
            
            # Skip already visited URLs
            if absolute_url in self.visited_urls:
                continue
            
            # Check domain
            link_domain = urlparse(absolute_url).netloc
            
            # Decide whether to follow
            should_follow = False
            
            if link_domain == current_domain:
                # Always follow internal links
                should_follow = True
            elif self.follow_external:
                # Follow external links if enabled and they look relevant
                should_follow = self._is_relevant_external_link(absolute_url, link_domain)
            
            if should_follow:
                # Ensure we stay within same entity appropriately, and gate by Hamburg relevance
                source_entity = self.entity_resolver.resolve(response.url)
                target_entity = self.entity_resolver.resolve(absolute_url)
                is_same_entity = (target_entity.entity_id == source_entity.entity_id)
                is_entity_relevant = self.entity_relevance.get(source_entity.entity_id)
                # Candidate pages often contain contact/address info
                is_candidate = any(k in absolute_url.lower() for k in [
                    'impressum', 'imprint', 'kontakt', 'contact', 'about', 'ueber', 'ueber-uns', 'privacy'
                ])

                if is_same_entity:
                    # Allow if depth 0 (discovery), or already marked relevant, or candidate page to determine relevance
                    if not (current_depth == 0 or is_entity_relevant or is_candidate):
                        continue
                else:
                    # Different entity under same domain (e.g., different TUHH institute)
                    if link_domain == current_domain and current_depth > 0 and not is_candidate:
                        continue

                self.visited_urls.add(absolute_url)
                
                yield self.make_request(
                    url=absolute_url,
                    callback=self.parse_page,
                    meta={
                        'depth': current_depth + 1,
                        'parent_url': response.url,
                    },
                    priority=-current_depth  # Prioritize shallower pages
                )
    
    def _is_relevant_external_link(self, url: str, domain: str) -> bool:
        """
        Determine if an external link is relevant for crawling
        
        Args:
            url: The URL to check
            domain: The domain of the URL
        
        Returns:
            True if the link should be followed
        """
        # Skip common non-relevant domains
        skip_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
            'linkedin.com', 'xing.com', 'google.com', 'wikipedia.org',
            'amazon.com', 'ebay.com',
        }
        
        if any(skip in domain for skip in skip_domains):
            return False
        
        # Prioritize German and Hamburg-related domains
        if '.de' in domain or 'hamburg' in domain.lower():
            return True
        
        # Check for relevant keywords in domain
        relevant_keywords = [
            'recycl', 'sustainab', 'circular', 'waste', 'umwelt',
            'research', 'institut', 'startup', 'innovat'
        ]
        
        domain_lower = domain.lower()
        return any(keyword in domain_lower for keyword in relevant_keywords)

    def _quick_hamburg_check(self, response: Response) -> bool:
        """Lightweight Hamburg relevance check for gating.
        Checks domain, simple keywords, postal pattern, and outbound links.
        """
        url = response.url.lower()
        if 'hamburg' in url:
            return True
        try:
            text = response.text.lower()
        except Exception:
            text = ''
        if 'hamburg' in text or 'freie und hansestadt hamburg' in text:
            return True
        # Hamburg postal codes (rough pattern 20xxx–22xxx)
        if re.search(r'\b2[0-2]\d{3}\b', text):
            return True
        try:
            links = response.css('a::attr(href)').getall()
        except Exception:
            links = []
        if any('hamburg' in (link or '').lower() for link in links):
            return True
        return False
    
    def handle_error(self, failure):
        """
        Handle request failures
        
        Args:
            failure: Twisted Failure instance
        """
        self.stats['failed_responses'] += 1
        
        # Create error item
        error_item = ErrorItem()
        error_item['url'] = failure.request.url
        error_item['domain'] = urlparse(failure.request.url).netloc
        error_item['error_type'] = failure.type.__name__
        error_item['error_message'] = str(failure.value)
        error_item['error_traceback'] = failure.getTraceback()
        error_item['occurred_at'] = datetime.now().isoformat()
        error_item['retry_count'] = failure.request.meta.get('retry_times', 0)
        error_item['parent_url'] = failure.request.meta.get('parent_url')
        error_item['crawl_depth'] = failure.request.meta.get('depth', 0)
        
        logger.error(f"Request failed for {failure.request.url}: {failure.value}")
        
        return error_item
    
    def closed(self, reason):
        """
        Called when spider closes
        
        Args:
            reason: Reason for closing
        """
        # Log final statistics
        logger.info(f"Spider {self.name} closed: {reason}")
        logger.info(f"Final statistics: {self.stats}")
        logger.info(f"Domains crawled: {len(self.domain_counts)}")
        logger.info(f"Total URLs visited: {len(self.visited_urls)}")
        
        # Save statistics to file
        stats_file = Path('data/exports') / f"spider_stats_{self.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump({
                'spider': self.name,
                'reason': reason,
                'stats': self.stats,
                'domain_counts': self.domain_counts,
                'total_urls': len(self.visited_urls),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")