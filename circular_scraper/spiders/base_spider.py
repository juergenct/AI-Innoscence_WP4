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
            comprehensive: Comprehensively crawl Hamburg+CE entities (default: True)
        """
        super().__init__(*args, **kwargs)
        
        # Configuration
        self.max_depth = int(kwargs.get('max_depth', 3))
        self.follow_external = kwargs.get('follow_external', 'true').lower() == 'true'
        self.comprehensive_crawl = kwargs.get('comprehensive', 'true').lower() == 'true'
        
        # URL management
        self.visited_urls = set()
        self.domain_counts = {}  # Track requests per domain
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'items_scraped': 0,
            'hamburg_entities': 0,
            'ce_entities': 0,
            'comprehensive_entities': 0,
        }
        # Entity resolver
        self.entity_resolver = EntityResolver()
        # Track entity relevance
        self.entity_relevance = {}  # Hamburg detection
        self.entity_ce_relevance = {}  # CE detection (from LLM)
        self.comprehensive_entities = set()  # Entities to crawl comprehensively
        
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
        item['encoding'] = response.encoding if hasattr(response, 'encoding') else 'utf-8'
        
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

        # Hamburg relevance detection
        if resolved.entity_id not in self.entity_relevance:
            # First time seeing this entity - check Hamburg relevance
            is_hamburg = self._quick_hamburg_check(response)
            self.entity_relevance[resolved.entity_id] = is_hamburg
            
            if not is_hamburg:
                logger.info(f"Entity {resolved.entity_id} is NOT Hamburg-related. Stopping crawl for this entity.")
                # Mark as non-Hamburg in item for statistics
                item['has_hamburg_reference'] = False
                yield item
                # DON'T follow any links from non-Hamburg entities
                return
            else:
                logger.info(f"Entity {resolved.entity_id} IS Hamburg-related. Continuing comprehensive crawl.")
                item['has_hamburg_reference'] = True
        else:
            # We already know this entity's relevance
            is_hamburg = self.entity_relevance[resolved.entity_id]
            item['has_hamburg_reference'] = is_hamburg
            
            if not is_hamburg:
                # Skip pages from non-Hamburg entities
                logger.debug(f"Skipping page from non-Hamburg entity: {response.url}")
                return

        self.stats['items_scraped'] += 1
        
        # Check if this entity should be crawled comprehensively
        is_ce = self.entity_ce_relevance.get(resolved.entity_id, item.get('ce_llm', item.get('has_circular_economy_terms', False)))
        
        if self.comprehensive_crawl and is_hamburg and is_ce:
            # Mark for comprehensive crawling
            if resolved.entity_id not in self.comprehensive_entities:
                self.comprehensive_entities.add(resolved.entity_id)
                self.stats['comprehensive_entities'] += 1
                logger.info(f"Entity {resolved.entity_id} marked for COMPREHENSIVE crawling (Hamburg+CE)")
        
        yield item
        
        # Follow links logic
        if not is_hamburg:
            # Don't follow any links from non-Hamburg entities
            return
        
        # For Hamburg entities:
        if resolved.entity_id in self.comprehensive_entities:
            # Comprehensive crawl - no depth limit within the entity
            logger.debug(f"Comprehensive crawl for {resolved.entity_id} - following all internal links")
            yield from self.follow_links(response, comprehensive=True)
        elif item['crawl_depth'] < self.max_depth:
            # Regular crawl with depth limit
            yield from self.follow_links(response, comprehensive=False)
    
    def follow_links(self, response: Response, comprehensive: bool = False) -> Generator[Request, None, None]:
        """
        Extract and follow links from the response
        
        Args:
            response: Scrapy Response object
            comprehensive: If True, ignore depth limits for same-entity links
        
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
            
            # Get entity relevance
            source_entity = self.entity_resolver.resolve(response.url)
            target_entity = self.entity_resolver.resolve(absolute_url)
            is_same_entity = (target_entity.entity_id == source_entity.entity_id)
            source_is_hamburg = self.entity_relevance.get(source_entity.entity_id, False)
            
            # Decide whether to follow
            should_follow = False
            
            if link_domain == current_domain:
                if is_same_entity:
                    # Internal link within same entity
                    if comprehensive and source_entity.entity_id in self.comprehensive_entities:
                        # Comprehensive crawl - follow ALL internal links regardless of depth
                        should_follow = True
                    else:
                        # Regular crawl - follow if Hamburg-related and within depth
                        should_follow = source_is_hamburg and (current_depth < self.max_depth)
                else:
                    # Different entity on same domain (e.g., different institute)
                    # Only follow if we're still discovering (depth 0) or it's a candidate page
                    is_candidate = any(k in absolute_url.lower() for k in [
                        'impressum', 'imprint', 'kontakt', 'contact', 'about', 'ueber', 'ueber-uns'
                    ])
                    should_follow = (current_depth == 0) or is_candidate
            elif self.follow_external:
                # Only follow external links from Hamburg entities
                # AND they should look relevant
                # But respect depth limits unless comprehensive
                if source_is_hamburg and current_depth < self.max_depth:
                    should_follow = self._is_relevant_external_link(absolute_url, link_domain)
                else:
                    should_follow = False
            
            if should_follow:
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
        """Enhanced Hamburg relevance check for gating.
        Multiple detection strategies:
        1. Domain/URL patterns
        2. Text content keywords
        3. Postal codes (20xxx-22xxx)
        4. Hamburg district names
        5. Hamburg institutions
        6. Outbound links to Hamburg sites
        """
        url = response.url.lower()
        
        # 1. URL-based detection
        hamburg_url_patterns = ['hamburg', 'hh.de', 'hansestadt']
        if any(pattern in url for pattern in hamburg_url_patterns):
            logger.debug(f"Hamburg detected via URL: {response.url}")
            return True
        
        # Get text content (limited for performance)
        try:
            text = response.text[:10000].lower()  # Check first 10k chars
        except Exception:
            text = ''
        
        # 2. Hamburg keywords and official references
        hamburg_keywords = [
            'hamburg',
            'freie und hansestadt hamburg',
            'hansestadt hamburg',
            'metropolregion hamburg',
            'hamburg port',
            'hafen hamburg'
        ]
        if any(keyword in text for keyword in hamburg_keywords):
            logger.debug(f"Hamburg detected via keywords in: {response.url}")
            return True
        
        # 3. Hamburg postal codes (20000-22999)
        if re.search(r'\b2[0-2]\d{3}\b', text):
            logger.debug(f"Hamburg detected via postal code in: {response.url}")
            return True
        
        # 4. Hamburg districts (Bezirke and Stadtteile)
        hamburg_districts = [
            'altona', 'eimsbüttel', 'hamburg-nord', 'wandsbek',
            'bergedorf', 'harburg', 'hamburg-mitte',
            'ottensen', 'blankenese', 'winterhude', 'eppendorf',
            'st. pauli', 'hafencity', 'speicherstadt', 'veddel',
            'wilhelmsburg', 'finkenwerder', 'alsterdorf'
        ]
        if any(district in text for district in hamburg_districts):
            logger.debug(f"Hamburg detected via district name in: {response.url}")
            return True
        
        # 5. Hamburg institutions and landmarks
        hamburg_institutions = [
            'tuhh', 'haw hamburg', 'uni hamburg', 'universität hamburg',
            'hcu hamburg', 'hsba', 'hamburger hafen', 'elbphilharmonie',
            'handelskammer hamburg', 'hwf hamburg', 'hamburg chamber',
            'behörde für umwelt', 'stadtreinigung hamburg'
        ]
        if any(inst in text for inst in hamburg_institutions):
            logger.debug(f"Hamburg detected via institution in: {response.url}")
            return True
        
        # 6. Check outbound links for Hamburg references
        try:
            links = response.css('a::attr(href)').getall()[:50]  # Check first 50 links
        except Exception:
            links = []
        
        hamburg_link_count = sum(1 for link in links if link and 'hamburg' in link.lower())
        if hamburg_link_count >= 2:  # At least 2 Hamburg-related links
            logger.debug(f"Hamburg detected via outbound links in: {response.url}")
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
        logger.info(f"Hamburg entities found: {len([e for e in self.entity_relevance.values() if e])}")
        logger.info(f"CE entities found: {len([e for e in self.entity_ce_relevance.values() if e])}")
        logger.info(f"Comprehensive entities (Hamburg+CE): {len(self.comprehensive_entities)}")
        
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