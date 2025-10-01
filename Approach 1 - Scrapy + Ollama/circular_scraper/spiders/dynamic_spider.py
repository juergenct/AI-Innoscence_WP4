# circular_scraper/spiders/dynamic_spider.py
"""
Spider for scraping JavaScript-heavy websites
Uses Playwright for rendering dynamic content
"""

import logging
from typing import Generator
from urllib.parse import urlparse

from scrapy.http import Response
from scrapy_playwright.page import PageMethod

from circular_scraper.spiders.base_spider import BaseCircularEconomySpider


logger = logging.getLogger(__name__)


class DynamicSpider(BaseCircularEconomySpider):
    """
    Spider for JavaScript-heavy websites
    Uses Playwright to render pages before extraction
    """
    
    name = 'dynamic_spider'
    
    # Custom settings for dynamic crawling
    custom_settings = {
        'DOWNLOAD_DELAY': 3,  # Longer delay for JavaScript rendering
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # Fewer concurrent requests
        'DEPTH_LIMIT': 2,  # Shallower depth for performance
        'DOWNLOAD_TIMEOUT': 45,  # Longer timeout for rendering
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 45000,
    }
    
    def start_requests(self) -> Generator:
        """
        Generate initial requests with Playwright configuration
        """
        for url in self.start_urls:
            yield self.make_playwright_request(
                url=url,
                callback=self.parse,
                meta={'depth': 0, 'parent_url': None}
            )
    
    def make_playwright_request(self, url: str, callback=None, 
                               meta: dict = None, wait_for: str = 'networkidle',
                               wait_timeout: int = 3000):
        """
        Create a Playwright-enabled request
        
        Args:
            url: URL to request
            callback: Callback function
            meta: Request metadata
            wait_for: Wait condition ('load', 'domcontentloaded', 'networkidle')
            wait_timeout: Additional wait time in milliseconds
        
        Returns:
            Request configured for Playwright
        """
        if meta is None:
            meta = {}
        
        # Enable Playwright
        meta['playwright'] = True
        meta['playwright_include_page'] = True
        
        # Configure page methods
        page_methods = [
            PageMethod('wait_for_load_state', wait_for),
        ]
        
        # Add extra wait for dynamic content
        if wait_timeout > 0:
            page_methods.append(
                PageMethod('wait_for_timeout', wait_timeout)
            )
        
        # Try to wait for specific selectors on known sites
        domain = urlparse(url).netloc
        if 'cirplus' in domain:
            page_methods.insert(0, 
                PageMethod('wait_for_selector', 'main', timeout=10000)
            )
        
        meta['playwright_page_methods'] = page_methods
        
        # Configure context
        meta['playwright_context_kwargs'] = {
            'viewport': {'width': 1920, 'height': 1080},
            'java_script_enabled': True,
            'ignore_https_errors': True,
            'locale': 'de-DE',  # German locale
            'timezone_id': 'Europe/Berlin',
        }
        
        # Add page initialization script to hide cookie banners
        meta['playwright_page_init'] = """
        async (page) => {
            // Try to auto-accept cookie banners
            await page.evaluateOnNewDocument(() => {
                // Common cookie banner selectors
                const selectors = [
                    '[id*="cookie"] [id*="accept"]',
                    '[class*="cookie"] [class*="accept"]',
                    '[id*="consent"] [id*="accept"]',
                    'button[onclick*="accept"]'
                ];
                
                setTimeout(() => {
                    selectors.forEach(selector => {
                        const button = document.querySelector(selector);
                        if (button) button.click();
                    });
                }, 2000);
            });
        }
        """
        
        return self.make_request(
            url=url,
            callback=callback or self.parse,
            meta=meta
        )
    
    def parse(self, response: Response) -> Generator:
        """
        Parse initial response from start URLs
        
        Args:
            response: Scrapy Response object with Playwright rendering
        
        Yields:
            Items and follow-up requests
        """
        # Log if Playwright was actually used
        if response.meta.get('playwright'):
            logger.info(f"Successfully rendered {response.url} with Playwright")
            
            # Check if we got the full content
            content_length = len(response.text)
            if content_length < 5000:
                logger.warning(f"Possibly incomplete render for {response.url}: only {content_length} chars")
        
        # Parse the page
        yield from self.parse_page(response)
    
    def parse_page(self, response: Response) -> Generator:
        """
        Parse a dynamically rendered page
        
        Args:
            response: Response with rendered content
        
        Yields:
            Items and follow-up requests
        """
        # Use base implementation for main parsing
        yield from super().parse_page(response)
        
        # Handle infinite scroll or load more buttons
        if self._has_infinite_scroll(response):
            logger.info(f"Detected infinite scroll on {response.url}")
            # Note: Actual infinite scroll handling would require 
            # more complex Playwright interaction
    
    def follow_links(self, response: Response) -> Generator:
        """
        Follow links with Playwright support for discovered pages
        
        Args:
            response: Scrapy Response object
        
        Yields:
            New Playwright-enabled requests
        """
        current_depth = response.meta.get('depth', 0)
        current_domain = urlparse(response.url).netloc
        
        # Extract all links
        links = response.css('a::attr(href)').getall()
        
        # Also look for JavaScript-based navigation
        onclick_links = response.css('*[onclick*="location.href"]').getall()
        for onclick in onclick_links:
            import re
            href_match = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick)
            if href_match:
                links.append(href_match.group(1))
        
        for link in links:
            absolute_url = response.urljoin(link)
            
            if not absolute_url.startswith(('http://', 'https://')):
                continue
            
            if absolute_url in self.visited_urls:
                continue
            
            link_domain = urlparse(absolute_url).netloc
            
            # Decide whether to follow
            should_follow = False
            
            if link_domain == current_domain:
                should_follow = True
            elif self.follow_external:
                should_follow = self._is_relevant_external_link(absolute_url, link_domain)
            
            if should_follow:
                self.visited_urls.add(absolute_url)
                
                # Determine if the new page needs Playwright
                needs_playwright = self._needs_playwright(link_domain)
                
                if needs_playwright:
                    yield self.make_playwright_request(
                        url=absolute_url,
                        callback=self.parse_page,
                        meta={
                            'depth': current_depth + 1,
                            'parent_url': response.url,
                        }
                    )
                else:
                    # Use regular request for simple pages
                    yield self.make_request(
                        url=absolute_url,
                        callback=self.parse_page,
                        meta={
                            'depth': current_depth + 1,
                            'parent_url': response.url,
                        }
                    )
    
    def _needs_playwright(self, domain: str) -> bool:
        """
        Determine if a domain needs Playwright rendering
        
        Args:
            domain: Domain to check
        
        Returns:
            True if Playwright is recommended
        """
        # Known JavaScript-heavy domains
        js_domains = {
            'cirplus.com', 'linkedin.com', 'xing.com',
            'app.', 'portal.', 'dashboard.'
        }
        
        return any(js_domain in domain.lower() for js_domain in js_domains)
    
    def _has_infinite_scroll(self, response: Response) -> bool:
        """
        Check if page has infinite scroll
        
        Args:
            response: Scrapy Response
        
        Returns:
            True if infinite scroll is detected
        """
        # Look for common infinite scroll indicators
        indicators = [
            'infinite-scroll',
            'InfiniteScroll',
            'load-more',
            'loadMore',
            'show-more',
            'showMore',
        ]
        
        page_text = response.text[:10000]  # Check first 10k chars
        
        return any(indicator in page_text for indicator in indicators)