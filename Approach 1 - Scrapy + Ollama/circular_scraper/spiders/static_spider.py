# circular_scraper/spiders/static_spider.py
"""
Spider for scraping static HTML websites
Optimized for simple sites that don't require JavaScript rendering
"""

import logging
from typing import Generator
from urllib.parse import urlparse

from scrapy.http import Response

from circular_scraper.spiders.base_spider import BaseCircularEconomySpider
from circular_scraper.items import CircularEconomyItem


logger = logging.getLogger(__name__)


class StaticSpider(BaseCircularEconomySpider):
    """
    Spider optimized for static HTML content
    Uses regular HTTP requests without JavaScript rendering
    """
    
    name = 'static_spider'
    
    # Custom settings for static crawling
    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,  # Slightly faster for static sites
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,  # Can handle more concurrent requests
        'DEPTH_LIMIT': 3,
        'DOWNLOAD_TIMEOUT': 20,  # Shorter timeout for static pages
    }
    
    def parse(self, response: Response) -> Generator:
        """
        Initial parsing method for start URLs
        
        Args:
            response: Scrapy Response object
        
        Yields:
            Items and follow-up requests
        """
        # Check if we got a valid HTML response
        if response.status == 200:
            # Check if this might be a JavaScript-heavy site
            if self._needs_javascript(response):
                logger.warning(f"Site {response.url} might need JavaScript rendering. "
                             f"Consider using dynamic_spider instead.")
            
            # Parse the page
            yield from self.parse_page(response)
        else:
            logger.warning(f"Got status {response.status} for {response.url}")
            
            # Still try to parse if it's a redirect
            if 300 <= response.status < 400:
                yield from self.parse_page(response)
    
    def _needs_javascript(self, response: Response) -> bool:
        """
        Check if the page might need JavaScript rendering
        
        Args:
            response: Scrapy Response object
        
        Returns:
            True if JavaScript rendering might be needed
        """
        # Check for common SPA indicators
        indicators = [
            '<div id="root"',
            '<div id="app"',
            'React.createElement',
            'angular.module',
            'vue-app',
            '__NEXT_DATA__',
            'window.__INITIAL_STATE__',
        ]
        
        body_text = response.text[:5000]  # Check first 5000 chars
        
        for indicator in indicators:
            if indicator in body_text:
                return True
        
        # Check if body is suspiciously small
        if len(response.text) < 1000:
            # But has JavaScript files
            if response.css('script[src]').getall():
                return True
        
        return False
    
    def parse_page(self, response: Response) -> Generator:
        """
        Parse a page and extract all relevant information
        Overrides base implementation with static-specific logic
        
        Args:
            response: Scrapy Response object
        
        Yields:
            CircularEconomyItem and follow-up Requests
        """
        # Use base implementation for main parsing
        yield from super().parse_page(response)
        
        # Additional static-specific extraction can be added here
        # For example, looking for specific patterns in government sites
        
        domain = urlparse(response.url).netloc
        
        # Special handling for known site types
        if 'tuhh.de' in domain or 'uni-hamburg' in domain:
            yield from self._parse_university_site(response)
        elif any(corp in domain for corp in ['veolia', 'remondis', 'alba']):
            yield from self._parse_corporate_site(response)
        elif '.hamburg.de' in domain:
            yield from self._parse_government_site(response)
    
    def _parse_university_site(self, response: Response) -> Generator:
        """
        Special parsing for university websites
        
        Args:
            response: Scrapy Response object
        
        Yields:
            Additional items or requests specific to university sites
        """
        # Look for research project pages
        project_links = response.css(
            'a[href*="project"]::attr(href), '
            'a[href*="forschung"]::attr(href), '
            'a[href*="research"]::attr(href)'
        ).getall()
        
        for link in project_links[:10]:  # Limit to 10 project links
            absolute_url = response.urljoin(link)
            
            if absolute_url not in self.visited_urls:
                self.visited_urls.add(absolute_url)
                
                yield self.make_request(
                    url=absolute_url,
                    callback=self.parse_page,
                    meta={
                        'depth': response.meta.get('depth', 0) + 1,
                        'parent_url': response.url,
                        'site_type': 'university_project'
                    },
                    priority=5  # Higher priority for project pages
                )
        
        # Look for team/staff pages (might have contact info)
        team_links = response.css(
            'a[href*="team"]::attr(href), '
            'a[href*="staff"]::attr(href), '
            'a[href*="mitarbeiter"]::attr(href)'
        ).getall()
        
        for link in team_links[:5]:  # Limit to 5 team links
            absolute_url = response.urljoin(link)
            
            if absolute_url not in self.visited_urls:
                self.visited_urls.add(absolute_url)
                
                yield self.make_request(
                    url=absolute_url,
                    callback=self.parse_page,
                    meta={
                        'depth': response.meta.get('depth', 0) + 1,
                        'parent_url': response.url,
                        'site_type': 'university_team'
                    }
                )
    
    def _parse_corporate_site(self, response: Response) -> Generator:
        """
        Special parsing for corporate websites
        
        Args:
            response: Scrapy Response object
        
        Yields:
            Additional items or requests specific to corporate sites
        """
        # Look for sustainability/CSR pages
        sustainability_links = response.css(
            'a[href*="sustainability"]::attr(href), '
            'a[href*="nachhaltigkeit"]::attr(href), '
            'a[href*="umwelt"]::attr(href), '
            'a[href*="csr"]::attr(href), '
            'a[href*="responsibility"]::attr(href)'
        ).getall()
        
        for link in sustainability_links[:5]:
            absolute_url = response.urljoin(link)
            
            if absolute_url not in self.visited_urls:
                self.visited_urls.add(absolute_url)
                
                yield self.make_request(
                    url=absolute_url,
                    callback=self.parse_page,
                    meta={
                        'depth': response.meta.get('depth', 0) + 1,
                        'parent_url': response.url,
                        'site_type': 'corporate_sustainability'
                    },
                    priority=5
                )
        
        # Look for location/branch pages
        location_links = response.css(
            'a[href*="standort"]::attr(href), '
            'a[href*="location"]::attr(href), '
            'a[href*="hamburg"]::attr(href), '
            'a[href*="niederlassung"]::attr(href)'
        ).getall()
        
        for link in location_links[:5]:
            absolute_url = response.urljoin(link)
            
            if 'hamburg' in absolute_url.lower():  # Prioritize Hamburg locations
                if absolute_url not in self.visited_urls:
                    self.visited_urls.add(absolute_url)
                    
                    yield self.make_request(
                        url=absolute_url,
                        callback=self.parse_page,
                        meta={
                            'depth': response.meta.get('depth', 0) + 1,
                            'parent_url': response.url,
                            'site_type': 'corporate_location'
                        },
                        priority=10  # High priority for Hamburg locations
                    )
    
    def _parse_government_site(self, response: Response) -> Generator:
        """
        Special parsing for government websites
        
        Args:
            response: Scrapy Response object
        
        Yields:
            Additional items or requests specific to government sites
        """
        # Look for initiative and program pages
        initiative_links = response.css(
            'a[href*="initiative"]::attr(href), '
            'a[href*="programm"]::attr(href), '
            'a[href*="foerderung"]::attr(href), '
            'a[href*="projekt"]::attr(href)'
        ).getall()
        
        for link in initiative_links[:10]:
            absolute_url = response.urljoin(link)
            
            # Check if it's related to environment/sustainability
            if any(keyword in absolute_url.lower() for keyword in 
                   ['umwelt', 'klima', 'energie', 'abfall', 'recycling']):
                
                if absolute_url not in self.visited_urls:
                    self.visited_urls.add(absolute_url)
                    
                    yield self.make_request(
                        url=absolute_url,
                        callback=self.parse_page,
                        meta={
                            'depth': response.meta.get('depth', 0) + 1,
                            'parent_url': response.url,
                            'site_type': 'government_initiative'
                        },
                        priority=7
                    )