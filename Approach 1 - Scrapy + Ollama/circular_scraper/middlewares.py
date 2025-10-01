# circular_scraper/middlewares.py
"""
Custom middlewares for smart scraping
Handles user-agent rotation and decides when to use JavaScript rendering
"""

import random
import logging
from urllib.parse import urlparse
from typing import Union, Optional
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.exceptions import IgnoreRequest
from fake_useragent import UserAgent
from scrapy_playwright.page import PageMethod


logger = logging.getLogger(__name__)


class RotateUserAgentMiddleware:
    """Rotate user agents to avoid detection"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.user_agents = [
            self.ua.chrome,
            self.ua.firefox,
            self.ua.safari,
            self.ua.edge,
        ]
        # Add some specific user agents that work well
        self.user_agents.extend([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ])
        logger.info(f"Initialized RotateUserAgentMiddleware with {len(self.user_agents)} user agents")
    
    def process_request(self, request, spider):
        """Assign a random user agent to each request"""
        user_agent = random.choice(self.user_agents)
        request.headers['User-Agent'] = user_agent
        
        # Add other headers to appear more legitimate
        request.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        })
        
        return None


class SmartRenderMiddleware:
    """
    Decides whether to use Playwright for JavaScript rendering
    based on domain patterns and page indicators
    """
    
    # Domains known to require JavaScript rendering
    JS_REQUIRED_DOMAINS = {
        'cirplus.com',
        'linkedin.com',
        'facebook.com',
        'instagram.com',
        'twitter.com',
        'x.com',
    }
    
    # Domains that definitely don't need JavaScript
    STATIC_DOMAINS = {
        'tuhh.de',
        'uni-hamburg.de',
        'hamburg.de',
        '.gov',
        '.edu',
    }
    
    # Patterns in URLs that suggest JavaScript is needed
    JS_URL_PATTERNS = [
        '/app/',
        '/dashboard/',
        '/portal/',
        '#/',
        'react',
        'angular',
        'vue',
    ]
    
    def __init__(self):
        self.stats = {}
        logger.info("Initialized SmartRenderMiddleware")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware instance from crawler"""
        middleware = cls()
        crawler.signals.connect(
            middleware.spider_opened,
            signal=signals.spider_opened
        )
        return middleware
    
    def spider_opened(self, spider):
        """Initialize stats when spider opens"""
        self.stats = {
            'static_requests': 0,
            'dynamic_requests': 0,
            'failed_static_converted': 0,
        }
        logger.info(f'SmartRenderMiddleware enabled for spider: {spider.name}')
    
    def should_use_playwright(self, request: Request) -> bool:
        """
        Determine if Playwright should be used for this request
        
        Returns:
            bool: True if JavaScript rendering is needed
        """
        url = request.url
        domain = urlparse(url).netloc.lower()
        
        # Check if already marked for Playwright
        if request.meta.get('playwright'):
            return True
        
        # Check known JavaScript-heavy domains
        for js_domain in self.JS_REQUIRED_DOMAINS:
            if js_domain in domain:
                logger.debug(f"Using Playwright for known JS domain: {domain}")
                return True
        
        # Check static domains (skip JavaScript)
        for static_domain in self.STATIC_DOMAINS:
            if static_domain in domain:
                logger.debug(f"Skipping Playwright for static domain: {domain}")
                return False
        
        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.JS_URL_PATTERNS:
            if pattern in url_lower:
                logger.debug(f"Using Playwright for URL pattern: {pattern}")
                return True
        
        # Check if this is a retry after static fetch failed
        if request.meta.get('retry_with_js'):
            logger.info(f"Retrying with Playwright after static fetch failed: {url}")
            return True
        
        # Default to static fetching for first attempt
        return False
    
    def process_request(self, request: Request, spider):
        """Process request and add Playwright if needed"""
        
        if self.should_use_playwright(request):
            self.stats['dynamic_requests'] += 1
            
            # Configure Playwright settings
            request.meta['playwright'] = True
            request.meta['playwright_include_page'] = True
            
            # Add page methods for better scraping
            request.meta['playwright_page_methods'] = [
                # Wait for network to be idle
                PageMethod('wait_for_load_state', 'networkidle'),
                # Wait a bit for dynamic content
                PageMethod('wait_for_timeout', 2000),
                # Screenshot for debugging (optional)
                # PageMethod('screenshot', path=f"screenshots/{urlparse(request.url).netloc}.png", full_page=True),
            ]
            
            # Additional Playwright options
            request.meta['playwright_context_kwargs'] = {
                'viewport': {'width': 1920, 'height': 1080},
                'java_script_enabled': True,
                'ignore_https_errors': True,
            }
            
            logger.debug(f"Configured Playwright for: {request.url}")
        else:
            self.stats['static_requests'] += 1
            logger.debug(f"Using static fetching for: {request.url}")
        
        return None
    
    def process_response(self, request: Request, response: Response, spider):
        """
        Check response and potentially retry with JavaScript if needed
        """
        
        # If response is too small and wasn't rendered with JS, might need JS
        if (not request.meta.get('playwright') and 
            len(response.body) < 1000 and 
            not request.meta.get('retry_with_js')):
            
            # Check for common SPA indicators in response
            indicators = [
                b'<div id="root"',
                b'<div id="app"',
                b'React.createElement',
                b'angular',
                b'vue-app',
                b'__NEXT_DATA__',
                b'window.__INITIAL_STATE__',
            ]
            
            for indicator in indicators:
                if indicator in response.body:
                    logger.info(f"Detected SPA indicator, retrying with Playwright: {request.url}")
                    self.stats['failed_static_converted'] += 1
                    
                    # Create new request with Playwright
                    new_request = request.copy()
                    new_request.meta['retry_with_js'] = True
                    new_request.dont_filter = True
                    new_request.priority = request.priority + 10
                    
                    return new_request
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Handle exceptions during request processing"""
        logger.error(f"Exception processing {request.url}: {exception}")
        return None


class DepthLimitMiddleware:
    """
    Custom depth limiting per domain
    Allows different depth limits for different types of sites
    """
    
    def __init__(self, default_depth=3):
        self.default_depth = default_depth
        self.domain_depths = {}
        
        # Special depth limits for certain domains
        self.special_depths = {
            'startup': 2,  # Startup sites usually smaller
            'university': 4,  # University sites can be deeper
            'corporate': 3,  # Corporate sites medium depth
        }
    
    def get_domain_type(self, domain: str) -> str:
        """Categorize domain type"""
        if any(x in domain for x in ['uni-', 'tuhh', '.edu', 'haw-hamburg']):
            return 'university'
        elif any(x in domain for x in ['gmbh', 'ag', 'veolia', 'corporate']):
            return 'corporate'
        else:
            return 'startup'
    
    def process_request(self, request, spider):
        """Check and enforce depth limits"""
        domain = urlparse(request.url).netloc
        depth = request.meta.get('depth', 0)
        
        # Get appropriate depth limit
        if domain not in self.domain_depths:
            domain_type = self.get_domain_type(domain)
            self.domain_depths[domain] = self.special_depths.get(
                domain_type, 
                self.default_depth
            )
        
        max_depth = self.domain_depths[domain]
        
        if depth > max_depth:
            logger.info(f"Ignoring {request.url} - exceeds depth limit {max_depth}")
            raise IgnoreRequest(f"Depth {depth} exceeds limit {max_depth}")
        
        return None