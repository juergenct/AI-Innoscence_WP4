# circular_scraper/settings.py
"""
Scrapy settings for circular_scraper project
Optimized for scraping ~1000 websites without getting blocked
"""

import os
from pathlib import Path

BOT_NAME = 'circular_scraper'

SPIDER_MODULES = ['circular_scraper.spiders']
NEWSPIDER_MODULE = 'circular_scraper.spiders'

# Use asyncio reactor required by scrapy-playwright
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 16  # Total concurrent requests
CONCURRENT_REQUESTS_PER_DOMAIN = 4  # Max 4 requests per domain simultaneously

# Configure delays (in seconds)
DOWNLOAD_DELAY = 2  # Base delay between requests (will be randomized 0.5 * to 1.5 * DOWNLOAD_DELAY)
RANDOMIZE_DOWNLOAD_DELAY = True

# AutoThrottle for automatic adjustment of delays
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 15
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0
AUTOTHROTTLE_DEBUG = True  # Enable to see throttling stats

# Configure timeouts
DOWNLOAD_TIMEOUT = 30  # Timeout for regular requests
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000  # 30 seconds for Playwright

# Disable cookies to appear more like a regular browser
COOKIES_ENABLED = False

# Depth limit for crawling
DEPTH_LIMIT = 3  # Maximum depth to crawl from seed URLs

# User Agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Configure middlewares
DOWNLOADER_MIDDLEWARES = {
    'circular_scraper.middlewares.RotateUserAgentMiddleware': 400,
    'circular_scraper.middlewares.SmartRenderMiddleware': 500,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 550,
}

# Configure pipelines
ITEM_PIPELINES = {
    'circular_scraper.pipelines.ValidationPipeline': 100,
    'circular_scraper.pipelines.TextExtractionPipeline': 200,
    'circular_scraper.pipelines.LLMClassificationPipeline': 250,  # Real-time LLM classification
    'circular_scraper.pipelines.DataStoragePipeline': 300,
    'circular_scraper.pipelines.LinkExtractionPipeline': 400,
}

# Playwright settings
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_LAUNCH_OPTIONS = {
    "headless": True,
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled",
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
    ]
}

# Retry configuration
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Memory and performance settings
REACTOR_THREADPOOL_MAXSIZE = 20
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 4096  # Stop if using more than 4GB RAM
MEMUSAGE_WARNING_MB = 3584  # Warning at 3.5GB

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(levelname)s: %(message)s'

# Stats collection
STATS_CLASS = 'scrapy.statscollectors.MemoryStatsCollector'

# Data export settings
FEED_EXPORT_ENCODING = 'utf-8'
FEED_EXPORT_BATCH_ITEM_COUNT = 100  # Save every 100 items

# Custom settings for data storage
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXPORT_DIR = DATA_DIR / 'exports'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# LLM classification batching
LLM_CLASSIFY_BATCH_SIZE = 100
LLM_MIN_SAMPLES_PER_ENTITY = 1
LLM_CLASSIFY_FLUSH_INTERVAL_SECS = 10
LLM_TRANSLATE_TO_EN = True
LLM_TRANSLATE_MAX_CHARS = 1000
# Optional: override translation model (defaults to LLM classifier model)
LLM_TRANSLATE_MODEL = None

# Duplicate filtering
DUPEFILTER_CLASS = 'scrapy.dupefilters.RFPDupeFilter'
DUPEFILTER_DEBUG = True

# DNS settings for faster resolution
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000
DNS_RESOLVER = 'scrapy.resolver.CachingThreadedResolver'

# Request fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Extension settings
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,  # Disable telnet console
    'scrapy.extensions.memusage.MemoryUsage': 100,
    'scrapy.extensions.closespider.CloseSpider': 200,
}

# Close spider after certain conditions
CLOSESPIDER_PAGECOUNT = 100000  # Stop after 100,000 pages
CLOSESPIDER_TIMEOUT = 43200  # Stop after 12 hours