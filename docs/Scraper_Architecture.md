## Circular Scraper Architecture

This document explains the structure, responsibilities, and data flow of the circular economy scraper contained in the `circular_scraper` package. It covers each module, class, and key function, and shows how data moves from a URL to your exported datasets.

### High-level Overview

- **Entrypoint**: Scrapy CLI with project settings from `scrapy.cfg` → `circular_scraper.settings`.
- **Spiders**: Two spiders tailored for different site types:
  - `StaticSpider` for regular HTML pages.
  - `DynamicSpider` for JavaScript-heavy pages using Playwright.
- **Middlewares**: Request-time user-agent rotation and dynamic decision to render pages with Playwright.
- **Pipelines**: Validation → Text extraction → Link extraction → Storage to CSV/Parquet/JSON + raw/processed artifacts.
- **Utilities**: URL management, normalization, state persistence.

Data flow: Spider makes Request → Downloader middlewares may enable Playwright → Response parsed into `CircularEconomyItem` → Pipelines enrich/validate/extract/save → Exports written under `data/`.

---

### Project Configuration

- `scrapy.cfg`
  - Points Scrapy to the project settings module.
  - Key line: `default = circular_scraper.settings`.

- `circular_scraper/settings.py`
  - Global Scrapy settings for the project.
  - Notable settings:
    - `BOT_NAME`, `SPIDER_MODULES`, `NEWSPIDER_MODULE`.
    - Crawl behavior: `CONCURRENT_REQUESTS`, `CONCURRENT_REQUESTS_PER_DOMAIN`, `DOWNLOAD_DELAY`, `DEPTH_LIMIT`, `DOWNLOAD_TIMEOUT`, `AUTOTHROTTLE_*`.
    - Middlewares: enables `RotateUserAgentMiddleware`, `SmartRenderMiddleware`, and Scrapy defaults; uses `RetryMiddleware`.
    - Download handlers for Playwright via `DOWNLOAD_HANDLERS` for `http/https`.
    - Reactors: `TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'` (required by scrapy-playwright).
    - Playwright: browser type and launch options (headless, sandbox flags, etc.).
    - Pipelines order: `ValidationPipeline` → `TextExtractionPipeline` → `DataStoragePipeline` → `LinkExtractionPipeline`.
    - Output & dirs: `FEED_EXPORT_*`, and pre-creates `data/raw`, `data/processed`, `data/exports`.

---

### Items

- `circular_scraper/items.py`
  - `CircularEconomyItem` (primary payload)
    - Metadata: `url`, `domain`, `scraped_at`, `crawl_depth`, `parent_url`.
    - Page info: `title`, `meta_description`, `language`, `page_type`.
    - Content: `raw_html`, `extracted_text`, `main_content`, `structured_data`.
    - Links/contacts: `internal_links`, `external_links`, `emails`, `phone_numbers`, `social_media`.
    - Indicators: `keywords`, `has_circular_economy_terms`, `has_hamburg_reference`.
    - Tech: `response_status`, `response_time`, `content_length`, `encoding`.
    - Errors: `error_message`, `retry_count`.
    - Helpers: `to_dict()`, `get_export_dict()` for compact exports.
  - `LinkItem`
    - Captures link relations between pages (source/target, text, context, type).
  - `ErrorItem`
    - Records request/processing errors with traceback and context.

---

### Spiders

- `circular_scraper/spiders/base_spider.py` → `BaseCircularEconomySpider`
  - Common configuration and logic shared by all spiders.
  - Init parameters (via `-a`):
    - `start_url` or `seed_file` (CSV with `url`/`URL`/`website` columns).
    - `max_depth` (default 3), `follow_external` (default true).
  - State tracking: `visited_urls`, `domain_counts`, basic `stats`.
  - Startup:
    - `_load_start_urls`: from args, CSV, or default TUHH URLs.
    - `start_requests`: seeds initial `Request` objects with `depth`, `parent_url`.
  - Request factory:
    - `make_request(url, callback, meta, priority, dont_filter)` ensures consistent metadata, error handling, and accounting.
  - Core parsing:
    - `parse` (abstract) to be implemented by subclasses.
    - `parse_page(response)` builds a `CircularEconomyItem`:
      - Basics: URL/domain/time/depth/parent/status/encoding.
      - Content: `raw_html` (`response.text`), `content_length`, `page_type` (static/dynamic), `title`, `meta_description`, rough `language` from HTML/meta.
      - Structured data: collect and JSON-decode JSON-LD blocks.
      - Emits the item, then, if depth allows, `follow_links`.
  - Link discovery:
    - `follow_links(response)` collects `a::attr(href)` and decides whether to follow internal links always, and external links if relevant (`_is_relevant_external_link`).
  - Error handling:
    - `handle_error(failure)` returns an `ErrorItem` with context.
  - Shutdown:
    - `closed(reason)` writes a per-run stats JSON to `data/exports/spider_stats_<name>_<timestamp>.json` and logs summary.

- `circular_scraper/spiders/static_spider.py` → `StaticSpider`
  - Target: standard HTML sites without heavy JS.
  - Overrides `parse` to warn if a page seems JS-driven and then delegates to `parse_page`.
  - Domain-specific hints:
    - `_parse_university_site`, `_parse_corporate_site`, `_parse_government_site` look for relevant links (project/team/sustainability/location/initiative) and enqueue follow-ups with adjusted priority.
  - Recent fix: CSS selectors explicitly use `::attr(href)` to avoid malformed links.

- `circular_scraper/spiders/dynamic_spider.py` → `DynamicSpider`
  - Target: JS-heavy sites; uses Playwright via request metadata.
  - `start_requests` yields Playwright-enabled requests.
  - `make_playwright_request` sets up:
    - `playwright` flags, `playwright_page_methods` (wait strategies, optional selectors per domain), `playwright_context_kwargs` (viewport, JS, locale/timezone), and an init script to auto-accept cookie banners when possible.
  - `parse` logs whether Playwright rendered and warns if content length is suspiciously short, then delegates to `parse_page`.
  - `follow_links` mirrors base behavior but can decide per-link whether to keep using Playwright.

---

### Middlewares

- `circular_scraper/middlewares.py`
  - `RotateUserAgentMiddleware`
    - Randomizes `User-Agent` header and sets realistic request headers to reduce blocking.
  - `SmartRenderMiddleware`
    - Decides when to enable Playwright based on:
      - Known JS-heavy domains and URL patterns (e.g., `/app/`, SPA indicators).
      - Retry path: if a static response looks like an SPA (e.g., low byte size + SPA markers), it re-issues the request with Playwright enabled.
    - Configures Playwright metadata: include page, waits (`networkidle`), viewport, JS enabled, HTTPS ignore.
  - `DepthLimitMiddleware`
    - Optional pattern-based per-domain depth limits (university/corporate/startup heuristics) to keep crawl bounded.

---

### Pipelines

- `circular_scraper/pipelines.py`
  - `ValidationPipeline`
    - Deduplicates by URL hash, ensures required fields (`url`), sets defaults, and records simple stats.
  - `TextExtractionPipeline`
    - Uses `trafilatura.extract(..., output_format='json')` for high-quality main content extraction.
    - Falls back to BeautifulSoup (`lxml` parser) if trafilatura returns nothing.
    - Adds metadata (description/language/organization name if available), extracts emails/phones/social links via regex, and tags circular-economy/Hamburg relevance via keyword lists.
    - Hardened to avoid NoneType concatenations and passes `url` into trafilatura (useful for context and future features).
  - `DataStoragePipeline`
    - Buffers items and writes periodically (default every 50) to:
      - CSV: `data/exports/csv/`
      - Parquet: `data/exports/parquet/`
      - JSON: `data/exports/json/` (debug/inspection)
    - Also persists:
      - Raw HTML: `data/raw/<session_id>/*.html`
      - Processed text: `data/processed/<session_id>/*.txt`
    - Writes a session summary JSON on close.
  - `LinkExtractionPipeline`
    - Parses raw HTML to extract internal/external links, tracks potential entities by heuristics, and writes a list to `data/exports/discovered_links_<date>.txt` on close.

---

### Utilities

- `circular_scraper/utils/url_manager.py` → `URLManager`
  - Normalizes URLs (lowercase, remove default ports, sort query params, drop trackers, strip fragments).
  - Tracks seen/normalized URLs and domain-level stats (counts, response times, relevance flag).
  - Decides whether a URL is allowed (skip patterns and filetypes) and computes per-domain priority (Hamburg/DE/uni/relevant keywords/penalties for slow or very large sites).
  - Persists and restores state to `data/url_manager_state.json`.

---

### Running the Scraper

- Static example:
  - `scrapy crawl static_spider -a start_url="https://example.com" -a max_depth=1`
- Dynamic example (requires Playwright browsers installed):
  - `scrapy crawl dynamic_spider -a start_url="https://app.example.com" -a max_depth=1`

Outputs are written under `data/`:
- `data/raw/<session_id>/*.html`: exact crawled HTML
- `data/processed/<session_id>/*.txt`: text summaries with metadata
- `data/exports/{csv,parquet,json}/*`: structured datasets
- `data/exports/spider_stats_<name>_<timestamp>.json`: per-run statistics
- `data/exports/discovered_links_<date>.txt`: candidates for future crawls

---

### Notes on Playwright Integration

- Ensure browsers are installed once:
  - `python -m playwright install chromium`
- The project sets `TWISTED_REACTOR` to AsyncioSelectorReactor, which is required by scrapy-playwright.
- Middleware can auto-upgrade a static request to a Playwright-enabled retry if SPA indicators are detected.

---

### Troubleshooting

- Reactor mismatch error:
  - Ensure `TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'` in settings.
- No items saved:
  - Check logs for DropItem/validation errors and verify the spider’s `parse_page` is yielding items.
- Empty `extracted_text`:
  - Verify trafilatura installation; fallback uses BeautifulSoup. Some pages require dynamic rendering.
- Playwright errors:
  - Confirm browser install and system deps. Try `--loglevel=DEBUG` for more detail.


