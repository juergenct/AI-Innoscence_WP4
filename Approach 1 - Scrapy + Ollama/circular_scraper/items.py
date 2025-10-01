# circular_scraper/items.py
"""
Data models for scraped content
Defines the structure of data we collect from each website
"""

import scrapy
from datetime import datetime
from typing import List, Dict, Optional


class CircularEconomyItem(scrapy.Item):
    """Main data model for scraped entities"""
    
    # Basic metadata
    url = scrapy.Field()
    domain = scrapy.Field()
    scraped_at = scrapy.Field()
    crawl_depth = scrapy.Field()
    parent_url = scrapy.Field()
    
    # Page information
    title = scrapy.Field()
    meta_description = scrapy.Field()
    language = scrapy.Field()
    page_type = scrapy.Field()  # 'static' or 'dynamic'
    
    # Extracted content
    raw_html = scrapy.Field()
    extracted_text = scrapy.Field()
    main_content = scrapy.Field()  # Clean text from main content area
    structured_data = scrapy.Field()  # JSON-LD, microdata if present
    
    # Links and relationships
    internal_links = scrapy.Field()
    external_links = scrapy.Field()
    contact_links = scrapy.Field()  # Email, phone, social media
    
    # Organization information (extracted from content)
    organization_name = scrapy.Field()
    organization_type = scrapy.Field()
    address = scrapy.Field()
    city = scrapy.Field()
    postal_code = scrapy.Field()
    country = scrapy.Field()

    # Entity grouping
    entity_id = scrapy.Field()         # Stable grouping key (e.g., 'tuhh.de/logu' or 'liqtra.de')
    entity_name = scrapy.Field()       # Human-friendly entity name
    entity_root_url = scrapy.Field()   # Root URL used to scope crawling within the entity
    
    # Contact information
    emails = scrapy.Field()
    phone_numbers = scrapy.Field()
    social_media = scrapy.Field()
    
    # Content indicators (for later LLM analysis)
    keywords = scrapy.Field()
    topics = scrapy.Field()
    has_circular_economy_terms = scrapy.Field()
    has_hamburg_reference = scrapy.Field()
    
    # Technical metadata
    response_status = scrapy.Field()
    response_time = scrapy.Field()
    content_length = scrapy.Field()
    encoding = scrapy.Field()
    
    # Error handling
    error_message = scrapy.Field()
    retry_count = scrapy.Field()
    
    def __repr__(self):
        """Clean representation for logging"""
        return f"<CircularEconomyItem: {self.get('url', 'No URL')}>"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            key: value for key, value in self._values.items()
            if value is not None
        }
    
    def get_export_dict(self) -> Dict:
        """Get a simplified dictionary for CSV/Parquet export"""
        return {
            'url': self.get('url'),
            'domain': self.get('domain'),
            'entity_id': self.get('entity_id'),
            'entity_name': self.get('entity_name'),
            'entity_root_url': self.get('entity_root_url'),
            'title': self.get('title'),
            'organization_name': self.get('organization_name'),
            'city': self.get('city'),
            'language': self.get('language'),
            'has_circular_economy_terms': self.get('has_circular_economy_terms'),
            'has_hamburg_reference': self.get('has_hamburg_reference'),
            'scraped_at': self.get('scraped_at'),
            'crawl_depth': self.get('crawl_depth'),
            'external_links_count': len(self.get('external_links', [])),
            'internal_links_count': len(self.get('internal_links', [])),
            'content_length': len(self.get('extracted_text', '')),
            'emails': ';'.join(self.get('emails', [])),
            'phone_numbers': ';'.join(self.get('phone_numbers', []))
        }


class LinkItem(scrapy.Item):
    """Model for tracking links between entities"""
    
    source_url = scrapy.Field()
    source_domain = scrapy.Field()
    target_url = scrapy.Field()
    target_domain = scrapy.Field()
    link_text = scrapy.Field()
    link_context = scrapy.Field()  # Surrounding text
    link_type = scrapy.Field()  # 'navigation', 'content', 'footer', etc.
    found_at = scrapy.Field()
    crawl_depth = scrapy.Field()
    
    def __repr__(self):
        return f"<LinkItem: {self.get('source_domain')} -> {self.get('target_domain')}>"


class ErrorItem(scrapy.Item):
    """Model for tracking scraping errors"""
    
    url = scrapy.Field()
    domain = scrapy.Field()
    error_type = scrapy.Field()
    error_message = scrapy.Field()
    error_traceback = scrapy.Field()
    occurred_at = scrapy.Field()
    retry_count = scrapy.Field()
    parent_url = scrapy.Field()
    crawl_depth = scrapy.Field()
    
    def __repr__(self):
        return f"<ErrorItem: {self.get('url')} - {self.get('error_type')}>"