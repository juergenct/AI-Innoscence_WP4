# circular_scraper/utils/entity_resolver.py
"""
Entity resolution utilities
Determines which organization/entity a URL belongs to and provides
stable identifiers and root URLs for grouping scraped pages.
"""

from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass
class ResolvedEntity:
    entity_id: str
    entity_name: str
    entity_root_url: str
    domain: str


class EntityResolver:
    """Resolves URLs to entity identifiers suitable for grouping.

    Rules:
    - Default: group by registrable domain (netloc), e.g., "liqtra.de".
    - Multi-entity academic domains (e.g., tuhh.de): group by first
      path segment if it represents an institute, e.g., "tuhh.de/logu".
    """

    MULTI_ENTITY_DOMAINS = {
        'tuhh.de',
        'uni-hamburg.de',
        'haw-hamburg.de',
        'hcu-hamburg.de',
    }

    IGNORED_SEGMENTS = {'', 'en', 'de', 'tuhh', 'index.php'}

    def resolve(self, url: str, page_title: str | None = None) -> ResolvedEntity:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = (parsed.path or '/').strip('/')
        first_segment = path.split('/', 1)[0] if path else ''

        # Determine entity granularity
        if domain.endswith(tuple(self.MULTI_ENTITY_DOMAINS)) and first_segment not in self.IGNORED_SEGMENTS:
            entity_key = f"{domain}/{first_segment}"
            root_path = f"/{first_segment}"
        else:
            entity_key = domain
            root_path = '/'

        entity_root_url = f"{parsed.scheme or 'https'}://{domain}{root_path}"
        entity_name = self._guess_entity_name(domain, first_segment, page_title)

        return ResolvedEntity(
            entity_id=entity_key,
            entity_name=entity_name,
            entity_root_url=entity_root_url,
            domain=domain,
        )

    def is_same_entity(self, url_a: str, url_b: str) -> bool:
        return self.resolve(url_a).entity_id == self.resolve(url_b).entity_id

    def get_entity_root_url(self, url: str) -> str:
        return self.resolve(url).entity_root_url

    def _guess_entity_name(self, domain: str, first_segment: str, page_title: str | None) -> str:
        # Prefer institute-like segment for multi-entity domains
        if domain.endswith(('tuhh.de', 'uni-hamburg.de', 'haw-hamburg.de', 'hcu-hamburg.de')) and first_segment and first_segment not in self.IGNORED_SEGMENTS:
            return first_segment.upper()
        # Use page title prefix if available (before "-" or "|")
        if page_title:
            for sep in (' - ', ' | ', ' — '):
                if sep in page_title:
                    return page_title.split(sep)[0].strip()[:80]
            return page_title.strip()[:80]
        # Fallback: capitalize domain label
        label = domain.split('.')[0]
        return label.capitalize()


