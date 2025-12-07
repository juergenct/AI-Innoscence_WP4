from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


class Geocoder:
    def __init__(
        self,
        user_agent: str = "AI-Innoscence_Cahul_CE_Mapper/1.0",
        cache_file: Optional[Path] = None,
        timeout: int = 15,
        max_retries: int = 3
    ):
        """
        Robust geocoder with caching, retries, and fallback strategies.
        
        Args:
            user_agent: User agent for Nominatim (required by TOS)
            cache_file: Optional path to CSV cache file
            timeout: Timeout in seconds for geocoding requests
            max_retries: Maximum number of retries for failed requests
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_file = cache_file
        self.cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        
        # Load cache if available
        if cache_file and cache_file.exists():
            self._load_cache()
        
        # Initialize geolocator with proper settings
        geolocator = Nominatim(
            user_agent=user_agent,
            timeout=timeout
        )
        self._geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1.1,  # Nominatim TOS: max 1 req/sec
            max_retries=max_retries,
            error_wait_seconds=3.0
        )

    def _load_cache(self) -> None:
        """Load geocoding cache from disk."""
        try:
            df = pd.read_csv(self.cache_file)
            for _, row in df.iterrows():
                query = str(row.get('query', '')).strip()
                if query:
                    lat = row.get('lat')
                    lon = row.get('lon')
                    try:
                        lat = float(lat) if not pd.isna(lat) else None
                        lon = float(lon) if not pd.isna(lon) else None
                    except (ValueError, TypeError):
                        lat, lon = None, None
                    self.cache[query] = (lat, lon)
            self.logger.info(f"Loaded {len(self.cache)} entries from geocoding cache")
        except Exception as e:
            self.logger.warning(f"Could not load geocoding cache: {e}")

    def _save_to_cache(self, query: str, lat: Optional[float], lon: Optional[float]) -> None:
        """Save a geocoding result to cache."""
        if not self.cache_file:
            return
        
        try:
            # Update in-memory cache
            self.cache[query] = (lat, lon)
            
            # Append to file
            df = pd.DataFrame([{'query': query, 'lat': lat, 'lon': lon}])
            if self.cache_file.exists():
                df.to_csv(self.cache_file, mode='a', header=False, index=False)
            else:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.cache_file, index=False)
        except Exception as e:
            self.logger.warning(f"Could not save to geocoding cache: {e}")

    def _geocode_with_retry(self, query: str, country_codes: str = "md") -> Tuple[Optional[float], Optional[float]]:
        """Geocode with retries and proper error handling."""
        for attempt in range(1, self.max_retries + 1):
            try:
                location = self._geocode(
                    query,
                    exactly_one=True,
                    addressdetails=False,
                    country_codes=country_codes,
                    language='ro'
                )
                
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
                    
            except GeocoderTimedOut:
                if attempt < self.max_retries:
                    self.logger.debug(f"Geocoding timeout for '{query}' (attempt {attempt}/{self.max_retries}), retrying...")
                    time.sleep(2.0 * attempt)
                else:
                    self.logger.warning(f"Geocoding timeout for '{query}' after {self.max_retries} attempts")
                    return None, None
                    
            except GeocoderServiceError as e:
                self.logger.warning(f"Geocoding service error for '{query}': {e}")
                return None, None
                
            except Exception as e:
                self.logger.warning(f"Geocoding failed for '{query}': {e}")
                return None, None
        
        return None, None

    def geocode_cahul(
        self,
        address: Optional[str],
        entity_name: Optional[str] = None,
        url: Optional[str] = None,
        use_fallback: bool = True
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Geocode an address in Cahul with fallback strategies.

        Args:
            address: Physical address to geocode
            entity_name: Entity name (used as fallback if address fails)
            url: Entity URL (used to derive domain name as fallback)
            use_fallback: Whether to use fallback strategies

        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        # Normalize inputs (handle pandas NaN, None, etc.)
        if address is not None and not isinstance(address, str):
            address = str(address) if not (hasattr(address, '__float__') and pd.isna(address)) else None
        if entity_name is not None and not isinstance(entity_name, str):
            entity_name = str(entity_name) if not (hasattr(entity_name, '__float__') and pd.isna(entity_name)) else None
        
        # Strategy 1: Try exact address
        if address and address.strip():
            query = f"{address.strip()}, Cahul, Moldova"
            
            # Check cache first
            if query in self.cache:
                cached = self.cache[query]
                if cached[0] is not None and cached[1] is not None:
                    return cached
            
            # Try geocoding
            lat, lon = self._geocode_with_retry(query)
            if lat is not None and lon is not None:
                self._save_to_cache(query, lat, lon)
                return lat, lon
            
            self.logger.debug(f"Failed to geocode address: {address}")
        
        # Strategy 2: Fallback to entity name + Cahul
        if use_fallback and entity_name and entity_name.strip() and entity_name.strip().lower() not in ['unknown', 'na', 'n/a']:
            query = f"{entity_name.strip()}, Cahul, Moldova"

            # Check cache
            if query in self.cache:
                cached = self.cache[query]
                if cached[0] is not None and cached[1] is not None:
                    self.logger.debug(f"Using name-based geocoding fallback for: {entity_name}")
                    return cached

            # Try geocoding
            lat, lon = self._geocode_with_retry(query)
            if lat is not None and lon is not None:
                self._save_to_cache(query, lat, lon)
                self.logger.debug(f"Successfully geocoded by name: {entity_name}")
                return lat, lon

        # Strategy 2.5: Try URL domain name + Cahul (before generic Cahul fallback)
        if use_fallback and url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                # Extract domain name: "recyclabs.de" → "recyclabs"
                domain_name = domain.replace('www.', '').split('.')[0]
                if domain_name and domain_name.lower() not in ['unknown', 'na', 'n/a', 'localhost', '127']:
                    query = f"{domain_name}, Cahul, Moldova"

                    # Check cache
                    if query in self.cache:
                        cached = self.cache[query]
                        if cached[0] is not None and cached[1] is not None:
                            self.logger.debug(f"Using URL domain-based geocoding fallback: {domain_name}")
                            return cached

                    # Try geocoding
                    lat, lon = self._geocode_with_retry(query)
                    if lat is not None and lon is not None:
                        self._save_to_cache(query, lat, lon)
                        self.logger.debug(f"Successfully geocoded by URL domain: {domain_name}")
                        return lat, lon
            except Exception as e:
                self.logger.debug(f"Failed to extract domain from URL: {e}")

        # Strategy 3: Last resort - Cahul city center (for incomplete extractions)
        if use_fallback:
            query = "Cahul, Moldova"
            
            if query in self.cache:
                cached = self.cache[query]
                if cached[0] is not None and cached[1] is not None:
                    self.logger.debug("Using Cahul city center fallback")
                    return cached
            
            lat, lon = self._geocode_with_retry(query)
            if lat is not None and lon is not None:
                self._save_to_cache(query, lat, lon)
                self.logger.debug("Using Cahul city center fallback")
                return lat, lon
        
        # All strategies failed
        return None, None
