from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout


class GoogleMapsGeocoder:
    """
    Google Maps Geocoding API client with caching and retry logic.

    API Documentation: https://developers.google.com/maps/documentation/geocoding/overview
    Pricing: $5.00 per 1,000 requests (0-100,000 requests/month)
    Rate Limit: 50 requests per second
    """

    def __init__(
        self,
        api_key: str,
        cache_file: Optional[Path] = None,
        timeout: int = 10,
        max_retries: int = 3,
        rate_limit_delay: float = 0.05  # 50 requests/second = 0.02s, using 0.05s to be safe
    ):
        """
        Initialize Google Maps Geocoder.

        Args:
            api_key: Google Maps API key
            cache_file: Optional path to CSV cache file
            timeout: Timeout in seconds for API requests
            max_retries: Maximum number of retries for failed requests
            rate_limit_delay: Delay between requests in seconds (default: 0.05s = 20 req/s)
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.cache_file = cache_file
        self.cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.last_request_time = 0

        # Google Maps Geocoding API endpoint
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"

        # Load cache if available
        if cache_file and cache_file.exists():
            self._load_cache()

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
            self.logger.info(f"Loaded {len(self.cache)} entries from Google Maps cache")
        except Exception as e:
            self.logger.warning(f"Could not load Google Maps cache: {e}")

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
            self.logger.warning(f"Could not save to Google Maps cache: {e}")

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _geocode_with_retry(self, query: str, region: str = "de") -> Tuple[Optional[float], Optional[float]]:
        """
        Geocode a query using Google Maps API with retry logic.

        Args:
            query: Address or place name to geocode
            region: Region bias (default: "de" for Germany)

        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Enforce rate limiting
                self._rate_limit()

                # Build request parameters
                params = {
                    'address': query,
                    'key': self.api_key,
                    'region': region,
                    'language': 'de'
                }

                # Make API request
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )

                # Check for HTTP errors
                response.raise_for_status()

                # Parse response
                data = response.json()
                status = data.get('status')

                if status == 'OK' and data.get('results'):
                    # Extract coordinates from first result
                    location = data['results'][0]['geometry']['location']
                    lat = location.get('lat')
                    lon = location.get('lng')

                    if lat is not None and lon is not None:
                        return float(lat), float(lon)

                elif status == 'ZERO_RESULTS':
                    # No results found - not an error, just no match
                    self.logger.debug(f"No results found for: {query}")
                    return None, None

                elif status == 'OVER_QUERY_LIMIT':
                    # Rate limit exceeded - wait and retry
                    if attempt < self.max_retries:
                        wait_time = 2.0 * attempt
                        self.logger.warning(f"Google Maps API rate limit exceeded, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(f"Google Maps API rate limit exceeded after {self.max_retries} attempts")
                        return None, None

                elif status == 'REQUEST_DENIED':
                    # API key issue
                    error_msg = data.get('error_message', 'Unknown error')
                    self.logger.error(f"Google Maps API request denied: {error_msg}")
                    return None, None

                elif status == 'INVALID_REQUEST':
                    # Invalid query
                    self.logger.warning(f"Invalid request for query: {query}")
                    return None, None

                else:
                    # Other errors
                    error_msg = data.get('error_message', status)
                    self.logger.warning(f"Google Maps API error for '{query}': {error_msg}")
                    return None, None

            except Timeout:
                if attempt < self.max_retries:
                    self.logger.debug(f"Timeout for '{query}' (attempt {attempt}/{self.max_retries}), retrying...")
                    time.sleep(1.0 * attempt)
                else:
                    self.logger.warning(f"Timeout for '{query}' after {self.max_retries} attempts")
                    return None, None

            except RequestException as e:
                self.logger.warning(f"Request error for '{query}': {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)
                else:
                    return None, None

            except Exception as e:
                self.logger.warning(f"Unexpected error geocoding '{query}': {e}")
                return None, None

        return None, None

    def geocode_address(
        self,
        address: str,
        region: str = "de"
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Geocode an address using Google Maps API.

        Args:
            address: Address to geocode
            region: Region bias (default: "de" for Germany)

        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        if not address or not address.strip():
            return None, None

        # Normalize address
        query = address.strip()

        # Check cache first
        if query in self.cache:
            cached = self.cache[query]
            if cached[0] is not None and cached[1] is not None:
                return cached

        # Geocode using API
        lat, lon = self._geocode_with_retry(query, region=region)

        # Save to cache
        if lat is not None and lon is not None:
            self._save_to_cache(query, lat, lon)

        return lat, lon

    def geocode_hamburg(
        self,
        address: Optional[str]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Geocode an address in Hamburg with proper formatting.

        Args:
            address: Physical address to geocode

        Returns:
            Tuple of (latitude, longitude) or (None, None) if failed
        """
        if not address or not address.strip():
            return None, None

        # Format address with Hamburg, Germany for better results
        formatted_address = f"{address.strip()}, Hamburg, Germany"

        return self.geocode_address(formatted_address, region="de")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        return {
            'cache_size': len(self.cache),
            'cache_hits': sum(1 for coords in self.cache.values() if coords[0] is not None)
        }
