from __future__ import annotations

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from typing import Optional, Tuple


class Geocoder:
    def __init__(self, user_agent: str = "hamburg_ce_mapper"):
        geolocator = Nominatim(user_agent=user_agent)
        self._geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)

    def geocode_hamburg(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        if not address:
            return None, None
        try:
            location = self._geocode(f"{address}, Hamburg, Germany")
            if location:
                return location.latitude, location.longitude
        except Exception:
            return None, None
        return None, None
