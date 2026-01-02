from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque


class DomainRateLimiter:
    def __init__(self, max_per_second: float = 2.0):
        self.max_per_second = max_per_second
        self.lock = threading.Lock()
        self.domain_requests: dict[str, Deque[float]] = defaultdict(deque)

    def acquire(self, domain: str) -> None:
        with self.lock:
            dq = self.domain_requests[domain]
            now = time.time()
            window = 1.0
            # Drop old timestamps
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) >= self.max_per_second:
                # Need to wait until earliest expires
                sleep_time = window - (now - dq[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            dq.append(time.time())
