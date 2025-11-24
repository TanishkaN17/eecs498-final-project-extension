"""
Global rate limiter for coordinating API requests across all agents.
Ensures we don't exceed API rate limits regardless of coordination mode.
"""

import time
from threading import Lock
from typing import Optional

from marble.utils.logger import get_logger


class RateLimiter:
    """
    Thread-safe rate limiter that coordinates API requests across all agents.
    
    Tracks request timestamps and automatically waits when approaching rate limits.
    This ensures that even in graph mode with parallel agents, we don't exceed
    the API's rate limit.
    """

    def __init__(self, max_requests: int = 5, time_window: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds (default 60 = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: list[float] = []  # Timestamps of recent requests
        self.lock = Lock()
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(
            f"RateLimiter initialized: {max_requests} requests per {time_window} seconds"
        )

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to avoid exceeding rate limit.
        
        This method:
        1. Checks how many requests were made in the last time_window seconds
        2. If we're at the limit, waits until the oldest request expires
        3. Records this request timestamp
        
        Thread-safe: Multiple agents can call this simultaneously,
        and they'll be coordinated to stay within limits.
        """
        with self.lock:
            now = time.time()
            
            # Remove requests outside the time window
            cutoff_time = now - self.time_window
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # If we're at the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                # Calculate how long to wait (until oldest request expires)
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    self.logger.info(
                        f"Rate limit reached ({len(self.requests)}/{self.max_requests}). "
                        f"Waiting {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                    
                    # Recalculate after waiting
                    now = time.time()
                    cutoff_time = now - self.time_window
                    self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # Record this request
            self.requests.append(now)
            
            self.logger.debug(
                f"Request allowed. Current rate: {len(self.requests)}/{self.max_requests} "
                f"requests in last {self.time_window}s"
            )

    def get_status(self) -> dict[str, any]:
        """
        Get current rate limiter status.

        Returns:
            Dict with current request count and time until next available slot
        """
        with self.lock:
            now = time.time()
            cutoff_time = now - self.time_window
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            if len(self.requests) >= self.max_requests:
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
            else:
                wait_time = 0
            
            return {
                "current_requests": len(self.requests),
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "wait_time_seconds": max(0, wait_time),
                "available_slots": max(0, self.max_requests - len(self.requests))
            }

    def reset(self) -> None:
        """Reset the rate limiter (clear all request history)."""
        with self.lock:
            self.requests.clear()
            self.logger.info("Rate limiter reset")


# Global instance - shared across all agents
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(max_requests: int = 5, time_window: int = 60) -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    This ensures all agents share the same rate limiter,
    so requests are coordinated across the entire system.

    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds

    Returns:
        The global RateLimiter instance
    """
    global _global_rate_limiter
    
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(max_requests=max_requests, time_window=time_window)
    
    return _global_rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """
    Set a custom rate limiter (useful for testing).

    Args:
        limiter: RateLimiter instance to use
    """
    global _global_rate_limiter
    _global_rate_limiter = limiter

