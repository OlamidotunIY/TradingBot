"""
Session Filter

Filters trading based on market sessions and timing:
- London session (best for GBP pairs)
- New York session (best for USD pairs)
- Avoid low-liquidity periods
"""
from datetime import datetime, time
from typing import Tuple, Optional
import logging

logger = logging.getLogger('trading_bot')


class SessionFilter:
    """
    Filters trades based on optimal trading times.

    Best times to trade:
    - London: 08:00-16:00 UTC (highest GBP volume)
    - New York: 13:00-21:00 UTC (highest USD volume)
    - London-NY overlap: 13:00-16:00 UTC (best liquidity)

    Avoid:
    - Asian session for major pairs (low volume)
    - Friday after 17:00 UTC (weekend gap risk)
    - Sunday open (gaps and low liquidity)
    """

    # Session times in UTC
    SESSIONS = {
        'sydney': (time(22, 0), time(6, 0)),   # 22:00-06:00 UTC
        'tokyo': (time(0, 0), time(8, 0)),     # 00:00-08:00 UTC
        'london': (time(8, 0), time(16, 0)),   # 08:00-16:00 UTC
        'new_york': (time(13, 0), time(21, 0)) # 13:00-21:00 UTC
    }

    # Optimal sessions by currency
    CURRENCY_SESSIONS = {
        'GBP': ['london'],
        'EUR': ['london', 'new_york'],
        'USD': ['new_york', 'london'],
        'CHF': ['london', 'new_york'],
        'JPY': ['tokyo', 'london'],
        'AUD': ['sydney', 'tokyo'],
        'NZD': ['sydney', 'tokyo'],
        'CAD': ['new_york']
    }

    def __init__(
        self,
        allow_asian: bool = False,
        allow_friday_close: bool = False,
        overlap_bonus: bool = True
    ):
        self.allow_asian = allow_asian
        self.allow_friday_close = allow_friday_close
        self.overlap_bonus = overlap_bonus

    def _is_time_in_session(self, t: time, session_start: time, session_end: time) -> bool:
        """Check if time is within session (handles overnight sessions)."""
        if session_start <= session_end:
            return session_start <= t <= session_end
        else:
            # Overnight session (e.g., Sydney: 22:00-06:00)
            return t >= session_start or t <= session_end

    def get_active_sessions(self, dt: Optional[datetime] = None) -> list:
        """Get currently active trading sessions."""
        if dt is None:
            dt = datetime.utcnow()

        current_time = dt.time()
        active = []

        for session_name, (start, end) in self.SESSIONS.items():
            if self._is_time_in_session(current_time, start, end):
                active.append(session_name)

        return active

    def is_london_ny_overlap(self, dt: Optional[datetime] = None) -> bool:
        """Check if we're in the London-NY overlap (best liquidity)."""
        if dt is None:
            dt = datetime.utcnow()

        t = dt.time()
        overlap_start = time(13, 0)
        overlap_end = time(16, 0)

        return overlap_start <= t <= overlap_end

    def should_trade(
        self,
        symbol: str,
        dt: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if we should trade this symbol at this time.

        Returns:
            (should_trade, reason)
        """
        if dt is None:
            dt = datetime.utcnow()

        # Don't trade on weekends
        if dt.weekday() == 6:  # Sunday
            return False, "Sunday - market closed"
        if dt.weekday() == 5:  # Saturday
            return False, "Saturday - market closed"

        # Avoid Friday close
        if not self.allow_friday_close:
            if dt.weekday() == 4 and dt.hour >= 17:
                return False, "Friday after 17:00 - weekend gap risk"

        # Get active sessions
        active = self.get_active_sessions(dt)

        if not active:
            return False, "No active sessions"

        # Check if good session for this pair's currencies
        base = symbol[:3]
        quote = symbol[3:6] if len(symbol) >= 6 else 'USD'

        good_sessions = set()
        for currency in [base, quote]:
            if currency in self.CURRENCY_SESSIONS:
                good_sessions.update(self.CURRENCY_SESSIONS[currency])

        matching = set(active) & good_sessions

        if not matching:
            if not self.allow_asian:
                return False, f"Current sessions {active} not optimal for {symbol}"

        # Bonus for overlap
        if self.overlap_bonus and self.is_london_ny_overlap(dt):
            return True, "London-NY overlap - optimal liquidity"

        return True, f"Active sessions: {', '.join(active)}"

    def get_session_quality(self, symbol: str, dt: Optional[datetime] = None) -> float:
        """
        Get a quality score (0-1) for trading this symbol now.

        1.0 = Optimal (London-NY overlap)
        0.7 = Good session
        0.4 = Acceptable
        0.0 = Poor/Avoid
        """
        if dt is None:
            dt = datetime.utcnow()

        should, reason = self.should_trade(symbol, dt)

        if not should:
            return 0.0

        if self.is_london_ny_overlap(dt):
            return 1.0

        active = self.get_active_sessions(dt)

        if 'london' in active or 'new_york' in active:
            return 0.7

        return 0.4
