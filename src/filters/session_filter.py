"""
Session Filter

Filters trading based on market sessions and timing.
"""
from datetime import datetime, time
from typing import Tuple, Optional


class SessionFilter:
    """Filters trades based on optimal trading times."""

    SESSIONS = {
        'london': (time(8, 0), time(16, 0)),
        'new_york': (time(13, 0), time(21, 0))
    }

    def __init__(self):
        pass

    def should_trade(self, symbol: str, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """Check if we should trade at this time."""
        if dt is None:
            dt = datetime.utcnow()

        # Don't trade on weekends
        if dt.weekday() >= 5:
            return False, "Weekend"

        # Avoid Friday close
        if dt.weekday() == 4 and dt.hour >= 17:
            return False, "Friday close"

        # Check if in London or NY session
        t = dt.time()
        in_london = time(8, 0) <= t <= time(16, 0)
        in_ny = time(13, 0) <= t <= time(21, 0)

        if in_london or in_ny:
            return True, "Good session"

        return False, "Outside main sessions"
