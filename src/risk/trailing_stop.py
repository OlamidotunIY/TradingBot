"""
Trailing Stop System

Implements dynamic stop loss management:
- ATR-based trailing stops
- Breakeven stops
- Partial position closing
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np
import logging

logger = logging.getLogger('trading_bot')


@dataclass
class TrailingStopState:
    """State for a single position's trailing stop."""
    ticket: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_price: float
    initial_stop: float
    current_atr: float
    highest_price: float  # For BUY
    lowest_price: float   # For SELL
    is_breakeven: bool = False
    partial_closed: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TrailingStopManager:
    """
    Manages trailing stops for all positions.

    Features:
    1. ATR-based trailing (stop follows price at 2×ATR distance)
    2. Breakeven after +20 pips
    3. Partial close at +30 pips (close 50%)
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        breakeven_pips: float = 20,
        partial_close_pips: float = 30,
        partial_close_pct: float = 0.5,
        min_trail_pips: float = 10
    ):
        self.atr_multiplier = atr_multiplier
        self.breakeven_pips = breakeven_pips
        self.partial_close_pips = partial_close_pips
        self.partial_close_pct = partial_close_pct
        self.min_trail_pips = min_trail_pips

        self.positions: Dict[int, TrailingStopState] = {}

    def add_position(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        atr_pips: float
    ) -> TrailingStopState:
        """Add a new position to track."""
        initial_stop = self._calculate_initial_stop(
            direction, entry_price, atr_pips
        )

        state = TrailingStopState(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_price=initial_stop,
            initial_stop=initial_stop,
            current_atr=atr_pips,
            highest_price=entry_price if direction == 'BUY' else 0,
            lowest_price=entry_price if direction == 'SELL' else 999999
        )

        self.positions[ticket] = state
        logger.info(f"Trailing stop added for {ticket}: stop at {initial_stop:.5f}")
        return state

    def _calculate_initial_stop(
        self,
        direction: str,
        entry_price: float,
        atr_pips: float
    ) -> float:
        """Calculate initial stop loss."""
        stop_distance = (atr_pips * self.atr_multiplier) / 10000

        if direction == 'BUY':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def update(
        self,
        ticket: int,
        current_price: float,
        current_atr: Optional[float] = None
    ) -> Dict:
        """
        Update trailing stop for a position.

        Returns:
            {
                'should_close': bool,
                'should_partial_close': bool,
                'new_stop': float,
                'reason': str
            }
        """
        if ticket not in self.positions:
            return {'should_close': False, 'should_partial_close': False}

        state = self.positions[ticket]
        result = {
            'should_close': False,
            'should_partial_close': False,
            'new_stop': state.stop_price,
            'reason': None
        }

        # Update ATR if provided
        if current_atr:
            state.current_atr = current_atr

        # Calculate current profit in pips
        if state.direction == 'BUY':
            profit_pips = (current_price - state.entry_price) * 10000
            state.highest_price = max(state.highest_price, current_price)
        else:
            profit_pips = (state.entry_price - current_price) * 10000
            state.lowest_price = min(state.lowest_price, current_price)

        # Check stop hit
        if state.direction == 'BUY' and current_price <= state.stop_price:
            result['should_close'] = True
            result['reason'] = f"Stop hit ({profit_pips:.1f} pips)"
            return result
        elif state.direction == 'SELL' and current_price >= state.stop_price:
            result['should_close'] = True
            result['reason'] = f"Stop hit ({profit_pips:.1f} pips)"
            return result

        # Check partial close (once)
        if not state.partial_closed and profit_pips >= self.partial_close_pips:
            result['should_partial_close'] = True
            result['partial_pct'] = self.partial_close_pct
            result['reason'] = f"Partial close at +{profit_pips:.1f} pips"
            state.partial_closed = True

        # Move to breakeven
        if not state.is_breakeven and profit_pips >= self.breakeven_pips:
            state.stop_price = state.entry_price
            state.is_breakeven = True
            result['new_stop'] = state.stop_price
            result['reason'] = f"Moved to breakeven at +{profit_pips:.1f} pips"
            logger.info(f"Position {ticket}: Stop moved to breakeven")

        # Trail stop (only after breakeven)
        if state.is_breakeven:
            trail_distance = (state.current_atr * self.atr_multiplier) / 10000
            trail_distance = max(trail_distance, self.min_trail_pips / 10000)

            if state.direction == 'BUY':
                new_stop = state.highest_price - trail_distance
                if new_stop > state.stop_price:
                    state.stop_price = new_stop
                    result['new_stop'] = new_stop
            else:
                new_stop = state.lowest_price + trail_distance
                if new_stop < state.stop_price:
                    state.stop_price = new_stop
                    result['new_stop'] = new_stop

        return result

    def remove_position(self, ticket: int) -> None:
        """Remove a closed position."""
        if ticket in self.positions:
            del self.positions[ticket]

    def get_stop_price(self, ticket: int) -> Optional[float]:
        """Get current stop price for a position."""
        if ticket in self.positions:
            return self.positions[ticket].stop_price
        return None

    def get_all_positions(self) -> List[TrailingStopState]:
        """Get all tracked positions."""
        return list(self.positions.values())


def calculate_atr_pips(df, period: int = 14) -> float:
    """Calculate current ATR in pips."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )

    atr = tr.rolling(period).mean().iloc[-1]
    return atr * 10000  # Convert to pips
