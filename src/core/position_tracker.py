"""
Position Tracker - Track Open Positions and P&L

This module monitors open positions and calculates real-time P&L.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .mt5_handler import MT5Handler

logger = logging.getLogger('trading_bot')


@dataclass
class Position:
    """Position data class."""
    ticket: int
    symbol: str
    type: str  # 'BUY' or 'SELL'
    volume: float
    open_price: float
    current_price: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    magic: int
    comment: str
    open_time: datetime
    strategy_name: str = ""

    @property
    def pips(self) -> float:
        """Calculate pips profit/loss."""
        if self.type == 'BUY':
            return (self.current_price - self.open_price) * 10000
        else:
            return (self.open_price - self.current_price) * 10000

    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.profit > 0


@dataclass
class PositionSummary:
    """Summary of all positions."""
    total_positions: int = 0
    total_profit: float = 0.0
    total_volume: float = 0.0
    buy_positions: int = 0
    sell_positions: int = 0
    profitable_positions: int = 0
    losing_positions: int = 0
    positions_by_symbol: Dict[str, int] = field(default_factory=dict)
    profit_by_symbol: Dict[str, float] = field(default_factory=dict)


class PositionTracker:
    """Tracks and monitors open positions."""

    def __init__(self, mt5_handler: MT5Handler):
        """
        Initialize Position Tracker.

        Args:
            mt5_handler: MT5Handler instance
        """
        self.mt5 = mt5_handler
        self._positions: Dict[int, Position] = {}
        self._last_update: Optional[datetime] = None

    def update(self) -> None:
        """Update position data from MT5."""
        raw_positions = self.mt5.get_positions()

        # Update internal position cache
        current_tickets = set()

        for pos_data in raw_positions:
            ticket = pos_data['ticket']
            current_tickets.add(ticket)

            self._positions[ticket] = Position(
                ticket=ticket,
                symbol=pos_data['symbol'],
                type=pos_data['type'],
                volume=pos_data['volume'],
                open_price=pos_data['open_price'],
                current_price=pos_data['current_price'],
                sl=pos_data['sl'],
                tp=pos_data['tp'],
                profit=pos_data['profit'],
                magic=pos_data['magic'],
                comment=pos_data['comment'],
                open_time=pos_data['open_time'],
                strategy_name=pos_data.get('comment', '')
            )

        # Remove closed positions
        closed_tickets = set(self._positions.keys()) - current_tickets
        for ticket in closed_tickets:
            del self._positions[ticket]

        self._last_update = datetime.now()

    def get_position(self, ticket: int) -> Optional[Position]:
        """
        Get a specific position by ticket.

        Args:
            ticket: Position ticket number

        Returns:
            Position or None
        """
        self.update()
        return self._positions.get(ticket)

    def get_all_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            list: List of Position objects
        """
        self.update()
        return list(self._positions.values())

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get positions for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            list: List of Position objects for the symbol
        """
        self.update()
        return [p for p in self._positions.values() if p.symbol == symbol]

    def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """
        Get positions for a specific strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            list: List of Position objects for the strategy
        """
        self.update()
        return [p for p in self._positions.values() if strategy_name in p.comment]

    def get_summary(self) -> PositionSummary:
        """
        Get summary of all positions.

        Returns:
            PositionSummary: Summary statistics
        """
        self.update()

        summary = PositionSummary()

        for position in self._positions.values():
            summary.total_positions += 1
            summary.total_profit += position.profit
            summary.total_volume += position.volume

            if position.type == 'BUY':
                summary.buy_positions += 1
            else:
                summary.sell_positions += 1

            if position.is_profitable:
                summary.profitable_positions += 1
            else:
                summary.losing_positions += 1

            # By symbol
            if position.symbol not in summary.positions_by_symbol:
                summary.positions_by_symbol[position.symbol] = 0
                summary.profit_by_symbol[position.symbol] = 0.0

            summary.positions_by_symbol[position.symbol] += 1
            summary.profit_by_symbol[position.symbol] += position.profit

        return summary

    def get_total_profit(self) -> float:
        """Get total profit across all positions."""
        return self.get_summary().total_profit

    def get_position_count(self, symbol: Optional[str] = None) -> int:
        """
        Get count of open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            int: Number of positions
        """
        if symbol:
            return len(self.get_positions_by_symbol(symbol))
        return len(self.get_all_positions())

    def has_position(self, symbol: str, position_type: Optional[str] = None) -> bool:
        """
        Check if a position exists for a symbol.

        Args:
            symbol: Trading symbol
            position_type: Optional type filter ('BUY' or 'SELL')

        Returns:
            bool: True if position exists
        """
        positions = self.get_positions_by_symbol(symbol)

        if position_type:
            positions = [p for p in positions if p.type == position_type]

        return len(positions) > 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Export position data as dictionary.

        Returns:
            dict: Position data
        """
        self.update()

        return {
            'positions': [
                {
                    'ticket': p.ticket,
                    'symbol': p.symbol,
                    'type': p.type,
                    'volume': p.volume,
                    'open_price': p.open_price,
                    'current_price': p.current_price,
                    'sl': p.sl,
                    'tp': p.tp,
                    'profit': p.profit,
                    'pips': p.pips,
                    'open_time': p.open_time.isoformat(),
                    'strategy': p.strategy_name
                }
                for p in self._positions.values()
            ],
            'summary': {
                'total': len(self._positions),
                'total_profit': self.get_total_profit(),
                'last_update': self._last_update.isoformat() if self._last_update else None
            }
        }
