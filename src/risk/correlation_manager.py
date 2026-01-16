"""
Correlation Manager

Manages position exposure based on currency correlations:
- Limits exposure to correlated pairs
- Tracks overall currency exposure
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger('trading_bot')


# Pre-defined correlation groups (approximate, updated quarterly)
CORRELATION_GROUPS = {
    'usd_positive': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],  # Move together
    'usd_negative': ['USDCHF', 'USDJPY', 'USDCAD'],  # Move opposite to USD positive
    'eur_correlated': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF'],
    'gbp_correlated': ['GBPUSD', 'GBPJPY', 'GBPCHF', 'EURGBP'],
    'risk_on': ['AUDUSD', 'NZDUSD', 'USDJPY'],  # Move with risk sentiment
}


class CorrelationManager:
    """
    Manages correlated positions to avoid overexposure.

    Rules:
    1. Max 2 positions in same correlation group
    2. Max exposure to any single currency (e.g., not 3 USD longs)
    3. Warns on opposing positions in correlated pairs
    """

    def __init__(
        self,
        max_correlated_positions: int = 2,
        max_currency_exposure: int = 3,
        correlation_threshold: float = 0.7
    ):
        self.max_correlated_positions = max_correlated_positions
        self.max_currency_exposure = max_currency_exposure
        self.correlation_threshold = correlation_threshold

        self.open_positions: Dict[str, str] = {}  # ticket -> symbol
        self.position_directions: Dict[str, str] = {}  # ticket -> 'BUY'/'SELL'

    def add_position(self, ticket: str, symbol: str, direction: str) -> None:
        """Track a new position."""
        self.open_positions[ticket] = symbol
        self.position_directions[ticket] = direction

    def remove_position(self, ticket: str) -> None:
        """Remove a closed position."""
        self.open_positions.pop(ticket, None)
        self.position_directions.pop(ticket, None)

    def get_currency_exposure(self) -> Dict[str, int]:
        """
        Calculate current exposure to each currency.

        Returns:
            {'USD': 2, 'GBP': 1, ...} - count of positions per currency
        """
        exposure = {}

        for ticket, symbol in self.open_positions.items():
            direction = self.position_directions[ticket]
            base = symbol[:3]
            quote = symbol[3:6] if len(symbol) >= 6 else 'USD'

            # For BUY: long base, short quote
            # For SELL: short base, long quote
            if direction == 'BUY':
                exposure[base] = exposure.get(base, 0) + 1
                exposure[quote] = exposure.get(quote, 0) - 1
            else:
                exposure[base] = exposure.get(base, 0) - 1
                exposure[quote] = exposure.get(quote, 0) + 1

        return exposure

    def get_correlation_group_count(self, symbol: str) -> int:
        """Count positions in same correlation group as symbol."""
        count = 0

        for group_name, group_symbols in CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                for pos_symbol in self.open_positions.values():
                    if pos_symbol in group_symbols:
                        count += 1

        return count

    def can_open_position(
        self,
        symbol: str,
        direction: str
    ) -> Tuple[bool, str]:
        """
        Check if we can open a new position without exceeding limits.

        Returns:
            (can_open, reason)
        """
        # Check correlation group limit
        group_count = self.get_correlation_group_count(symbol)
        if group_count >= self.max_correlated_positions:
            return False, f"Max correlated positions ({self.max_correlated_positions}) reached"

        # Check currency exposure limit
        exposure = self.get_currency_exposure()
        base = symbol[:3]
        quote = symbol[3:6] if len(symbol) >= 6 else 'USD'

        # Simulate adding this position
        if direction == 'BUY':
            new_base_exp = abs(exposure.get(base, 0) + 1)
            new_quote_exp = abs(exposure.get(quote, 0) - 1)
        else:
            new_base_exp = abs(exposure.get(base, 0) - 1)
            new_quote_exp = abs(exposure.get(quote, 0) + 1)

        if new_base_exp > self.max_currency_exposure:
            return False, f"Max {base} exposure ({self.max_currency_exposure}) would be exceeded"
        if new_quote_exp > self.max_currency_exposure:
            return False, f"Max {quote} exposure ({self.max_currency_exposure}) would be exceeded"

        # Check for opposing positions (hedge warning)
        for ticket, pos_symbol in self.open_positions.items():
            if pos_symbol == symbol:
                pos_dir = self.position_directions[ticket]
                if pos_dir != direction:
                    return False, f"Already have opposing {pos_dir} on {symbol}"

        return True, "OK"

    def get_position_summary(self) -> str:
        """Get summary of current positions and exposure."""
        if not self.open_positions:
            return "No positions"

        exposure = self.get_currency_exposure()
        symbols = list(set(self.open_positions.values()))

        return f"Positions: {len(self.open_positions)} on {symbols}. Exposure: {exposure}"

    @staticmethod
    def calculate_correlation(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        period: int = 30
    ) -> float:
        """Calculate rolling correlation between two price series."""
        returns1 = df1['close'].pct_change().dropna().tail(period)
        returns2 = df2['close'].pct_change().dropna().tail(period)

        # Align indices
        common = returns1.index.intersection(returns2.index)
        if len(common) < period // 2:
            return 0.0

        return returns1.loc[common].corr(returns2.loc[common])
