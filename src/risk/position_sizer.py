"""
Position Sizer - Calculate Optimal Position Sizes

This module calculates position sizes based on risk parameters.
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger('trading_bot')


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_LOT = "fixed_lot"
    FIXED_RISK = "fixed_risk"
    KELLY_CRITERION = "kelly"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"


class PositionSizer:
    """Calculates optimal position sizes."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Position Sizer.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.default_risk = config.get('max_risk_per_trade', 0.02)
        self.min_lot = 0.01
        self.max_lot = 100.0

        logger.info("Position Sizer initialized")

    def calculate(
        self,
        method: SizingMethod,
        account_balance: float,
        entry_price: float,
        sl_price: float,
        symbol_info: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Calculate position size.

        Args:
            method: Sizing method to use
            account_balance: Current account balance
            entry_price: Entry price
            sl_price: Stop loss price
            symbol_info: Symbol information
            **kwargs: Additional parameters for specific methods

        Returns:
            float: Position size in lots
        """
        if method == SizingMethod.FIXED_LOT:
            return self._fixed_lot(kwargs.get('lot_size', 0.1))

        elif method == SizingMethod.FIXED_RISK:
            return self._fixed_risk(
                account_balance,
                entry_price,
                sl_price,
                symbol_info,
                kwargs.get('risk_percent', self.default_risk)
            )

        elif method == SizingMethod.KELLY_CRITERION:
            return self._kelly_criterion(
                account_balance,
                entry_price,
                sl_price,
                symbol_info,
                kwargs.get('win_rate', 0.5),
                kwargs.get('avg_win', 1.0),
                kwargs.get('avg_loss', 1.0)
            )

        elif method == SizingMethod.MARTINGALE:
            return self._martingale(
                kwargs.get('base_lot', 0.01),
                kwargs.get('consecutive_losses', 0),
                kwargs.get('multiplier', 2.0)
            )

        elif method == SizingMethod.ANTI_MARTINGALE:
            return self._anti_martingale(
                kwargs.get('base_lot', 0.01),
                kwargs.get('consecutive_wins', 0),
                kwargs.get('multiplier', 1.5)
            )

        return self.min_lot

    def _fixed_lot(self, lot_size: float) -> float:
        """Fixed lot size method."""
        return self._validate_lot(lot_size)

    def _fixed_risk(
        self,
        balance: float,
        entry: float,
        sl: float,
        symbol_info: Dict[str, Any],
        risk_percent: float
    ) -> float:
        """
        Calculate lot size based on fixed risk percentage.

        Args:
            balance: Account balance
            entry: Entry price
            sl: Stop loss price
            symbol_info: Symbol information
            risk_percent: Risk as decimal (0.02 = 2%)

        Returns:
            float: Lot size
        """
        # Calculate risk amount
        risk_amount = balance * risk_percent

        # Calculate pip distance
        point = symbol_info.get('point', 0.00001)
        pip_distance = abs(entry - sl) / point / 10

        if pip_distance == 0:
            return self.min_lot

        # Calculate pip value (simplified - adjust for your broker)
        contract_size = symbol_info.get('contract_size', 100000)
        pip_value = point * 10 * contract_size

        # Calculate lot size
        lot_size = risk_amount / (pip_distance * pip_value)

        return self._validate_lot(lot_size, symbol_info)

    def _kelly_criterion(
        self,
        balance: float,
        entry: float,
        sl: float,
        symbol_info: Dict[str, Any],
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate lot size using Kelly Criterion.

        Kelly % = W - [(1-W) / R]
        Where W = Win rate, R = Win/Loss ratio

        Args:
            balance: Account balance
            entry: Entry price
            sl: Stop loss price
            symbol_info: Symbol information
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            float: Lot size
        """
        if avg_loss == 0:
            return self.min_lot

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Use half-Kelly for safety
        kelly_percent = max(0, kelly_percent * 0.5)

        # Cap at max risk
        kelly_percent = min(kelly_percent, 0.05)

        # Calculate using fixed risk with Kelly percentage
        return self._fixed_risk(balance, entry, sl, symbol_info, kelly_percent)

    def _martingale(
        self,
        base_lot: float,
        consecutive_losses: int,
        multiplier: float
    ) -> float:
        """
        Martingale position sizing (increase after losses).

        WARNING: High risk strategy!

        Args:
            base_lot: Starting lot size
            consecutive_losses: Number of consecutive losses
            multiplier: Lot multiplier after each loss

        Returns:
            float: Lot size
        """
        lot_size = base_lot * (multiplier ** consecutive_losses)
        return self._validate_lot(lot_size)

    def _anti_martingale(
        self,
        base_lot: float,
        consecutive_wins: int,
        multiplier: float
    ) -> float:
        """
        Anti-Martingale position sizing (increase after wins).

        Args:
            base_lot: Starting lot size
            consecutive_wins: Number of consecutive wins
            multiplier: Lot multiplier after each win

        Returns:
            float: Lot size
        """
        lot_size = base_lot * (multiplier ** consecutive_wins)
        return self._validate_lot(lot_size)

    def _validate_lot(self, lot_size: float, symbol_info: Dict[str, Any] = None) -> float:
        """
        Validate and round lot size.

        Args:
            lot_size: Raw lot size
            symbol_info: Optional symbol info for specific limits

        Returns:
            float: Valid lot size
        """
        min_lot = self.min_lot
        max_lot = self.max_lot
        step = 0.01

        if symbol_info:
            min_lot = symbol_info.get('volume_min', self.min_lot)
            max_lot = symbol_info.get('volume_max', self.max_lot)
            step = symbol_info.get('volume_step', 0.01)

        # Clamp to limits
        lot_size = max(min_lot, min(lot_size, max_lot))

        # Round to step
        lot_size = round(lot_size / step) * step

        return round(lot_size, 2)

    def estimate_risk(
        self,
        lot_size: float,
        entry: float,
        sl: float,
        symbol_info: Dict[str, Any],
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Estimate risk for a given position size.

        Args:
            lot_size: Position size
            entry: Entry price
            sl: Stop loss price
            symbol_info: Symbol information
            account_balance: Account balance

        Returns:
            dict: Risk estimation
        """
        point = symbol_info.get('point', 0.00001)
        contract_size = symbol_info.get('contract_size', 100000)

        # Calculate pip distance and value
        pip_distance = abs(entry - sl) / point / 10
        pip_value = point * 10 * contract_size * lot_size

        # Calculate potential loss
        potential_loss = pip_distance * pip_value
        risk_percent = potential_loss / account_balance if account_balance > 0 else 0

        return {
            'lot_size': lot_size,
            'pip_distance': pip_distance,
            'pip_value': pip_value,
            'potential_loss': potential_loss,
            'risk_percent': risk_percent,
            'risk_percent_display': f"{risk_percent * 100:.2f}%"
        }
