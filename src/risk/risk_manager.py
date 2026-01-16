"""
Risk Manager - Risk Controls and Validation

This module enforces risk management rules before trade execution.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, date
import logging

logger = logging.getLogger('trading_bot')


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    approved: bool
    reason: str = ""
    max_volume: Optional[float] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiskManager:
    """Manages risk controls and validates trades."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Risk Manager.

        Args:
            config: Risk configuration from config.yaml
        """
        self.config = config

        # Risk limits
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)
        self.max_open_positions = config.get('max_open_positions', 5)
        self.max_positions_per_symbol = config.get('max_positions_per_symbol', 2)

        # Default SL/TP
        self.default_sl_pips = config.get('default_sl_pips', 50)
        self.default_tp_pips = config.get('default_tp_pips', 100)

        # Trailing stop
        self.trailing_stop = config.get('trailing_stop', False)
        self.trailing_stop_pips = config.get('trailing_stop_pips', 30)

        # Daily tracking
        self._daily_pnl: Dict[date, float] = {}
        self._daily_starting_balance: Dict[date, float] = {}

        logger.info("Risk Manager initialized")

    def check_trade(
        self,
        symbol: str,
        volume: float,
        current_positions: List[Dict[str, Any]],
        account_balance: float,
        account_equity: float
    ) -> RiskCheckResult:
        """
        Check if a trade passes all risk rules.

        Args:
            symbol: Trading symbol
            volume: Proposed volume
            current_positions: List of current open positions
            account_balance: Current account balance
            account_equity: Current account equity

        Returns:
            RiskCheckResult: Result of risk checks
        """
        warnings = []

        # Check max open positions
        if len(current_positions) >= self.max_open_positions:
            return RiskCheckResult(
                approved=False,
                reason=f"Max open positions ({self.max_open_positions}) reached"
            )

        # Check positions per symbol
        symbol_positions = [p for p in current_positions if p.get('symbol') == symbol]
        if len(symbol_positions) >= self.max_positions_per_symbol:
            return RiskCheckResult(
                approved=False,
                reason=f"Max positions for {symbol} ({self.max_positions_per_symbol}) reached"
            )

        # Check daily drawdown
        today = date.today()
        if today not in self._daily_starting_balance:
            self._daily_starting_balance[today] = account_balance
            self._daily_pnl[today] = 0.0

        starting_balance = self._daily_starting_balance[today]
        current_drawdown = (starting_balance - account_equity) / starting_balance

        if current_drawdown >= self.max_daily_drawdown:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily drawdown limit ({self.max_daily_drawdown * 100:.1f}%) reached"
            )

        # Warn if approaching drawdown limit
        if current_drawdown >= self.max_daily_drawdown * 0.7:
            warnings.append(f"Approaching daily drawdown limit ({current_drawdown * 100:.1f}%)")

        # Calculate max allowed volume based on risk
        max_volume = self._calculate_max_volume(account_balance)
        if volume > max_volume:
            warnings.append(f"Volume adjusted from {volume} to {max_volume}")
            volume = max_volume

        return RiskCheckResult(
            approved=True,
            reason="All risk checks passed",
            max_volume=max_volume,
            warnings=warnings
        )

    def _calculate_max_volume(self, account_balance: float) -> float:
        """
        Calculate maximum allowed volume.

        Args:
            account_balance: Account balance

        Returns:
            float: Maximum volume
        """
        # Simplified calculation - adjust based on your broker
        risk_amount = account_balance * self.max_risk_per_trade
        pips_risk = self.default_sl_pips

        # Rough calculation: $10 per pip per lot
        max_volume = risk_amount / (pips_risk * 10)

        # Round and ensure minimum
        return max(0.01, round(max_volume, 2))

    def validate_sl_tp(
        self,
        order_type: str,
        entry_price: float,
        sl: Optional[float],
        tp: Optional[float],
        symbol_info: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        """
        Validate and adjust SL/TP levels.

        Args:
            order_type: 'BUY' or 'SELL'
            entry_price: Entry price
            sl: Stop loss price
            tp: Take profit price
            symbol_info: Symbol information

        Returns:
            dict: Validated SL and TP
        """
        point = symbol_info.get('point', 0.00001)

        # Calculate default SL/TP if not provided
        if sl is None:
            if order_type == 'BUY':
                sl = entry_price - (self.default_sl_pips * point * 10)
            else:
                sl = entry_price + (self.default_sl_pips * point * 10)

        if tp is None:
            if order_type == 'BUY':
                tp = entry_price + (self.default_tp_pips * point * 10)
            else:
                tp = entry_price - (self.default_tp_pips * point * 10)

        # Validate SL is on correct side
        if order_type == 'BUY' and sl >= entry_price:
            sl = entry_price - (self.default_sl_pips * point * 10)
            logger.warning("Adjusted invalid SL for BUY order")

        if order_type == 'SELL' and sl <= entry_price:
            sl = entry_price + (self.default_sl_pips * point * 10)
            logger.warning("Adjusted invalid SL for SELL order")

        return {
            'sl': round(sl, symbol_info.get('digits', 5)),
            'tp': round(tp, symbol_info.get('digits', 5))
        }

    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L tracking.

        Args:
            pnl: P&L amount to add
        """
        today = date.today()
        if today not in self._daily_pnl:
            self._daily_pnl[today] = 0.0

        self._daily_pnl[today] += pnl

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily trading statistics."""
        today = date.today()

        return {
            'date': today.isoformat(),
            'starting_balance': self._daily_starting_balance.get(today, 0),
            'daily_pnl': self._daily_pnl.get(today, 0),
            'max_daily_drawdown': self.max_daily_drawdown,
            'max_risk_per_trade': self.max_risk_per_trade
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of trading day)."""
        today = date.today()
        self._daily_pnl[today] = 0.0
        logger.info("Daily statistics reset")

    def get_risk_settings(self) -> Dict[str, Any]:
        """Get current risk settings."""
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_drawdown': self.max_daily_drawdown,
            'max_open_positions': self.max_open_positions,
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'default_sl_pips': self.default_sl_pips,
            'default_tp_pips': self.default_tp_pips,
            'trailing_stop': self.trailing_stop,
            'trailing_stop_pips': self.trailing_stop_pips
        }
