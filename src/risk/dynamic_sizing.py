"""
Dynamic Position Sizing

Implements professional position sizing strategies:
- Kelly Criterion for optimal bet sizing
- ATR-based volatility adjustment
- Maximum position limits
"""
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger('trading_bot')


class DynamicPositionSizer:
    """
    Calculates optimal position size based on:
    1. Account balance and risk percentage
    2. Current market volatility (ATR)
    3. Kelly Criterion for edge optimization
    4. Regime-based adjustments
    """

    def __init__(
        self,
        base_risk_pct: float = 0.02,
        max_risk_pct: float = 0.04,
        min_risk_pct: float = 0.005,
        max_lot_size: float = 10.0,
        min_lot_size: float = 0.01
    ):
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.max_lot_size = max_lot_size
        self.min_lot_size = min_lot_size

        # Historical performance for Kelly
        self.trade_history = []

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion optimal betting fraction.

        Kelly % = W - (1-W)/R
        Where:
            W = Win probability
            R = Win/Loss ratio
        """
        if avg_loss == 0 or win_rate <= 0:
            return self.base_risk_pct

        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Use half-Kelly for safety (common practice)
        half_kelly = kelly / 2

        # Clamp to reasonable bounds
        return max(self.min_risk_pct, min(self.max_risk_pct, half_kelly))

    def calculate_atr_adjusted_size(
        self,
        balance: float,
        atr_pips: float,
        risk_pct: float,
        pip_value: float = 10.0
    ) -> float:
        """
        Calculate position size based on ATR volatility.

        Size = (Balance × Risk%) / (ATR × 2 × Pip Value)

        Uses 2×ATR as expected stop distance.
        """
        if atr_pips <= 0:
            return self.min_lot_size

        risk_amount = balance * risk_pct
        stop_pips = atr_pips * 2  # 2×ATR stop
        lot_size = risk_amount / (stop_pips * pip_value)

        return round(max(self.min_lot_size, min(self.max_lot_size, lot_size)), 2)

    def get_regime_multiplier(self, regime: str) -> float:
        """
        Adjust position size based on market regime.
        """
        multipliers = {
            'trending_up': 1.2,      # Increase in strong trends
            'trending_down': 1.2,
            'ranging': 0.8,          # Reduce in ranging markets
            'high_volatility': 0.5,  # Significantly reduce in high vol
        }
        return multipliers.get(regime, 1.0)

    def get_confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust position size based on signal confidence.
        """
        # Scale from 0.5x at 75% confidence to 1.5x at 95%+ confidence
        if confidence < 0.75:
            return 0.5
        elif confidence > 0.95:
            return 1.5
        else:
            # Linear interpolation
            return 0.5 + (confidence - 0.75) * 5  # 0.75 -> 0.5, 0.95 -> 1.5

    def calculate_position_size(
        self,
        balance: float,
        atr_pips: float,
        confidence: float,
        regime: str = 'ranging',
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        pip_value: float = 10.0
    ) -> Dict[str, float]:
        """
        Calculate optimal position size with all adjustments.

        Returns:
            Dictionary with lot_size, risk_pct, stop_pips, and adjustments
        """
        # Start with base or Kelly-adjusted risk
        if win_rate and avg_win and avg_loss:
            risk_pct = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        else:
            risk_pct = self.base_risk_pct

        # Apply regime multiplier
        regime_mult = self.get_regime_multiplier(regime)
        risk_pct *= regime_mult

        # Apply confidence multiplier
        conf_mult = self.get_confidence_multiplier(confidence)
        risk_pct *= conf_mult

        # Clamp final risk
        risk_pct = max(self.min_risk_pct, min(self.max_risk_pct, risk_pct))

        # Calculate ATR-adjusted lot size
        lot_size = self.calculate_atr_adjusted_size(
            balance, atr_pips, risk_pct, pip_value
        )

        # Stop distance
        stop_pips = atr_pips * 2

        return {
            'lot_size': lot_size,
            'risk_pct': risk_pct,
            'stop_pips': stop_pips,
            'regime_multiplier': regime_mult,
            'confidence_multiplier': conf_mult,
            'risk_amount': balance * risk_pct
        }

    def update_history(self, pnl: float, is_win: bool) -> None:
        """Record trade result for Kelly calculation."""
        self.trade_history.append({
            'pnl': pnl,
            'is_win': is_win
        })

        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics from history."""
        if len(self.trade_history) < 10:
            return {}

        wins = [t for t in self.trade_history if t['is_win']]
        losses = [t for t in self.trade_history if not t['is_win']]

        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': avg_win * win_rate / (avg_loss * (1 - win_rate)) if avg_loss > 0 else 0
        }


def calculate_atr(df, period: int = 14) -> float:
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
