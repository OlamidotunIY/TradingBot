"""
RSI Strategy

A Relative Strength Index strategy:
- BUY when RSI goes from oversold to neutral
- SELL when RSI goes from overbought to neutral
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger('trading_bot')


class RSIStrategy(BaseStrategy):
    """Relative Strength Index Strategy."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize RSI Strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.overbought = self.parameters.get('overbought', 70)
        self.oversold = self.parameters.get('oversold', 30)
        self.confirmation_candles = self.parameters.get('confirmation_candles', 2)

        logger.info(f"RSI Strategy initialized: Period={self.rsi_period}, OB={self.overbought}, OS={self.oversold}")

    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate RSI indicator.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            dict: Analysis with RSI values
        """
        # Ensure we have enough data
        if len(data) < self.rsi_period + self.confirmation_candles + 1:
            return {'error': 'Insufficient data'}

        # Calculate RSI
        data = data.copy()
        data['rsi'] = ta.rsi(data['close'], length=self.rsi_period)

        # Calculate ATR for SL/TP
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

        # Get recent values for confirmation
        recent_rsi = data['rsi'].iloc[-(self.confirmation_candles + 1):].tolist()
        current = data.iloc[-1]

        # Determine zones
        current_rsi = current['rsi']
        is_overbought = current_rsi >= self.overbought
        is_oversold = current_rsi <= self.oversold
        is_neutral = not is_overbought and not is_oversold

        # Check for zone transitions
        was_oversold = any(r <= self.oversold for r in recent_rsi[:-1])
        was_overbought = any(r >= self.overbought for r in recent_rsi[:-1])

        return {
            'symbol': symbol,
            'current_price': current['close'],
            'rsi': current_rsi,
            'recent_rsi': recent_rsi,
            'atr': current['atr'],
            'is_overbought': is_overbought,
            'is_oversold': is_oversold,
            'is_neutral': is_neutral,
            'was_oversold': was_oversold,
            'was_overbought': was_overbought,
            'exit_oversold': was_oversold and is_neutral,
            'exit_overbought': was_overbought and is_neutral
        }

    def generate_signals(self, analysis: Dict[str, Any], symbol: str) -> Optional[Signal]:
        """
        Generate trading signals based on RSI.

        Args:
            analysis: Analysis results
            symbol: Trading symbol

        Returns:
            Signal or None
        """
        if 'error' in analysis:
            return None

        current_price = analysis['current_price']
        rsi = analysis['rsi']
        atr = analysis.get('atr', 0.0050)

        # BUY signal: RSI exits oversold zone
        if analysis['exit_oversold']:
            sl, tp = self.calculate_sl_tp(current_price, SignalType.BUY, atr)

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                strength=0.7,
                price=current_price,
                sl=sl,
                tp=tp,
                reason=f"RSI exited oversold zone (RSI: {rsi:.2f})"
            )

        # SELL signal: RSI exits overbought zone
        if analysis['exit_overbought']:
            sl, tp = self.calculate_sl_tp(current_price, SignalType.SELL, atr)

            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                strength=0.7,
                price=current_price,
                sl=sl,
                tp=tp,
                reason=f"RSI exited overbought zone (RSI: {rsi:.2f})"
            )

        # Determine current state for hold reason
        if analysis['is_overbought']:
            state = "Overbought"
        elif analysis['is_oversold']:
            state = "Oversold"
        else:
            state = "Neutral"

        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            reason=f"RSI: {rsi:.2f} ({state})"
        )
