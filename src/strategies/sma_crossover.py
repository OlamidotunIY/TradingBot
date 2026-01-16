"""
SMA Crossover Strategy

A simple moving average crossover strategy:
- BUY when fast SMA crosses above slow SMA
- SELL when fast SMA crosses below slow SMA
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger('trading_bot')


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize SMA Crossover Strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)

        # Strategy parameters
        self.fast_period = self.parameters.get('fast_period', 10)
        self.slow_period = self.parameters.get('slow_period', 20)
        self.signal_threshold = self.parameters.get('signal_threshold', 0.0001)

        logger.info(f"SMA Crossover initialized: Fast={self.fast_period}, Slow={self.slow_period}")

    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate SMA indicators.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            dict: Analysis with SMA values
        """
        # Ensure we have enough data
        if len(data) < self.slow_period + 2:
            return {'error': 'Insufficient data'}

        # Calculate SMAs
        data = data.copy()
        data['sma_fast'] = ta.sma(data['close'], length=self.fast_period)
        data['sma_slow'] = ta.sma(data['close'], length=self.slow_period)

        # Calculate ATR for SL/TP
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

        # Get current and previous values
        current = data.iloc[-1]
        previous = data.iloc[-2]

        return {
            'symbol': symbol,
            'current_price': current['close'],
            'sma_fast': current['sma_fast'],
            'sma_slow': current['sma_slow'],
            'prev_sma_fast': previous['sma_fast'],
            'prev_sma_slow': previous['sma_slow'],
            'atr': current['atr'],
            'crossover_up': previous['sma_fast'] <= previous['sma_slow'] and current['sma_fast'] > current['sma_slow'],
            'crossover_down': previous['sma_fast'] >= previous['sma_slow'] and current['sma_fast'] < current['sma_slow'],
            'trend': 'bullish' if current['sma_fast'] > current['sma_slow'] else 'bearish'
        }

    def generate_signals(self, analysis: Dict[str, Any], symbol: str) -> Optional[Signal]:
        """
        Generate trading signals based on SMA crossover.

        Args:
            analysis: Analysis results
            symbol: Trading symbol

        Returns:
            Signal or None
        """
        if 'error' in analysis:
            return None

        current_price = analysis['current_price']
        atr = analysis.get('atr', 0.0050)

        # BUY signal: Fast SMA crosses above Slow SMA
        if analysis['crossover_up']:
            sl, tp = self.calculate_sl_tp(current_price, SignalType.BUY, atr)

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                strength=0.8,
                price=current_price,
                sl=sl,
                tp=tp,
                reason=f"SMA crossover UP (Fast: {analysis['sma_fast']:.5f} > Slow: {analysis['sma_slow']:.5f})"
            )

        # SELL signal: Fast SMA crosses below Slow SMA
        if analysis['crossover_down']:
            sl, tp = self.calculate_sl_tp(current_price, SignalType.SELL, atr)

            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                strength=0.8,
                price=current_price,
                sl=sl,
                tp=tp,
                reason=f"SMA crossover DOWN (Fast: {analysis['sma_fast']:.5f} < Slow: {analysis['sma_slow']:.5f})"
            )

        # No signal
        return Signal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            reason=f"No crossover - Trend: {analysis['trend']}"
        )
