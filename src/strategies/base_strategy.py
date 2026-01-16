"""
Base Strategy - Abstract Base Class for Trading Strategies

All trading strategies should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger('trading_bot')


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Trading signal data class."""
    signal_type: SignalType
    symbol: str
    strength: float = 1.0  # 0.0 to 1.0
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    volume: Optional[float] = None
    reason: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry signal."""
        return self.signal_type in [SignalType.BUY, SignalType.SELL]

    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit signal."""
        return self.signal_type in [SignalType.CLOSE_BUY, SignalType.CLOSE_SELL]


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement:
    - analyze(): Analyze market data
    - generate_signals(): Generate trading signals
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.symbols = config.get('symbols', [])
        self.timeframe = config.get('timeframe', 'H1')
        self.parameters = config.get('parameters', {})
        self.risk_config = config.get('risk', {})

        self._last_signal: Dict[str, Signal] = {}
        self._last_analysis: Optional[datetime] = None

        logger.info(f"Strategy '{name}' initialized")

    @abstractmethod
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Analyze market data.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            dict: Analysis results (indicators, patterns, etc.)
        """
        pass

    @abstractmethod
    def generate_signals(self, analysis: Dict[str, Any], symbol: str) -> Optional[Signal]:
        """
        Generate trading signals based on analysis.

        Args:
            analysis: Analysis results from analyze()
            symbol: Trading symbol

        Returns:
            Signal or None
        """
        pass

    def get_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Get trading signal for a symbol.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            Signal or None
        """
        if not self.enabled:
            return None

        if symbol not in self.symbols:
            return None

        try:
            # Run analysis
            analysis = self.analyze(data, symbol)

            # Generate signal
            signal = self.generate_signals(analysis, symbol)

            if signal and signal.signal_type != SignalType.HOLD:
                self._last_signal[symbol] = signal
                logger.info(f"[{self.name}] Signal: {signal.signal_type.value} {symbol} - {signal.reason}")

            self._last_analysis = datetime.now()
            return signal

        except Exception as e:
            logger.error(f"[{self.name}] Error analyzing {symbol}: {e}")
            return None

    def calculate_position_size(self, account_balance: float, entry_price: float, sl_price: float) -> float:
        """
        Calculate position size based on risk.

        Args:
            account_balance: Account balance
            entry_price: Entry price
            sl_price: Stop loss price

        Returns:
            float: Position size (lots)
        """
        risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)
        risk_amount = account_balance * risk_per_trade

        # Calculate pips risk
        pips_risk = abs(entry_price - sl_price) * 10000

        if pips_risk == 0:
            return 0.01  # Minimum lot size

        # Approximate lot size (simplified)
        lot_size = risk_amount / (pips_risk * 10)

        # Round to 2 decimal places and ensure minimum
        lot_size = max(0.01, round(lot_size, 2))

        return lot_size

    def calculate_sl_tp(
        self,
        entry_price: float,
        signal_type: SignalType,
        atr: Optional[float] = None
    ) -> tuple:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            signal_type: BUY or SELL
            atr: Average True Range (optional)

        Returns:
            tuple: (stop_loss, take_profit)
        """
        sl_multiplier = self.risk_config.get('sl_multiplier', 1.5)
        tp_multiplier = self.risk_config.get('tp_multiplier', 2.0)

        # Default to 50 pips if no ATR provided
        if atr is None:
            atr = 0.0050  # 50 pips default

        if signal_type == SignalType.BUY:
            sl = entry_price - (atr * sl_multiplier)
            tp = entry_price + (atr * tp_multiplier)
        else:
            sl = entry_price + (atr * sl_multiplier)
            tp = entry_price - (atr * tp_multiplier)

        return round(sl, 5), round(tp, 5)

    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        """Get last signal for a symbol."""
        return self._last_signal.get(symbol)

    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        logger.info(f"Strategy '{self.name}' enabled")

    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        logger.info(f"Strategy '{self.name}' disabled")

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'last_analysis': self._last_analysis.isoformat() if self._last_analysis else None,
            'parameters': self.parameters
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})>"
