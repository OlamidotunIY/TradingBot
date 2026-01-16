"""
Strategy Manager - Load, Manage and Coordinate Strategies

This module manages multiple trading strategies.
"""

from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import logging

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger('trading_bot')


class StrategyManager:
    """Manages multiple trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Strategy Manager.

        Args:
            config: Strategies configuration from strategies.yaml
        """
        self.config = config
        self.global_config = config.get('global', {})
        self.strategies_config = config.get('strategies', {})

        self._strategies: Dict[str, BaseStrategy] = {}
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}

        # Register built-in strategies
        self._register_builtin_strategies()

    def _register_builtin_strategies(self) -> None:
        """Register built-in strategy classes."""
        from .sma_crossover import SMACrossoverStrategy
        from .rsi_strategy import RSIStrategy

        self._strategy_classes['sma_crossover'] = SMACrossoverStrategy
        self._strategy_classes['rsi_strategy'] = RSIStrategy

    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a custom strategy class.

        Args:
            name: Strategy name
            strategy_class: Strategy class (must inherit from BaseStrategy)
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")

        self._strategy_classes[name] = strategy_class
        logger.info(f"Registered strategy class: {name}")

    def load_strategies(self) -> None:
        """Load and initialize all configured strategies."""
        for name, strategy_config in self.strategies_config.items():
            if name not in self._strategy_classes:
                logger.warning(f"Strategy class not found: {name}")
                continue

            try:
                strategy_class = self._strategy_classes[name]
                strategy = strategy_class(name=name, config=strategy_config)
                self._strategies[name] = strategy

                status = "enabled" if strategy.enabled else "disabled"
                logger.info(f"Loaded strategy: {name} ({status})")

            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            BaseStrategy or None
        """
        return self._strategies.get(name)

    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all loaded strategies."""
        return list(self._strategies.values())

    def get_enabled_strategies(self) -> List[BaseStrategy]:
        """Get all enabled strategies."""
        return [s for s in self._strategies.values() if s.enabled]

    def get_strategies_for_symbol(self, symbol: str) -> List[BaseStrategy]:
        """
        Get strategies that trade a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            list: Strategies for the symbol
        """
        return [s for s in self.get_enabled_strategies() if symbol in s.symbols]

    def enable_strategy(self, name: str) -> bool:
        """
        Enable a strategy.

        Args:
            name: Strategy name

        Returns:
            bool: Success
        """
        strategy = self._strategies.get(name)
        if strategy:
            strategy.enable()
            return True
        return False

    def disable_strategy(self, name: str) -> bool:
        """
        Disable a strategy.

        Args:
            name: Strategy name

        Returns:
            bool: Success
        """
        strategy = self._strategies.get(name)
        if strategy:
            strategy.disable()
            return True
        return False

    def toggle_strategy(self, name: str) -> Optional[bool]:
        """
        Toggle a strategy's enabled state.

        Args:
            name: Strategy name

        Returns:
            bool: New enabled state, or None if not found
        """
        strategy = self._strategies.get(name)
        if strategy:
            if strategy.enabled:
                strategy.disable()
            else:
                strategy.enable()
            return strategy.enabled
        return None

    def get_all_symbols(self) -> List[str]:
        """Get all unique symbols across enabled strategies."""
        symbols = set()
        for strategy in self.get_enabled_strategies():
            symbols.update(strategy.symbols)
        return list(symbols)

    def get_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Get signals from all enabled strategies.

        Args:
            market_data: Dict of symbol -> DataFrame

        Returns:
            list: All generated signals
        """
        signals = []

        for strategy in self.get_enabled_strategies():
            for symbol in strategy.symbols:
                if symbol not in market_data:
                    continue

                data = market_data[symbol]
                signal = strategy.get_signal(data, symbol)

                if signal and signal.signal_type != SignalType.HOLD:
                    signals.append(signal)

        return signals

    def get_status(self) -> Dict[str, Any]:
        """Get status of all strategies."""
        return {
            'total_strategies': len(self._strategies),
            'enabled_strategies': len(self.get_enabled_strategies()),
            'strategies': {
                name: strategy.get_status()
                for name, strategy in self._strategies.items()
            }
        }

    def reload_config(self, config: Dict[str, Any]) -> None:
        """
        Reload configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self.strategies_config = config.get('strategies', {})

        # Update existing strategies
        for name, strategy in self._strategies.items():
            if name in self.strategies_config:
                strategy.config = self.strategies_config[name]
                strategy.enabled = self.strategies_config[name].get('enabled', True)

        logger.info("Strategy configuration reloaded")
