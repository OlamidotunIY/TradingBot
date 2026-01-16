"""
MT5 Trading Bot - Main Entry Point

This is the main entry point for the trading bot.
"""

import time
import signal
import sys
from datetime import datetime
import logging

from src.core import MT5Handler, OrderManager, PositionTracker
from src.strategies import StrategyManager
from src.risk import RiskManager, PositionSizer
from src.data import Database
from src.utils import setup_logging, load_config, is_trading_hours


class TradingBot:
    """Main trading bot class."""

    def __init__(self):
        """Initialize the trading bot."""
        self.running = False
        self.config = None
        self.mt5 = None
        self.order_manager = None
        self.position_tracker = None
        self.strategy_manager = None
        self.risk_manager = None
        self.position_sizer = None
        self.database = None
        self.logger = None

    def setup(self) -> bool:
        """
        Setup all components.

        Returns:
            bool: True if setup successful
        """
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger('trading_bot')
        self.logger.info("Starting MT5 Trading Bot...")

        # Load configuration
        self.config = load_config()
        self.logger.info("Configuration loaded")

        # Initialize database
        self.database = Database(self.config.get('database', {}).get('path', 'data/trading_bot.db'))
        self.database.init_db()
        self.logger.info("Database initialized")

        # Initialize MT5 handler
        mt5_config = self.config.get('mt5', {})
        self.mt5 = MT5Handler(mt5_config)

        if not self.mt5.connect():
            self.logger.error("Failed to connect to MT5")
            return False

        # Initialize trading components
        trading_config = self.config.get('trading', {})
        self.order_manager = OrderManager(self.mt5, trading_config)
        self.position_tracker = PositionTracker(self.mt5)

        # Initialize risk management
        risk_config = self.config.get('risk', {})
        self.risk_manager = RiskManager(risk_config)
        self.position_sizer = PositionSizer(risk_config)

        # Initialize strategies
        strategies_config = self.config.get('strategies_config', {})
        self.strategy_manager = StrategyManager(strategies_config)
        self.strategy_manager.load_strategies()

        self.logger.info("All components initialized successfully")
        return True

    def run(self) -> None:
        """Main trading loop."""
        self.running = True
        check_interval = self.config.get('strategies_config', {}).get('global', {}).get('check_interval', 60)
        trading_config = self.config.get('trading', {})

        self.logger.info("Entering main trading loop...")

        while self.running:
            try:
                # Check if within trading hours
                trading_hours = trading_config.get('trading_hours', {'start': '00:00', 'end': '23:59'})
                trading_days = trading_config.get('trading_days', [0, 1, 2, 3, 4])

                if not is_trading_hours(trading_hours, trading_days):
                    self.logger.debug("Outside trading hours, waiting...")
                    time.sleep(60)
                    continue

                # Update positions
                self.position_tracker.update()

                # Get market data for all strategy symbols
                symbols = self.strategy_manager.get_all_symbols()
                market_data = {}

                for symbol in symbols:
                    data = self.mt5.get_ohlcv(symbol, 'H1', 100)
                    if data is not None:
                        market_data[symbol] = data

                # Get signals from strategies
                signals = self.strategy_manager.get_signals(market_data)

                # Process signals
                for signal in signals:
                    self._process_signal(signal)

                # Wait before next iteration
                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
                self.running = False
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(10)

    def _process_signal(self, signal) -> None:
        """Process a trading signal."""
        from src.strategies.base_strategy import SignalType

        self.logger.info(f"Processing signal: {signal.signal_type.value} {signal.symbol}")

        # Get account info for risk check
        account = self.mt5.get_account_info()
        if account is None:
            return

        positions = self.position_tracker.get_all_positions()

        # Check risk
        risk_check = self.risk_manager.check_trade(
            symbol=signal.symbol,
            volume=signal.volume or 0.1,
            current_positions=[p.__dict__ for p in positions],
            account_balance=account['balance'],
            account_equity=account['equity']
        )

        if not risk_check.approved:
            self.logger.warning(f"Trade rejected by risk manager: {risk_check.reason}")
            return

        # Calculate position size if not specified
        volume = signal.volume
        if volume is None and signal.price and signal.sl:
            volume = self.position_sizer.calculate(
                method=PositionSizer.SizingMethod.FIXED_RISK,
                account_balance=account['balance'],
                entry_price=signal.price,
                sl_price=signal.sl,
                symbol_info=self.mt5.get_symbol_info(signal.symbol) or {}
            )

        volume = volume or 0.1

        # Execute trade
        from src.core.order_manager import OrderRequest, OrderType

        order_type = OrderType.BUY if signal.signal_type == SignalType.BUY else OrderType.SELL

        request = OrderRequest(
            symbol=signal.symbol,
            order_type=order_type,
            volume=volume,
            sl=signal.sl,
            tp=signal.tp,
            comment=signal.reason[:50] if signal.reason else ""
        )

        result = self.order_manager.execute_order(request)

        if result.success:
            self.logger.info(f"Trade executed: Ticket {result.ticket}")
        else:
            self.logger.error(f"Trade failed: {result.error_message}")

    def shutdown(self) -> None:
        """Shutdown the bot."""
        self.logger.info("Shutting down trading bot...")
        self.running = False

        if self.mt5:
            self.mt5.disconnect()

        if self.database:
            self.database.close()

        self.logger.info("Trading bot stopped")


def main():
    """Main function."""
    bot = TradingBot()

    # Setup signal handlers
    def signal_handler(sig, frame):
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup and run
    if bot.setup():
        try:
            bot.run()
        finally:
            bot.shutdown()
    else:
        print("Failed to setup trading bot")
        sys.exit(1)


if __name__ == '__main__':
    main()
