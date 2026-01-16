"""
Logger - Logging Setup and Configuration

This module handles logging configuration.
"""

import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


def setup_logging(
    config_path: str = 'config/logging.yaml',
    default_level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.

    Args:
        config_path: Path to logging config file
        default_level: Default logging level if config not found
    """
    path = Path(config_path)

    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    if path.exists():
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Create log file directories
        for handler in config.get('handlers', {}).values():
            if 'filename' in handler:
                Path(handler['filename']).parent.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        # Basic configuration
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/trading_bot.log')
            ]
        )


def get_logger(name: str = 'trading_bot') -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger: Logger instance
    """
    return logging.getLogger(name)


class TradeLogger:
    """Specialized logger for trades."""

    def __init__(self):
        self.logger = logging.getLogger('trading_bot.trades')

    def log_trade_opened(
        self,
        ticket: int,
        symbol: str,
        trade_type: str,
        volume: float,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> None:
        """Log trade opened."""
        self.logger.info(
            f"OPENED | Ticket: {ticket} | {trade_type} {volume} {symbol} @ {price} | "
            f"SL: {sl or 'None'} | TP: {tp or 'None'}"
        )

    def log_trade_closed(
        self,
        ticket: int,
        symbol: str,
        trade_type: str,
        volume: float,
        close_price: float,
        profit: float,
        reason: str = 'Manual'
    ) -> None:
        """Log trade closed."""
        profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
        self.logger.info(
            f"CLOSED | Ticket: {ticket} | {trade_type} {volume} {symbol} @ {close_price} | "
            f"Profit: {profit_str} | Reason: {reason}"
        )

    def log_trade_modified(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> None:
        """Log trade modified."""
        self.logger.info(
            f"MODIFIED | Ticket: {ticket} | New SL: {new_sl or 'None'} | New TP: {new_tp or 'None'}"
        )

    def log_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        reason: str
    ) -> None:
        """Log trading signal."""
        self.logger.info(
            f"SIGNAL | Strategy: {strategy} | {signal_type} {symbol} | {reason}"
        )
