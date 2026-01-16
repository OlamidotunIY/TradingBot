"""
Helpers - Utility Functions

This module contains common helper functions.
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, time
import logging

logger = logging.getLogger('trading_bot')


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        dict: Parsed YAML content
    """
    path = Path(filepath)

    if not path.exists():
        logger.error(f"YAML file not found: {filepath}")
        return {}

    with open(path, 'r') as f:
        content = f.read()

    # Replace environment variables
    content = _replace_env_vars(content)

    return yaml.safe_load(content)


def _replace_env_vars(content: str) -> str:
    """Replace ${VAR} with environment variable values."""
    pattern = r'\$\{(\w+)\}'

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replacer, content)


def load_config(config_dir: str = 'config') -> Dict[str, Any]:
    """
    Load all configuration files.

    Args:
        config_dir: Configuration directory

    Returns:
        dict: Combined configuration
    """
    # Load .env file first
    from dotenv import load_dotenv
    load_dotenv()

    config = {}

    # Load main config
    main_config = load_yaml(f'{config_dir}/config.yaml')
    config.update(main_config)

    # Load strategies config
    strategies = load_yaml(f'{config_dir}/strategies.yaml')
    config['strategies_config'] = strategies

    logger.info("Configuration loaded")
    return config


def is_trading_hours(
    trading_hours: Dict[str, str],
    trading_days: list,
    current_time: Optional[datetime] = None
) -> bool:
    """
    Check if current time is within trading hours.

    Args:
        trading_hours: Dict with 'start' and 'end' times (HH:MM format)
        trading_days: List of trading days (0=Monday, 6=Sunday)
        current_time: Optional datetime, defaults to now

    Returns:
        bool: True if within trading hours
    """
    now = current_time or datetime.utcnow()

    # Check day
    if now.weekday() not in trading_days:
        return False

    # Parse times
    start = datetime.strptime(trading_hours['start'], '%H:%M').time()
    end = datetime.strptime(trading_hours['end'], '%H:%M').time()
    current = now.time()

    # Check time
    if start <= end:
        return start <= current <= end
    else:
        # Overnight session
        return current >= start or current <= end


def format_price(price: float, digits: int = 5) -> str:
    """Format price with specified decimal places."""
    return f"{price:.{digits}f}"


def pips_to_price(pips: float, digits: int = 5) -> float:
    """Convert pips to price difference."""
    if digits == 5 or digits == 3:
        return pips * 0.0001
    elif digits == 2:
        return pips * 0.01
    return pips * 0.0001


def price_to_pips(price_diff: float, digits: int = 5) -> float:
    """Convert price difference to pips."""
    if digits == 5 or digits == 3:
        return price_diff / 0.0001
    elif digits == 2:
        return price_diff / 0.01
    return price_diff / 0.0001


def calculate_lot_value(
    lot_size: float,
    contract_size: int = 100000,
    rate: float = 1.0
) -> float:
    """Calculate the value of a lot size."""
    return lot_size * contract_size * rate


def validate_lot_size(
    lot_size: float,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    lot_step: float = 0.01
) -> float:
    """Validate and round lot size to valid increment."""
    # Clamp to limits
    lot_size = max(min_lot, min(lot_size, max_lot))

    # Round to step
    lot_size = round(lot_size / lot_step) * lot_step

    return round(lot_size, 2)


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    tf_map = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400,
        'W1': 604800,
        'MN1': 2592000
    }
    return tf_map.get(timeframe.upper(), 3600)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.now()
        return self

    def stop(self):
        self.end_time = datetime.now()
        return self

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
