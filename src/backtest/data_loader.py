"""
Data Loader - Historical Data Loading

This module handles loading historical data for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('trading_bot')


class DataLoader:
    """Loads historical data for backtesting."""

    def __init__(self, mt5_handler=None, data_dir: str = 'data'):
        """
        Initialize Data Loader.

        Args:
            mt5_handler: Optional MT5Handler for live data
            data_dir: Directory for cached data
        """
        self.mt5 = mt5_handler
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, pd.DataFrame] = {}

    def load_from_mt5(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        from_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data from MT5.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            count: Number of bars
            from_date: Start date

        Returns:
            DataFrame: OHLCV data
        """
        if self.mt5 is None:
            logger.error("MT5 handler not available")
            return None

        data = self.mt5.get_ohlcv(symbol, timeframe, count, from_date)

        if data is not None:
            self._cache[f"{symbol}_{timeframe}"] = data
            logger.info(f"Loaded {len(data)} bars for {symbol} {timeframe}")

        return data

    def load_from_csv(
        self,
        filepath: str,
        date_column: str = 'time',
        parse_dates: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            parse_dates: Whether to parse dates

        Returns:
            DataFrame: OHLCV data
        """
        try:
            path = Path(filepath)

            if not path.exists():
                logger.error(f"File not found: {filepath}")
                return None

            data = pd.read_csv(
                filepath,
                parse_dates=[date_column] if parse_dates else False
            )

            # Standardize column names
            data.columns = data.columns.str.lower()

            # Set index
            if date_column.lower() in data.columns:
                data.set_index(date_column.lower(), inplace=True)

            logger.info(f"Loaded {len(data)} bars from {filepath}")
            return data

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None

    def save_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        include_index: bool = True
    ) -> bool:
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            filename: Output filename
            include_index: Whether to include index

        Returns:
            bool: Success status
        """
        try:
            filepath = self.data_dir / filename
            data.to_csv(filepath, index=include_index)
            logger.info(f"Saved data to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            return False

    def generate_sample_data(
        self,
        symbol: str = 'EURUSD',
        timeframe: str = 'H1',
        bars: int = 1000,
        start_price: float = 1.1000,
        volatility: float = 0.0002
    ) -> pd.DataFrame:
        """
        Generate sample OHLCV data for testing.

        Args:
            symbol: Symbol name
            timeframe: Timeframe
            bars: Number of bars
            start_price: Starting price
            volatility: Price volatility

        Returns:
            DataFrame: Sample OHLCV data
        """
        np.random.seed(42)

        # Generate datetime index
        tf_minutes = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440
        }

        minutes = tf_minutes.get(timeframe, 60)
        dates = pd.date_range(
            end=datetime.now(),
            periods=bars,
            freq=f'{minutes}min'
        )

        # Generate prices with random walk
        returns = np.random.normal(0, volatility, bars)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLC
        data = pd.DataFrame(index=dates)
        data['open'] = prices
        data['high'] = prices * (1 + np.random.uniform(0, volatility * 2, bars))
        data['low'] = prices * (1 - np.random.uniform(0, volatility * 2, bars))
        data['close'] = prices * (1 + np.random.normal(0, volatility, bars))
        data['tick_volume'] = np.random.randint(100, 10000, bars)
        data['spread'] = np.random.randint(1, 5, bars)
        data['real_volume'] = data['tick_volume'] * np.random.randint(10, 100, bars)

        # Ensure high >= close, open, low and low <= close, open
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)

        data.index.name = 'time'

        logger.info(f"Generated {bars} sample bars for {symbol}")
        return data

    def resample(
        self,
        data: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            data: Source data
            target_timeframe: Target timeframe

        Returns:
            DataFrame: Resampled data
        """
        tf_map = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1h', 'H4': '4h', 'D1': '1D', 'W1': '1W', 'MN1': '1M'
        }

        freq = tf_map.get(target_timeframe, '1h')

        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }).dropna()

        logger.info(f"Resampled to {target_timeframe}: {len(resampled)} bars")
        return resampled

    def get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data."""
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """Clear data cache."""
        self._cache.clear()
        logger.info("Data cache cleared")
