"""
Data Collector - Fetch historical data from MT5 for ML training.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger('trading_bot')


class DataCollector:
    """Collects and prepares data from MT5 for ML training."""

    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
    }

    def __init__(self, symbols: List[str] = None):
        """
        Initialize Data Collector.

        Args:
            symbols: List of symbols to collect data for
        """
        self.symbols = symbols or ['GBPUSD']
        self.connected = False

    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        if account:
            logger.info(f"Connected to MT5 - Balance: ${account.balance:,.2f}")
            self.connected = True
            return True
        return False

    def disconnect(self):
        """Disconnect from MT5."""
        mt5.shutdown()
        self.connected = False

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = 'H1',
        bars: int = 50000,
        from_date: datetime = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1)
            bars: Number of bars to fetch
            from_date: Start date (if None, fetches most recent)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            self.connect()

        tf = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_H1)

        if from_date:
            rates = mt5.copy_rates_from(symbol, tf, from_date, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get data for {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Rename columns for consistency
        df.rename(columns={
            'tick_volume': 'volume'
        }, inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str] = ['M15', 'H1', 'H4'],
        bars: int = 50000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Returns:
            Dict mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            df = self.get_historical_data(symbol, tf, bars)
            if not df.empty:
                data[tf] = df
        return data

    def prepare_training_data(
        self,
        symbol: str = 'GBPUSD',
        timeframe: str = 'H1',
        years: int = 5,
        lookahead: int = 24,  # Hours to look ahead for label
        threshold_pips: float = 10,  # Min movement for label
        binary_only: bool = False  # If True, exclude NEUTRAL samples
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training with labels.

        Labels:
            1 = Price went UP by threshold_pips
            0 = Price went DOWN by threshold_pips
            2 = Price stayed within threshold (NEUTRAL) - excluded if binary_only

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            years: Years of historical data
            lookahead: Bars to look ahead for labeling
            threshold_pips: Minimum pip movement for directional label
            binary_only: If True, only return BUY/SELL samples (no NEUTRAL)

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Calculate bars needed (approximate)
        tf_hours = {'M15': 0.25, 'H1': 1, 'H4': 4, 'D1': 24}
        hours_per_year = 365 * 24
        bars_needed = int(years * hours_per_year / tf_hours.get(timeframe, 1))

        print(f"Fetching {bars_needed} bars ({years} years of {timeframe} data)...")
        df = self.get_historical_data(symbol, timeframe, bars_needed)

        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Create labels based on future price movement
        pip_multiplier = 10000 if 'JPY' not in symbol else 100
        threshold_price = threshold_pips / pip_multiplier

        # Calculate future returns
        df['future_close'] = df['close'].shift(-lookahead)
        df['future_return'] = df['future_close'] - df['close']

        # Create labels
        def get_label(row):
            if pd.isna(row['future_return']):
                return np.nan
            if row['future_return'] > threshold_price:
                return 1  # BUY
            elif row['future_return'] < -threshold_price:
                return 0  # SELL
            else:
                return 2  # NEUTRAL

        df['label'] = df.apply(get_label, axis=1)

        # Remove rows without labels (last lookahead rows)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # For binary classification, remove NEUTRAL samples
        if binary_only:
            before_count = len(df)
            df = df[df['label'] != 2]
            print(f"Binary mode: Removed {before_count - len(df)} NEUTRAL samples")

        # Keep only OHLCV for feature engineering
        features = df[['open', 'high', 'low', 'close', 'volume']].copy()
        labels = df['label']

        print(f"Prepared {len(features)} samples")
        print(f"Label distribution:")
        print(f"  BUY (1):     {(labels == 1).sum()} ({(labels == 1).sum()/len(labels)*100:.1f}%)")
        print(f"  SELL (0):    {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
        if not binary_only:
            print(f"  NEUTRAL (2): {(labels == 2).sum()} ({(labels == 2).sum()/len(labels)*100:.1f}%)")

        return features, labels

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV."""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        df.to_csv(filepath)
        print(f"✓ Saved data to: {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV."""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        return pd.DataFrame()
