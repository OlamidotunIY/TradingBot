"""
Feature Engineer - Create ML features from OHLCV data.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger('trading_bot')


class FeatureEngineer:
    """Creates features for ML models from raw OHLCV data."""

    def __init__(self):
        """Initialize Feature Engineer."""
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume columns

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        # 1. Price-based features
        features = self._add_price_features(features)

        # 2. Trend indicators
        features = self._add_trend_features(features)

        # 3. Momentum indicators
        features = self._add_momentum_features(features)

        # 4. Volatility indicators
        features = self._add_volatility_features(features)

        # 5. Volume features
        features = self._add_volume_features(features)

        # 6. Time features
        features = self._add_time_features(features)

        # 7. Pattern features
        features = self._add_pattern_features(features)

        # Drop NaN rows (from indicator calculations)
        features = features.dropna()

        # Store feature names (exclude OHLCV and label)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label',
                        'future_close', 'future_return']
        self.feature_names = [c for c in features.columns if c not in exclude_cols]

        logger.info(f"Created {len(self.feature_names)} features")
        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns over different periods
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)

        # Price ratios
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['close']

        # Distance from highs/lows
        df['dist_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Range position (0-1 scale)
        h20 = df['high'].rolling(20).max()
        l20 = df['low'].rolling(20).min()
        df['range_position'] = (df['close'] - l20) / (h20 - l20 + 1e-10)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            df[f'sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df['close']

        # Exponential Moving Averages
        for period in [9, 21, 50]:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df['close']

        # MA crossover signals
        df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['ema_9_21_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)

        # ADX - Trend strength
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
            df['di_diff'] = df['di_plus'] - df['di_minus']

        # Trend direction
        df['trend_up'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['trend_strong'] = (df['adx'] > 25).astype(int) if 'adx' in df.columns else 0

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)

        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
            df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
            df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)

        # CCI
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)

        # Williams %R
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)

        # ROC (Rate of Change)
        df['roc_10'] = ta.roc(df['close'], length=10)
        df['roc_20'] = ta.roc(df['close'], length=20)

        # Momentum
        df['momentum_10'] = ta.mom(df['close'], length=10)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = ta.atr(df['high'], df['low'], df['close'], length=period)
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            # Find column names dynamically (different pandas-ta versions use different names)
            bb_cols = bb.columns.tolist()
            upper_col = [c for c in bb_cols if 'BBU' in c or 'upper' in c.lower()]
            lower_col = [c for c in bb_cols if 'BBL' in c or 'lower' in c.lower()]
            mid_col = [c for c in bb_cols if 'BBM' in c or 'mid' in c.lower()]

            if upper_col and lower_col and mid_col:
                df['bb_upper'] = bb[upper_col[0]]
                df['bb_lower'] = bb[lower_col[0]]
                df['bb_mid'] = bb[mid_col[0]]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Standard deviation
        for period in [10, 20]:
            df[f'std_{period}'] = df['close'].rolling(period).std() / df['close']

        # True Range
        df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])
        df['true_range_pct'] = df['true_range'] / df['close']

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            df['volume_sma_ratio'] = 0
            df['volume_change'] = 0
            return df

        # Volume SMA ratio
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_sma_ratio'] = df['volume'] / (vol_sma + 1)

        # Volume change
        df['volume_change'] = df['volume'].pct_change()

        # Volume trend
        df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()

        # Price-volume correlation
        df['pv_corr'] = df['close'].pct_change().rolling(20).corr(df['volume'].pct_change())

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Hour of day (cyclical)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (cyclical)
        df['dayofweek'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Trading sessions (binary)
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_newyork'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)  # London & NY overlap

        # Is weekend approaching
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern-based features."""
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)

        # Consecutive moves
        df['consecutive_up'] = df['higher_close'].rolling(5).sum()
        df['consecutive_down'] = (1 - df['higher_close']).rolling(5).sum()

        # Candle body size
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Doji detection
        avg_body = df['body_size'].rolling(20).mean()
        df['is_doji'] = (df['body_size'] < avg_body * 0.3).astype(int)

        # Engulfing pattern
        df['bullish_engulf'] = ((df['close'] > df['open']) &
                                (df['close'].shift(1) < df['open'].shift(1)) &
                                (df['close'] > df['open'].shift(1)) &
                                (df['open'] < df['close'].shift(1))).astype(int)

        df['bearish_engulf'] = ((df['close'] < df['open']) &
                                (df['close'].shift(1) > df['open'].shift(1)) &
                                (df['close'] < df['open'].shift(1)) &
                                (df['open'] > df['close'].shift(1))).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def get_feature_importance_ready_data(
        self,
        df: pd.DataFrame,
        labels: pd.Series
    ) -> tuple:
        """
        Prepare data for model training with only numeric features.

        Returns:
            Tuple of (X features, y labels, feature names)
        """
        # Get only numeric columns that are features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label',
                        'future_close', 'future_return']

        feature_cols = [c for c in df.columns if c not in exclude_cols
                        and df[c].dtype in ['float64', 'int64', 'int32', 'float32']]

        X = df[feature_cols].values
        y = labels.values

        return X, y, feature_cols
