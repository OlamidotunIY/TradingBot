"""
Advanced Features - Multi-timeframe and Market Regime Features
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger('trading_bot')


class AdvancedFeatureEngineer:
    """Creates advanced features including multi-timeframe and market regime."""

    def __init__(self):
        self.feature_names: List[str] = []

    def create_advanced_features(
        self,
        df_h1: pd.DataFrame,
        df_h4: pd.DataFrame = None,
        df_d1: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create advanced features with multi-timeframe analysis.

        Args:
            df_h1: H1 OHLCV data (primary)
            df_h4: H4 OHLCV data (optional, for MTF)
            df_d1: D1 OHLCV data (optional, for MTF)
        """
        features = df_h1.copy()

        # 1. Basic features
        features = self._add_basic_features(features)

        # 2. Market regime features
        features = self._add_regime_features(features)

        # 3. Session features
        features = self._add_session_features(features)

        # 4. Multi-timeframe features (if available)
        if df_h4 is not None:
            features = self._add_mtf_features(features, df_h4, 'h4')
        if df_d1 is not None:
            features = self._add_mtf_features(features, df_d1, 'd1')

        # 5. Pattern quality features
        features = self._add_pattern_quality_features(features)

        # Drop NaN
        features = features.dropna()

        # Store feature names
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label',
                        'future_close', 'future_return', 'spread', 'real_volume']
        self.feature_names = [c for c in features.columns if c not in exclude_cols]

        logger.info(f"Created {len(self.feature_names)} advanced features")
        return features

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add core technical indicators."""
        # Trend indicators
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}']

        for period in [9, 21, 50]:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)

        # EMA crossovers (key signal!)
        df['ema_9_21_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        df['ema_21_50_cross'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)

        # Momentum
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(14)

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

        # Volatility
        for period in [14, 21]:
            df[f'atr_{period}'] = ta.atr(df['high'], df['low'], df['close'], length=period)
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
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

        # ADX - Trend strength
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
            df['di_diff'] = df['di_plus'] - df['di_minus']

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        # Trend regime based on ADX
        if 'adx' in df.columns:
            df['regime_trending'] = (df['adx'] > 25).astype(int)
            df['regime_strong_trend'] = (df['adx'] > 40).astype(int)

        # Volatility regime
        if 'atr_14' in df.columns:
            atr_mean = df['atr_14'].rolling(100).mean()
            atr_std = df['atr_14'].rolling(100).std()
            df['volatility_zscore'] = (df['atr_14'] - atr_mean) / (atr_std + 1e-10)
            df['regime_high_vol'] = (df['volatility_zscore'] > 1).astype(int)
            df['regime_low_vol'] = (df['volatility_zscore'] < -1).astype(int)

        # Trend direction
        df['trend_direction'] = np.where(
            (df['sma_20'] > df['sma_50']) & (df['sma_50'] > df['sma_100']), 1,
            np.where(
                (df['sma_20'] < df['sma_50']) & (df['sma_50'] < df['sma_100']), -1, 0
            )
        )

        # Range-bound detection
        price_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        avg_range = price_range.rolling(50).mean()
        df['regime_ranging'] = (price_range < avg_range * 0.7).astype(int)

        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        hour = df.index.hour

        # Trading sessions (UTC)
        df['session_asian'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_london'] = ((hour >= 8) & (hour < 16)).astype(int)
        df['session_newyork'] = ((hour >= 13) & (hour < 21)).astype(int)
        df['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)  # Best time!

        # Day of week
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        return df

    def _add_mtf_features(
        self,
        df: pd.DataFrame,
        df_higher: pd.DataFrame,
        suffix: str
    ) -> pd.DataFrame:
        """Add multi-timeframe features from higher timeframe."""
        # Calculate higher TF indicators
        htf = df_higher.copy()
        htf[f'sma_20_{suffix}'] = ta.sma(htf['close'], length=20)
        htf[f'sma_50_{suffix}'] = ta.sma(htf['close'], length=50)
        htf[f'rsi_{suffix}'] = ta.rsi(htf['close'], length=14)
        htf[f'trend_{suffix}'] = np.where(htf[f'sma_20_{suffix}'] > htf[f'sma_50_{suffix}'], 1, -1)

        # ADX for higher TF
        adx = ta.adx(htf['high'], htf['low'], htf['close'], length=14)
        if adx is not None:
            htf[f'adx_{suffix}'] = adx['ADX_14']

        # Merge with lower timeframe using forward fill
        merge_cols = [f'sma_20_{suffix}', f'sma_50_{suffix}', f'rsi_{suffix}',
                      f'trend_{suffix}', f'adx_{suffix}']
        existing_cols = [c for c in merge_cols if c in htf.columns]

        htf_subset = htf[existing_cols]
        df = df.join(htf_subset, how='left')
        df[existing_cols] = df[existing_cols].ffill()

        # Alignment feature - does H1 trend match H4/D1 trend?
        if f'trend_{suffix}' in df.columns and 'ema_9_21_cross' in df.columns:
            df[f'alignment_{suffix}'] = (df['ema_9_21_cross'] == df[f'trend_{suffix}']).astype(int)

        return df

    def _add_pattern_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features measuring pattern/signal quality."""
        # Confirmation count - how many indicators agree
        confirmations = pd.Series(0, index=df.index)

        # Count bullish confirmations
        if 'ema_9_21_cross' in df.columns:
            confirmations += (df['ema_9_21_cross'] == 1).astype(int)
        if 'macd_cross' in df.columns:
            confirmations += (df['macd_cross'] == 1).astype(int)
        if 'rsi_14' in df.columns:
            confirmations += ((df['rsi_14'] > 50) & (df['rsi_14'] < 70)).astype(int)
        if 'stoch_cross' in df.columns:
            confirmations += (df['stoch_cross'] == 1).astype(int)
        if 'di_diff' in df.columns:
            confirmations += (df['di_diff'] > 0).astype(int)

        df['bullish_confirmations'] = confirmations

        # Count bearish confirmations
        confirmations = pd.Series(0, index=df.index)
        if 'ema_9_21_cross' in df.columns:
            confirmations += (df['ema_9_21_cross'] == -1).astype(int)
        if 'macd_cross' in df.columns:
            confirmations += (df['macd_cross'] == -1).astype(int)
        if 'rsi_14' in df.columns:
            confirmations += ((df['rsi_14'] < 50) & (df['rsi_14'] > 30)).astype(int)
        if 'stoch_cross' in df.columns:
            confirmations += (df['stoch_cross'] == -1).astype(int)
        if 'di_diff' in df.columns:
            confirmations += (df['di_diff'] < 0).astype(int)

        df['bearish_confirmations'] = confirmations

        # Signal strength (max of bullish/bearish)
        df['signal_strength'] = df[['bullish_confirmations', 'bearish_confirmations']].max(axis=1)

        # Trend alignment score
        alignment = 0
        if 'ema_9_21_cross' in df.columns and 'ema_21_50_cross' in df.columns:
            alignment = (df['ema_9_21_cross'] == df['ema_21_50_cross']).astype(int)
        df['trend_alignment'] = alignment

        return df

    def get_feature_names(self) -> List[str]:
        return self.feature_names
