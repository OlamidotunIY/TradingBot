"""
Market Regime Detector

Classifies market into regimes: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY
Used to filter trades - only trade when regime matches strategy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger('trading_bot')


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """Detects current market regime using ML and technical indicators."""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.scaler = None

    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']

        # Trend indicators
        features['sma_20'] = close.rolling(20).mean()
        features['sma_50'] = close.rolling(50).mean()
        features['sma_200'] = close.rolling(200).mean()

        features['price_vs_sma20'] = (close - features['sma_20']) / features['sma_20']
        features['price_vs_sma50'] = (close - features['sma_50']) / features['sma_50']
        features['price_vs_sma200'] = (close - features['sma_200']) / features['sma_200']

        features['sma_20_50_cross'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
        features['sma_50_200_cross'] = (features['sma_50'] - features['sma_200']) / features['sma_200']

        # ADX for trend strength (simplified)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        features['atr'] = atr
        features['atr_ratio'] = atr / close

        # Volatility using ATR
        features['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        features['volatility_50'] = close.pct_change().rolling(50).std() * np.sqrt(252)
        features['vol_ratio'] = features['volatility_20'] / features['volatility_50']

        # Bollinger Band width
        bb_std = close.rolling(20).std()
        features['bb_width'] = (4 * bb_std) / features['sma_20']

        # Momentum
        features['roc_10'] = close.pct_change(10)
        features['roc_20'] = close.pct_change(20)

        # Price range ratio
        features['range_ratio'] = (high.rolling(20).max() - low.rolling(20).min()) / close

        # Trend consistency (how often price is above/below SMA)
        above_sma = (close > features['sma_20']).rolling(20).mean()
        features['trend_consistency'] = abs(above_sma - 0.5) * 2

        # Higher highs / lower lows
        features['higher_highs'] = (high > high.shift(1)).rolling(10).sum() / 10
        features['lower_lows'] = (low < low.shift(1)).rolling(10).sum() / 10

        return features.dropna()

    def generate_regime_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Generate regime labels based on price action."""
        labels = pd.Series(index=features.index, dtype=str)

        close = df.loc[features.index, 'close']

        # Calculate 20-bar forward return
        forward_return = close.shift(-20) / close - 1
        forward_volatility = close.pct_change().rolling(20).std().shift(-10)

        # Classify regimes
        for i in range(len(features) - 20):
            idx = features.index[i]
            ret = forward_return.iloc[i]
            vol = forward_volatility.iloc[i]
            atr_ratio = features.loc[idx, 'atr_ratio']

            # High volatility regime (ATR ratio > 1.5x normal)
            if vol > 0.02 or atr_ratio > features['atr_ratio'].quantile(0.85):
                labels.iloc[i] = MarketRegime.HIGH_VOLATILITY.value
            # Strong uptrend
            elif ret > 0.01:
                labels.iloc[i] = MarketRegime.TRENDING_UP.value
            # Strong downtrend
            elif ret < -0.01:
                labels.iloc[i] = MarketRegime.TRENDING_DOWN.value
            # Ranging
            else:
                labels.iloc[i] = MarketRegime.RANGING.value

        return labels.dropna()

    def train(self, df: pd.DataFrame) -> dict:
        """Train regime detection model."""
        print("Training regime detector...")

        features = self.calculate_regime_features(df)
        labels = self.generate_regime_labels(df, features)

        # Align data
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx].values
        y = labels.loc[common_idx].values

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Train
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # Accuracy
        accuracy = self.model.score(X, y)

        # Regime distribution
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts / len(y)))

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Regime distribution: {dist}")

        return {'accuracy': accuracy, 'distribution': dist}

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predict current market regime."""
        if self.model is None:
            return MarketRegime.RANGING.value, 0.5

        features = self.calculate_regime_features(df)
        if features.empty:
            return MarketRegime.RANGING.value, 0.5

        X = features.iloc[-1:].values
        X = np.nan_to_num(X, nan=0.0)
        X = self.scaler.transform(X)

        regime = self.model.predict(X)[0]
        proba = self.model.predict_proba(X).max()

        return regime, proba

    def should_trade(self, regime: str, signal_type: str) -> bool:
        """Check if we should trade given current regime and signal."""
        # Don't trade in high volatility
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            return False

        # Buy signals work best in uptrends or ranging
        if signal_type == 'BUY':
            return regime in [MarketRegime.TRENDING_UP.value, MarketRegime.RANGING.value]

        # Sell signals work best in downtrends or ranging
        if signal_type == 'SELL':
            return regime in [MarketRegime.TRENDING_DOWN.value, MarketRegime.RANGING.value]

        return True

    def save(self, name: str) -> None:
        """Save regime detector."""
        path = self.model_dir / f'regime_detector_{name}.joblib'
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
        print(f"✓ Regime detector saved: {path}")

    def load(self, name: str) -> None:
        """Load regime detector."""
        path = self.model_dir / f'regime_detector_{name}.joblib'
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        print(f"✓ Regime detector loaded: {name}")
