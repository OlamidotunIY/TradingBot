"""
ML Strategy - Trading strategy using ML predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .feature_engineer import FeatureEngineer
from .trainer import MLTrainer
from ..strategies.base_strategy import Signal, SignalType

logger = logging.getLogger('trading_bot')


class MLStrategy:
    """ML-based trading strategy (standalone, not inheriting from BaseStrategy)."""

    def __init__(
        self,
        model_name: str = 'trading_model',
        confidence_threshold: float = 0.6,
        symbols: list = None,
        timeframe: str = 'H1'
    ):
        """
        Initialize ML Strategy.

        Args:
            model_name: Name of saved model to load
            confidence_threshold: Min probability to generate signal
            symbols: List of trading symbols
            timeframe: Timeframe for analysis
        """
        self.name = f"ML_{model_name}"
        self.symbols = symbols or ['GBPUSD']
        self.timeframe = timeframe
        self.confidence_threshold = confidence_threshold
        self.enabled = True

        self.trainer = MLTrainer()
        self.feature_engineer = FeatureEngineer()

        # Try to load model
        try:
            self.trainer.load_model(model_name)
            self.model_loaded = True
            logger.info(f"Loaded ML model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load model {model_name}: {e}")
            self.model_loaded = False

    def get_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Get trading signal from ML model.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            Signal object
        """
        if not self.model_loaded:
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                reason="Model not loaded"
            )

        if len(data) < 200:  # Need enough data for features
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                reason="Insufficient data"
            )

        try:
            # Create features
            features_df = self.feature_engineer.create_features(data.copy())

            if features_df.empty:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    reason="Feature engineering failed"
                )

            # Get the last row for prediction
            last_row = features_df.tail(1)

            # Use ONLY the features that were used during training
            trained_feature_names = self.trainer.feature_names
            available_features = [f for f in trained_feature_names if f in last_row.columns]

            if len(available_features) < len(trained_feature_names):
                missing = set(trained_feature_names) - set(available_features)
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    reason=f"Missing features: {len(missing)}"
                )

            # Select only the trained features in the correct order
            X = last_row[trained_feature_names].values

            # Make prediction (scaler is applied inside predict)
            predictions, probabilities = self.trainer.predict(X)

            prediction = predictions[0]
            proba = probabilities[0]
            max_proba = proba.max()

            current_price = data['close'].iloc[-1]

            # Only generate signal if confidence is high enough
            if max_proba < self.confidence_threshold:
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    reason=f"Low confidence: {max_proba:.2f}"
                )

            # Calculate SL/TP based on ATR
            atr = self._calculate_atr(data)

            if prediction == 1:  # BUY
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    price=current_price,
                    sl=current_price - (atr * 2),
                    tp=current_price + (atr * 4),  # 2:1 RR
                    reason=f"ML BUY (conf: {max_proba:.2f})"
                )

            elif prediction == 0:  # SELL
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    price=current_price,
                    sl=current_price + (atr * 2),
                    tp=current_price - (atr * 4),  # 2:1 RR
                    reason=f"ML SELL (conf: {max_proba:.2f})"
                )

            else:  # NEUTRAL (for 3-class models)
                return Signal(
                    signal_type=SignalType.HOLD,
                    symbol=symbol,
                    reason=f"ML NEUTRAL (conf: {max_proba:.2f})"
                )

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                reason=f"Prediction error: {e}"
            )

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for SL/TP."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0.001

    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'model_loaded': self.model_loaded,
            'confidence_threshold': self.confidence_threshold,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'training_metrics': self.trainer.training_metrics if self.model_loaded else {}
        }
