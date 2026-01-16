"""
LSTM Model - Deep Learning for Sequence-Based Trading Prediction

Uses TensorFlow/Keras for LSTM implementation.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger('trading_bot')


class LSTMTrader:
    """LSTM-based trading model for sequence prediction."""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.model = None
        self.scaler = None
        self.sequence_length = 50
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, Any] = {}

    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert feature matrix to sequences for LSTM.

        Args:
            X: Feature matrix (samples, features)
            y: Labels
            sequence_length: Number of timesteps to look back

        Returns:
            X_seq: (samples, sequence_length, features)
            y_seq: (samples,)
        """
        self.sequence_length = sequence_length

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.

        Args:
            input_shape: (sequence_length, n_features)
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return

        self.model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),

            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),

            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),

            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),

            # Output layer (binary classification)
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("\nLSTM Model Architecture:")
        self.model.summary()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sequence_length: int = 50,
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Train LSTM model.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            sequence_length: Timesteps to look back
            test_size: Fraction for test set
            epochs: Training epochs
            batch_size: Batch size
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return {}

        self.feature_names = feature_names

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        print(f"\nCreating sequences (length={sequence_length})...")
        X_seq, y_seq = self.prepare_sequences(X_scaled, y, sequence_length)

        print(f"Sequence shape: {X_seq.shape}")
        print(f"Labels shape: {y_seq.shape}")

        # Time-based split
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        print(f"Training set: {len(X_train)} sequences")
        print(f"Test set: {len(X_test)} sequences")

        # Build model
        self.build_model((sequence_length, X.shape[1]))

        if self.model is None:
            return {}

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Train
        print(f"\nTraining LSTM for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        print("\n" + "=" * 60)
        print("LSTM MODEL EVALUATION")
        print("=" * 60)

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        # High confidence predictions
        high_conf_mask = (y_pred_proba > 0.70).flatten() | (y_pred_proba < 0.30).flatten()
        if high_conf_mask.sum() > 0:
            y_test_hc = y_test[high_conf_mask]
            y_pred_hc = y_pred[high_conf_mask]
            acc_hc = accuracy_score(y_test_hc, y_pred_hc)
            print(f"\nHigh Confidence (>70% or <30%):")
            print(f"  Samples: {high_conf_mask.sum()} ({high_conf_mask.sum()/len(y_test)*100:.1f}%)")
            print(f"  Accuracy: {acc_hc:.4f}")
            metrics['accuracy_high_conf'] = acc_hc

        # Very high confidence
        very_high_conf_mask = (y_pred_proba > 0.80).flatten() | (y_pred_proba < 0.20).flatten()
        if very_high_conf_mask.sum() > 0:
            y_test_vhc = y_test[very_high_conf_mask]
            y_pred_vhc = y_pred[very_high_conf_mask]
            acc_vhc = accuracy_score(y_test_vhc, y_pred_vhc)
            print(f"\nVery High Confidence (>80% or <20%):")
            print(f"  Samples: {very_high_conf_mask.sum()} ({very_high_conf_mask.sum()/len(y_test)*100:.1f}%)")
            print(f"  Accuracy: {acc_vhc:.4f}")
            metrics['accuracy_very_high_conf'] = acc_vhc

        self.training_metrics = metrics
        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make LSTM predictions.

        Args:
            X: Feature matrix (must be at least sequence_length rows)

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Scale
        X_scaled = self.scaler.transform(X)

        # Take last sequence_length rows as input
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows")

        X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)

        # Predict
        proba = self.model.predict(X_seq, verbose=0)[0, 0]
        pred = 1 if proba > 0.5 else 0

        return np.array([pred]), np.array([[1-proba, proba]])

    def save_model(self, name: str = 'lstm_model'):
        """Save LSTM model."""
        import joblib

        if self.model is None:
            print("No model to save!")
            return

        # Save Keras model
        model_path = os.path.join(self.model_dir, f"{name}.keras")
        self.model.save(model_path)

        # Save scaler and metadata
        meta_path = os.path.join(self.model_dir, f"{name}_meta.joblib")
        joblib.dump({
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'metrics': self.training_metrics
        }, meta_path)

        print(f"\n✓ LSTM model saved to: {model_path}")

    def load_model(self, name: str = 'lstm_model'):
        """Load saved LSTM model."""
        try:
            import tensorflow as tf
            import joblib
        except ImportError:
            print("TensorFlow not installed")
            return

        model_path = os.path.join(self.model_dir, f"{name}.keras")
        meta_path = os.path.join(self.model_dir, f"{name}_meta.joblib")

        self.model = tf.keras.models.load_model(model_path)

        meta = joblib.load(meta_path)
        self.scaler = meta['scaler']
        self.sequence_length = meta['sequence_length']
        self.feature_names = meta['feature_names']
        self.training_metrics = meta['metrics']

        print(f"✓ LSTM model loaded: {name}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {len(self.feature_names)}")
