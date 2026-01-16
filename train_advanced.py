#!/usr/bin/env python
"""
Train Advanced ML Models for Trading

This script implements all 4 optimization options:
1. Advanced features (multi-timeframe, market regime)
2. Ensemble of models (XGBoost + RF + GB)
3. LSTM deep learning (optional, requires TensorFlow)
4. Ultra-high confidence filtering

Usage:
    python train_advanced.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src.ml.data_collector import DataCollector
from src.ml.advanced_features import AdvancedFeatureEngineer
from src.ml.ensemble_trainer import EnsembleTrainer


def main():
    print("=" * 80)
    print("ADVANCED ML TRADING MODEL TRAINER")
    print("=" * 80)
    print("\nImplementing ALL 4 optimizations:")
    print("  1. Advanced multi-timeframe features")
    print("  2. Ensemble of 3 models (XGBoost + RF + GB)")
    print("  3. Ultra-high confidence filtering")
    print("  4. LSTM deep learning (if TensorFlow available)")

    # Configuration
    SYMBOL = 'GBPUSD'
    YEARS = 5
    LOOKAHEAD_BARS = 48  # 2 days
    THRESHOLD_PIPS = 40  # Clear directional movement
    BINARY = True  # BUY vs SELL only

    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Years of data: {YEARS}")
    print(f"  Lookahead: {LOOKAHEAD_BARS} bars (48 hours)")
    print(f"  Threshold: {THRESHOLD_PIPS} pips")
    print(f"  Classification: BINARY (BUY/SELL)")

    # Step 1: Collect Data
    print("\n" + "=" * 80)
    print("STEP 1: DATA COLLECTION")
    print("=" * 80)

    collector = DataCollector([SYMBOL])
    if not collector.connect():
        print("✗ Failed to connect to MT5")
        return

    try:
        # Get H1 data (primary)
        print("\nFetching H1 data (5 years)...")
        df_h1, labels = collector.prepare_training_data(
            symbol=SYMBOL,
            timeframe='H1',
            years=YEARS,
            lookahead=LOOKAHEAD_BARS,
            threshold_pips=THRESHOLD_PIPS,
            binary_only=BINARY
        )

        if df_h1.empty:
            print("✗ Failed to get H1 data")
            return

        print(f"✓ H1 data: {len(df_h1)} samples")

        # Get H4 data (for multi-timeframe)
        print("\nFetching H4 data...")
        df_h4 = collector.get_historical_data(SYMBOL, 'H4', bars=10000)
        print(f"✓ H4 data: {len(df_h4)} bars")

        # Get D1 data (for multi-timeframe)
        print("\nFetching D1 data...")
        df_d1 = collector.get_historical_data(SYMBOL, 'D1', bars=2000)
        print(f"✓ D1 data: {len(df_d1)} bars")

    finally:
        collector.disconnect()

    # Step 2: Advanced Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 2: ADVANCED FEATURE ENGINEERING")
    print("=" * 80)

    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)

    # Align labels with features
    labels = labels.loc[features_df.index]

    print(f"\n✓ Created {len(engineer.feature_names)} advanced features")
    print(f"  Final samples: {len(features_df)}")

    # Prepare arrays
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label',
                    'future_close', 'future_return', 'spread', 'real_volume']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols
                    and features_df[c].dtype in ['float64', 'int64', 'int32', 'float32', 'int', 'float']]

    X = features_df[feature_cols].values
    y = labels.values

    print(f"\n  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  BUY labels: {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    print(f"  SELL labels: {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")

    # Step 3: Train Ensemble
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN ENSEMBLE (XGBoost + RF + GB)")
    print("=" * 80)

    ensemble = EnsembleTrainer()
    metrics = ensemble.train_ensemble(
        X, y, feature_cols,
        test_size=0.2,
        use_feature_selection=True,
        top_features=60
    )

    # Save ensemble
    ensemble.save_model('ensemble_model_GBPUSD')

    # Step 4: Try LSTM (if TensorFlow available)
    print("\n" + "=" * 80)
    print("STEP 4: LSTM DEEP LEARNING (Optional)")
    print("=" * 80)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} available")

        from src.ml.lstm_model import LSTMTrader

        lstm = LSTMTrader()
        lstm_metrics = lstm.train(
            X, y, feature_cols,
            sequence_length=50,
            test_size=0.2,
            epochs=30,
            batch_size=64
        )

        if lstm_metrics:
            lstm.save_model('lstm_model_GBPUSD')

    except ImportError:
        print("✗ TensorFlow not installed")
        print("  Install with: pip install tensorflow")
        print("  Skipping LSTM training...")
        lstm_metrics = {}
    except Exception as e:
        print(f"✗ LSTM training failed: {e}")
        lstm_metrics = {}

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)

    print("\n--- ENSEMBLE MODEL ---")
    print(f"  Majority Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  Unanimous Accuracy:    {metrics.get('accuracy_unanimous', 0):.4f}")
    print(f"  High Conf Accuracy:    {metrics.get('accuracy_high_conf', 0):.4f}")
    print(f"  BEST (Unan+HighConf):  {metrics.get('accuracy_best', 0):.4f}")
    print(f"  Best qualifying samples: {metrics.get('best_count', 0)}")

    if lstm_metrics:
        print("\n--- LSTM MODEL ---")
        print(f"  Accuracy:              {lstm_metrics.get('accuracy', 0):.4f}")
        print(f"  High Conf Accuracy:    {lstm_metrics.get('accuracy_high_conf', 0):.4f}")
        print(f"  Very High Conf Acc:    {lstm_metrics.get('accuracy_very_high_conf', 0):.4f}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nRun backtest with: python backtest_advanced.py")
    print("\nModels saved:")
    print("  - models/ensemble_model_GBPUSD.joblib")
    if lstm_metrics:
        print("  - models/lstm_model_GBPUSD.keras")


if __name__ == '__main__':
    main()
