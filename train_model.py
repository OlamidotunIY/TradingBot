#!/usr/bin/env python
"""
Train ML Model for Trading - ADVANCED VERSION

Implements all 4 optimizations:
1. Advanced multi-timeframe features
2. Ensemble of 3 models (XGBoost + RF + GB)
3. Ultra-high confidence filtering
4. LSTM deep learning (if TensorFlow available)
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
    LOOKAHEAD_BARS = 48
    THRESHOLD_PIPS = 40
    BINARY = True

    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Years of data: {YEARS}")
    print(f"  Lookahead: {LOOKAHEAD_BARS} bars (48 hours)")
    print(f"  Threshold: {THRESHOLD_PIPS} pips")

    # Step 1: Collect Data
    print("\n" + "=" * 80)
    print("STEP 1: DATA COLLECTION")
    print("=" * 80)

    collector = DataCollector([SYMBOL])
    if not collector.connect():
        print("✗ Failed to connect to MT5")
        return

    try:
        print("\nFetching H1 data (5 years)...")
        df_h1, labels = collector.prepare_training_data(
            symbol=SYMBOL, timeframe='H1', years=YEARS,
            lookahead=LOOKAHEAD_BARS, threshold_pips=THRESHOLD_PIPS,
            binary_only=BINARY
        )

        if df_h1.empty:
            print("✗ Failed to get data")
            return

        print(f"✓ H1 data: {len(df_h1)} samples")

        print("\nFetching H4 data...")
        df_h4 = collector.get_historical_data(SYMBOL, 'H4', bars=10000)
        print(f"✓ H4 data: {len(df_h4)} bars")

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
    labels = labels.loc[features_df.index]

    print(f"\n✓ Created {len(engineer.feature_names)} advanced features")

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'label',
                    'future_close', 'future_return', 'spread', 'real_volume']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols
                    and features_df[c].dtype in ['float64', 'int64', 'int32', 'float32', 'int', 'float']]

    X = features_df[feature_cols].values
    y = labels.values

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Step 3: Train Ensemble
    print("\n" + "=" * 80)
    print("STEP 3: TRAIN ENSEMBLE (XGBoost + RF + GB)")
    print("=" * 80)

    ensemble = EnsembleTrainer()
    metrics = ensemble.train_ensemble(X, y, feature_cols, test_size=0.2,
                                       use_feature_selection=True, top_features=60)

    ensemble.save_model(f'trading_model_{SYMBOL}_H1')

    # Step 4: LSTM (optional)
    print("\n" + "=" * 80)
    print("STEP 4: LSTM (Optional)")
    print("=" * 80)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} available")
        from src.ml.lstm_model import LSTMTrader
        lstm = LSTMTrader()
        lstm.train(X, y, feature_cols, sequence_length=50, epochs=30)
        lstm.save_model(f'lstm_model_{SYMBOL}')
    except ImportError:
        print("✗ TensorFlow not installed - skipping LSTM")
    except Exception as e:
        print(f"✗ LSTM failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n--- ENSEMBLE RESULTS ---")
    print(f"  Majority Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  Unanimous Accuracy:    {metrics.get('accuracy_unanimous', 0):.4f}")
    print(f"  High Conf Accuracy:    {metrics.get('accuracy_high_conf', 0):.4f}")
    print(f"  BEST Accuracy:         {metrics.get('accuracy_best', 0):.4f}")
    print(f"\nRun backtest: python backtest_ml.py")


if __name__ == '__main__':
    main()
