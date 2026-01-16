#!/usr/bin/env python
"""
Backtest Advanced ML Models

Uses ensemble predictions with ultra-high confidence filtering.
Only trades when ALL models agree with high confidence.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from src.ml.data_collector import DataCollector
from src.ml.advanced_features import AdvancedFeatureEngineer
from src.ml.ensemble_trainer import EnsembleTrainer


def main():
    print("=" * 80)
    print("ADVANCED ML BACKTESTER")
    print("=" * 80)
    print("\nUsing: Ensemble + Ultra-High Confidence Filtering")

    # Configuration
    SYMBOL = 'GBPUSD'
    BARS_H1 = 8760  # 1 year H1
    BARS_H4 = 2200  # ~1 year H4
    BARS_D1 = 400   # ~1 year D1

    # Trading parameters
    HOLD_BARS = 48  # Hold for 48 hours
    MIN_CONFIDENCE = 0.70  # Only trade when avg proba > 0.70 or < 0.30
    REQUIRE_UNANIMOUS = True  # Require all 3 models to agree
    LOT_SIZE = 1.0  # 1 lot = $10 per pip

    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Hold period: {HOLD_BARS} bars")
    print(f"  Min confidence: {MIN_CONFIDENCE}")
    print(f"  Require unanimous: {REQUIRE_UNANIMOUS}")
    print(f"  Lot size: {LOT_SIZE}")

    # Connect and get data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    collector = DataCollector([SYMBOL])
    if not collector.connect():
        print("✗ Failed to connect to MT5")
        return

    try:
        import MetaTrader5 as mt5
        account = mt5.account_info()
        initial_balance = account.balance
        print(f"✓ Connected - Balance: ${initial_balance:,.2f}")

        # Get multi-timeframe data
        print(f"\nFetching data...")
        df_h1 = collector.get_historical_data(SYMBOL, 'H1', BARS_H1)
        df_h4 = collector.get_historical_data(SYMBOL, 'H4', BARS_H4)
        df_d1 = collector.get_historical_data(SYMBOL, 'D1', BARS_D1)

        print(f"✓ H1: {len(df_h1)} bars")
        print(f"✓ H4: {len(df_h4)} bars")
        print(f"✓ D1: {len(df_d1)} bars")

    finally:
        collector.disconnect()

    # Load ensemble model
    print("\n" + "=" * 80)
    print("LOADING ENSEMBLE MODEL")
    print("=" * 80)

    ensemble = EnsembleTrainer()
    try:
        ensemble.load_model('ensemble_model_GBPUSD')
    except Exception as e:
        print(f"✗ Model not found: {e}")
        print("  Run train_advanced.py first!")
        return

    # Create features
    print("\n" + "=" * 80)
    print("CREATING FEATURES")
    print("=" * 80)

    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)

    print(f"✓ Features: {len(features_df)} bars")

    # Prepare feature matrix
    feature_names = ensemble.feature_names

    # Check for missing features
    missing = [f for f in feature_names if f not in features_df.columns]
    if missing:
        print(f"⚠ Missing {len(missing)} features")
        # Use intersection
        feature_names = [f for f in feature_names if f in features_df.columns]

    X_all = features_df[feature_names].values

    # Get ensemble predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    predictions, avg_proba, unanimous = ensemble.predict(X_all)

    print(f"✓ Total predictions: {len(predictions)}")
    print(f"  BUY predictions: {(predictions == 1).sum()}")
    print(f"  SELL predictions: {(predictions == 0).sum()}")
    print(f"  Unanimous predictions: {unanimous.sum()}")

    # High confidence mask
    high_conf = (avg_proba > MIN_CONFIDENCE) | (avg_proba < (1 - MIN_CONFIDENCE))
    print(f"  High confidence (>{MIN_CONFIDENCE}): {high_conf.sum()}")

    # Best signals: unanimous + high confidence
    best_signals = unanimous & high_conf
    print(f"  BEST signals (unanimous + high conf): {best_signals.sum()}")

    # Backtest
    print("\n" + "=" * 80)
    print("RUNNING BACKTEST")
    print("=" * 80)

    balance = initial_balance
    trades = []

    i = 0
    while i < len(features_df) - HOLD_BARS:
        current_time = features_df.index[i]

        # Check if this is a tradeable signal
        if best_signals[i]:
            entry_price = df_h1.loc[current_time, 'close']

            # Get exit after HOLD_BARS
            exit_idx = i + HOLD_BARS
            exit_time = features_df.index[exit_idx]
            exit_price = df_h1.loc[exit_time, 'close']

            # Determine direction based on prediction
            pred = predictions[i]
            conf = avg_proba[i]

            if pred == 1:  # BUY
                pips = (exit_price - entry_price) * 10000
                trade_type = 'BUY'
            else:  # SELL
                pips = (entry_price - exit_price) * 10000
                trade_type = 'SELL'

            pnl = pips * 10 * LOT_SIZE
            balance += pnl

            trades.append({
                'entry_time': current_time,
                'exit_time': exit_time,
                'type': trade_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': pips,
                'pnl': pnl,
                'confidence': conf if pred == 1 else 1 - conf
            })

            # Skip ahead
            i += HOLD_BARS
        else:
            i += 1

    # Results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    if len(trades) == 0:
        print("\n⚠ No trades generated!")
        print("Try lowering MIN_CONFIDENCE or REQUIRE_UNANIMOUS")
        return

    trades_df = pd.DataFrame(trades)

    total_profit = trades_df['pnl'].sum()
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100

    print(f"\nSymbol: {SYMBOL}")
    print(f"Period: {features_df.index[0].date()} to {features_df.index[-1].date()}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Total Profit/Loss: ${total_profit:,.2f}")
    print(f"Return: {(total_profit/initial_balance)*100:.2f}%")

    print(f"\nTotal Trades: {len(trades_df)}")
    print(f"Winning Trades: {len(wins)}")
    print(f"Losing Trades: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")

    if len(wins) > 0:
        print(f"\nAverage Win: ${wins['pnl'].mean():,.2f} ({wins['pips'].mean():.1f} pips)")
        print(f"Biggest Win: ${wins['pnl'].max():,.2f}")
    if len(losses) > 0:
        print(f"Average Loss: ${losses['pnl'].mean():,.2f} ({losses['pips'].mean():.1f} pips)")
        print(f"Biggest Loss: ${losses['pnl'].min():,.2f}")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"Profit Factor: {profit_factor:.2f}")

    # Monthly breakdown
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({'pnl': ['sum', 'count']})
    monthly.columns = ['profit', 'trades']

    print("\n" + "=" * 80)
    print("MONTHLY PERFORMANCE")
    print("=" * 80)

    running_balance = initial_balance
    print(f"\n{'Month':<12} {'Trades':<10} {'Profit':<15} {'Return%':<12} {'Balance':<15}")
    print("-" * 70)

    for month, row in monthly.iterrows():
        monthly_return = (row['profit'] / running_balance) * 100
        running_balance += row['profit']
        status = "✓" if row['profit'] > 0 else "✗"
        print(f"{str(month):<12} {int(row['trades']):<10} ${row['profit']:>12,.2f} {monthly_return:>10.2f}% ${running_balance:>12,.2f} {status}")

    print("-" * 70)
    total_return = (total_profit / initial_balance) * 100
    monthly_avg = total_return / max(len(monthly), 1)
    print(f"{'TOTAL':<12} {len(trades_df):<10} ${total_profit:>12,.2f} {total_return:>10.2f}%")
    print(f"\nAverage Monthly Return: {monthly_avg:.2f}%")

    # Save trades
    os.makedirs('backtest_reports', exist_ok=True)
    trades_df.to_csv('backtest_reports/advanced_backtest_trades.csv', index=False)
    print(f"\n✓ Trade log saved to: backtest_reports/advanced_backtest_trades.csv")


if __name__ == '__main__':
    main()
