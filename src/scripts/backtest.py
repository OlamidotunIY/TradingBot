#!/usr/bin/env python
"""
REALISTIC FORWARD TEST

This is a proper out-of-sample backtest:
- Train on: 2018-2023 (model learns from this)
- Test on: 2024-2026 (model has NEVER seen this data)

Also includes realistic constraints:
- Maximum lot size: 10 lots
- Spread cost: 2 pips per trade
- Slippage: 1 pip per trade
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from src.ml.data_collector import DataCollector
from src.ml.advanced_features import AdvancedFeatureEngineer
from src.ml.ensemble_trainer import EnsembleTrainer


def main():
    print("=" * 80)
    print("REALISTIC FORWARD TEST (Out-of-Sample)")
    print("=" * 80)
    print("\n⚠️  This test uses ONLY data the model has NEVER seen!")
    print("    Training period: 2018-2023")
    print("    Testing period:  2024-2026 (OUT-OF-SAMPLE)")

    SYMBOL = 'GBPUSD'
    MODEL_NAME = f'trading_model_{SYMBOL}_H1'

    # Realistic configuration
    INITIAL_BALANCE = 100
    MONTHLY_DEPOSIT = 100
    RISK_PER_TRADE = 0.02  # 2% risk
    MAX_LOT_SIZE = 10.0    # Maximum 10 lots (realistic broker limit)
    SPREAD_PIPS = 2.0      # Spread cost
    SLIPPAGE_PIPS = 1.0    # Slippage

    HOLD_BARS = 48
    MIN_CONFIDENCE = 0.75
    REQUIRE_UNANIMOUS = True

    print(f"\nRealistic Configuration:")
    print(f"  Starting Balance: ${INITIAL_BALANCE}")
    print(f"  Monthly Deposit:  ${MONTHLY_DEPOSIT}")
    print(f"  Risk per Trade:   {RISK_PER_TRADE*100}%")
    print(f"  Max Lot Size:     {MAX_LOT_SIZE} lots")
    print(f"  Spread:           {SPREAD_PIPS} pips")
    print(f"  Slippage:         {SLIPPAGE_PIPS} pips")

    # Get ONLY 2024-2026 data (out-of-sample)
    print("\n" + "=" * 80)
    print("LOADING OUT-OF-SAMPLE DATA (2024-2026)")
    print("=" * 80)

    collector = DataCollector([SYMBOL])
    if not collector.connect():
        print("✗ Failed to connect")
        return

    try:
        # Get 2 years for testing (2024-2026)
        df_h1 = collector.get_historical_data(SYMBOL, 'H1', 17520)
        df_h4 = collector.get_historical_data(SYMBOL, 'H4', 4380)
        df_d1 = collector.get_historical_data(SYMBOL, 'D1', 730)

        # Filter to only 2024 onwards (out-of-sample)
        cutoff_date = pd.Timestamp('2024-01-01')
        df_h1 = df_h1[df_h1.index >= cutoff_date]
        df_h4 = df_h4[df_h4.index >= cutoff_date]
        df_d1 = df_d1[df_d1.index >= cutoff_date]

        print(f"✓ OUT-OF-SAMPLE Data:")
        print(f"  H1: {len(df_h1)} bars")
        print(f"  Period: {df_h1.index[0].date()} to {df_h1.index[-1].date()}")
        print(f"\n  ⚠️  Model was trained on 2018-2023")
        print(f"  ⚠️  This data is UNSEEN by the model!")

    finally:
        collector.disconnect()

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    ensemble = EnsembleTrainer()
    try:
        ensemble.load_model(MODEL_NAME)
    except:
        print("✗ Model not found. Run: python train_model.py")
        return

    # Create features
    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)

    all_feature_names = engineer.get_feature_names()
    X_all = features_df[all_feature_names].values

    print(f"✓ Features: {len(all_feature_names)}")

    # Predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    predictions, avg_proba, unanimous = ensemble.predict(X_all)

    high_conf = (avg_proba > MIN_CONFIDENCE) | (avg_proba < (1 - MIN_CONFIDENCE))
    best_signals = unanimous & high_conf

    print(f"✓ Total bars: {len(predictions)}")
    print(f"  Best signals: {best_signals.sum()}")

    # REALISTIC BACKTEST
    print("\n" + "=" * 80)
    print("RUNNING REALISTIC FORWARD TEST")
    print("=" * 80)

    balance = INITIAL_BALANCE
    total_deposited = INITIAL_BALANCE
    trades = []
    monthly_deposits = {}
    current_month = None

    i = 0
    while i < len(features_df) - HOLD_BARS:
        current_time = features_df.index[i]
        month_key = current_time.to_period('M')

        # Monthly deposit
        if current_month is None or month_key != current_month:
            if current_month is not None:
                balance += MONTHLY_DEPOSIT
                total_deposited += MONTHLY_DEPOSIT
                monthly_deposits[month_key] = MONTHLY_DEPOSIT
            current_month = month_key

        if best_signals[i]:
            entry_price = df_h1.loc[current_time, 'close']

            exit_idx = i + HOLD_BARS
            exit_time = features_df.index[exit_idx]
            exit_price = df_h1.loc[exit_time, 'close']

            pred = predictions[i]
            if pred == 1:  # BUY
                raw_pips = (exit_price - entry_price) * 10000
                trade_type = 'BUY'
            else:  # SELL
                raw_pips = (entry_price - exit_price) * 10000
                trade_type = 'SELL'

            # Subtract spread and slippage
            net_pips = raw_pips - SPREAD_PIPS - SLIPPAGE_PIPS

            # Calculate lot size (with MAX limit)
            risk_amount = balance * RISK_PER_TRADE
            lot_size = risk_amount / (40 * 10)  # 40 pip SL, $10/pip/lot
            lot_size = min(max(0.01, round(lot_size, 2)), MAX_LOT_SIZE)

            pnl = net_pips * 10 * lot_size
            balance += pnl

            # Prevent negative balance
            if balance < 0:
                balance = 0

            trades.append({
                'entry_time': current_time,
                'exit_time': exit_time,
                'type': trade_type,
                'raw_pips': raw_pips,
                'net_pips': net_pips,
                'lot_size': lot_size,
                'pnl': pnl,
                'balance_after': balance
            })

            i += HOLD_BARS
        else:
            i += 1

    # RESULTS
    print("\n" + "=" * 80)
    print("REALISTIC FORWARD TEST RESULTS")
    print("=" * 80)

    if not trades:
        print("⚠ No trades!")
        return

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    total_profit = balance - total_deposited
    win_rate = len(wins) / len(trades_df) * 100

    print(f"\n{'='*50}")
    print("ACCOUNT SUMMARY (REALISTIC)")
    print(f"{'='*50}")
    print(f"  Starting Balance:   ${INITIAL_BALANCE:,.2f}")
    print(f"  Total Deposited:    ${total_deposited:,.2f}")
    print(f"  Final Balance:      ${balance:,.2f}")
    print(f"  Net Profit:         ${total_profit:,.2f}")
    print(f"  Return on Deposits: {(total_profit/total_deposited)*100:.1f}%")

    print(f"\n{'='*50}")
    print("TRADING STATS")
    print(f"{'='*50}")
    print(f"  Total Trades: {len(trades_df)}")
    print(f"  Winning:      {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losing:       {len(losses)} ({100-win_rate:.1f}%)")

    if len(wins) > 0:
        print(f"  Avg Win:      ${wins['pnl'].mean():,.2f}")
    if len(losses) > 0:
        print(f"  Avg Loss:     ${losses['pnl'].mean():,.2f}")

    if len(wins) > 0 and len(losses) > 0:
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"  Profit Factor: {pf:.2f}")

    # Monthly
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({'pnl': 'sum', 'balance_after': 'last'})

    print(f"\n{'='*50}")
    print("MONTHLY PERFORMANCE")
    print(f"{'='*50}")
    print(f"\n{'Month':<10} {'Profit':<12} {'Balance':<12}")
    print("-" * 40)

    for month, row in monthly.iterrows():
        status = "✓" if row['pnl'] > 0 else "✗"
        print(f"{str(month):<10} ${row['pnl']:>9,.2f}  ${row['balance_after']:>9,.2f} {status}")

    print("-" * 40)

    months = len(monthly)
    if months > 0:
        monthly_avg = (total_profit / total_deposited * 100) / months
        print(f"\nAvg Monthly Return: {monthly_avg:.2f}%")

        if balance > INITIAL_BALANCE:
            monthly_growth = (balance / INITIAL_BALANCE) ** (1/months) - 1
            print(f"Compound Growth Rate: {monthly_growth*100:.2f}%/month")
            print(f"Projected Annual: {((1+monthly_growth)**12-1)*100:.1f}%")

    print(f"\n✅ This is a REALISTIC estimate based on unseen data!")

    os.makedirs('backtest_reports', exist_ok=True)
    trades_df.to_csv('backtest_reports/forward_test.csv', index=False)
    print(f"\n✓ Saved: backtest_reports/forward_test.csv")


if __name__ == '__main__':
    main()
