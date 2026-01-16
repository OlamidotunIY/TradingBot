#!/usr/bin/env python
"""
PAPER TRADING BOT

Runs the ML model in real-time on MT5 demo account.
Generates signals but only logs them (or executes on demo).

Usage:
    python paper_trade.py          # Log signals only
    python paper_trade.py --live   # Execute on demo account
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
from datetime import datetime
import pandas as pd

from src.ml.data_collector import DataCollector
from src.ml.advanced_features import AdvancedFeatureEngineer
from src.ml.ensemble_trainer import EnsembleTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Execute trades on demo account')
    args = parser.parse_args()

    print("=" * 60)
    print("ML PAPER TRADING BOT")
    print("=" * 60)

    SYMBOL = 'GBPUSD'
    MODEL_NAME = f'trading_model_{SYMBOL}_H1'
    MIN_CONFIDENCE = 0.75
    CHECK_INTERVAL = 60  # Check every 60 seconds

    # Risk settings
    RISK_PER_TRADE = 0.02
    MAX_LOT_SIZE = 0.1  # Start small for paper trading

    mode = "LIVE (Demo Account)" if args.live else "SIGNAL LOGGING ONLY"
    print(f"\nMode: {mode}")
    print(f"Symbol: {SYMBOL}")
    print(f"Min Confidence: {MIN_CONFIDENCE}")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}%")

    # Load model
    print("\nLoading model...")
    ensemble = EnsembleTrainer()
    try:
        ensemble.load_model(MODEL_NAME)
    except Exception as e:
        print(f"✗ Model not found: {e}")
        return

    # Connect to MT5
    print("\nConnecting to MT5...")
    collector = DataCollector([SYMBOL])
    if not collector.connect():
        print("✗ Failed to connect to MT5")
        return

    import MetaTrader5 as mt5
    account = mt5.account_info()
    print(f"✓ Connected - Account: {account.login}")
    print(f"  Balance: ${account.balance:,.2f}")
    print(f"  Equity: ${account.equity:,.2f}")

    # Trading state
    current_position = None
    trade_log = []

    engineer = AdvancedFeatureEngineer()

    print("\n" + "=" * 60)
    print("STARTING PAPER TRADING LOOP")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            now = datetime.now()
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Checking for signals...")

            # Get latest data
            df_h1 = collector.get_historical_data(SYMBOL, 'H1', 300)
            df_h4 = collector.get_historical_data(SYMBOL, 'H4', 100)
            df_d1 = collector.get_historical_data(SYMBOL, 'D1', 50)

            if df_h1.empty:
                print("  ⚠ No data received")
                time.sleep(CHECK_INTERVAL)
                continue

            # Create features
            features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)

            if features_df.empty or len(features_df) < 10:
                print("  ⚠ Insufficient features")
                time.sleep(CHECK_INTERVAL)
                continue

            # Get latest bar features
            all_feature_names = engineer.get_feature_names()
            X = features_df[all_feature_names].tail(1).values

            # Predict
            predictions, avg_proba, unanimous = ensemble.predict(X)

            pred = predictions[0]
            proba = avg_proba[0]
            is_unanimous = unanimous[0]

            # Determine signal
            is_high_conf = proba > MIN_CONFIDENCE or proba < (1 - MIN_CONFIDENCE)

            current_price = df_h1['close'].iloc[-1]
            atr = df_h1['high'].iloc[-20:].max() - df_h1['low'].iloc[-20:].min()

            if is_unanimous and is_high_conf:
                signal = 'BUY' if pred == 1 else 'SELL'
                confidence = proba if pred == 1 else 1 - proba

                print(f"\n  🔔 SIGNAL: {signal}")
                print(f"     Confidence: {confidence:.2%}")
                print(f"     Price: {current_price}")
                print(f"     Unanimous: {is_unanimous}")

                # Calculate lot size
                balance = mt5.account_info().balance
                risk_amount = balance * RISK_PER_TRADE
                sl_pips = 40
                lot_size = min(risk_amount / (sl_pips * 10), MAX_LOT_SIZE)
                lot_size = max(0.01, round(lot_size, 2))

                # Calculate SL/TP
                if signal == 'BUY':
                    sl = current_price - (atr * 0.5)
                    tp = current_price + (atr * 1.0)
                else:
                    sl = current_price + (atr * 0.5)
                    tp = current_price - (atr * 1.0)

                log_entry = {
                    'time': now,
                    'signal': signal,
                    'price': current_price,
                    'confidence': confidence,
                    'lot_size': lot_size,
                    'sl': sl,
                    'tp': tp
                }
                trade_log.append(log_entry)

                if args.live and current_position is None:
                    # Execute trade on demo
                    order_type = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": SYMBOL,
                        "volume": lot_size,
                        "type": order_type,
                        "price": mt5.symbol_info_tick(SYMBOL).ask if signal == 'BUY' else mt5.symbol_info_tick(SYMBOL).bid,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": f"ML {signal}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"     ✓ Order executed: {result.order}")
                        current_position = signal
                    else:
                        print(f"     ✗ Order failed: {result.comment}")

                # Save log
                pd.DataFrame(trade_log).to_csv('backtest_reports/paper_trade_log.csv', index=False)

            else:
                print(f"  No signal (conf={proba:.2%}, unanimous={is_unanimous})")

            # Check current position
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions:
                pos = positions[0]
                profit = pos.profit
                print(f"\n  📊 Open Position: {pos.type} | Profit: ${profit:,.2f}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopping paper trading...")
    finally:
        collector.disconnect()

        if trade_log:
            df = pd.DataFrame(trade_log)
            df.to_csv('backtest_reports/paper_trade_log.csv', index=False)
            print(f"\n✓ Trade log saved: backtest_reports/paper_trade_log.csv")
            print(f"  Total signals: {len(trade_log)}")


if __name__ == '__main__':
    main()
