"""
MT5 Trading Bot - CLI Interface

Command-line interface for the trading bot.
"""

import argparse
import sys
import logging


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(
        prog='trading-bot',
        description='MT5 Algorithmic Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py run              # Start the trading bot
  python cli.py api              # Start the API server
  python cli.py backtest         # Run a backtest
  python cli.py status           # Check bot status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Start the trading bot')
    run_parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file path')
    run_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')

    # API command
    api_parser = subparsers.add_parser('api', help='Start the API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', '-p', type=int, default=8000, help='Port to listen on')
    api_parser.add_argument('--reload', '-r', action='store_true', help='Enable auto-reload')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('--strategy', '-s', required=True, help='Strategy name')
    backtest_parser.add_argument('--symbol', default='EURUSD', help='Trading symbol')
    backtest_parser.add_argument('--timeframe', '-t', default='H1', help='Timeframe')
    backtest_parser.add_argument('--bars', '-b', type=int, default=1000, help='Number of bars')
    backtest_parser.add_argument('--balance', type=float, default=10000, help='Initial balance')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check trading bot status')

    # Strategies command
    strategies_parser = subparsers.add_parser('strategies', help='List strategies')
    strategies_parser.add_argument('--enable', help='Enable a strategy')
    strategies_parser.add_argument('--disable', help='Disable a strategy')
    strategies_parser.add_argument('--list-all', action='store_true', help='List all hierarchical strategies')

    # Trade command (live/paper trading)
    trade_parser = subparsers.add_parser('trade', help='Start live or paper trading')
    trade_parser.add_argument('--mode', '-m', choices=['demo', 'real'], default='demo',
                              help='Trading mode (demo or real account)')
    trade_parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')

    # Train ML model command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--symbol', default='GBPUSD', help='Trading symbol')
    train_parser.add_argument('--years', '-y', type=int, default=5, help='Years of data')
    train_parser.add_argument('--lookahead', type=int, default=48, help='Lookahead bars')
    train_parser.add_argument('--threshold', type=int, default=40, help='Pip threshold')

    # ML Backtest command
    ml_backtest_parser = subparsers.add_parser('ml-backtest', help='Backtest ML model')
    ml_backtest_parser.add_argument('--symbol', default='GBPUSD', help='Trading symbol')
    ml_backtest_parser.add_argument('--years', '-y', type=int, default=2, help='Years to backtest')
    ml_backtest_parser.add_argument('--balance', type=float, default=100, help='Initial balance')
    ml_backtest_parser.add_argument('--deposit', type=float, default=100, help='Monthly deposit')

    # Backtest pairs command
    pairs_parser = subparsers.add_parser('backtest-pairs', help='Test strategy combinations')
    pairs_parser.add_argument('--symbol', default='EURUSD', help='Trading symbol')
    pairs_parser.add_argument('--bars', '-b', type=int, default=500, help='Number of bars')
    pairs_parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    pairs_parser.add_argument('--mode', choices=['pa_cp', 'pa_cs', 'full_triplet', 'all'],
                              default='pa_cp', help='Testing mode')
    pairs_parser.add_argument('--top', '-n', type=int, default=20, help='Show top N results')

    return parser


def cmd_run(args):
    """Run the trading bot."""
    print("Starting MT5 Trading Bot...")

    from main import main
    main()


def cmd_api(args):
    """Start the API server."""
    import uvicorn
    print(f"Starting API server on {args.host}:{args.port}...")

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


def cmd_backtest(args):
    """Run a backtest."""
    print(f"Running backtest: {args.strategy} on {args.symbol} ({args.timeframe})")

    from src.backtest import Backtester, DataLoader
    from src.strategies import StrategyManager
    from src.backtest.metrics import PerformanceMetrics
    from src.utils import load_config

    # Load config
    config = load_config()
    strategies_config = config.get('strategies_config', {})

    # Initialize strategy manager
    manager = StrategyManager(strategies_config)
    manager.load_strategies()

    strategy = manager.get_strategy(args.strategy)
    if strategy is None:
        print(f"Error: Strategy '{args.strategy}' not found")
        print(f"Available strategies: {list(manager._strategies.keys())}")
        sys.exit(1)

    # Override symbol
    strategy.symbols = [args.symbol]

    # Load data
    data_loader = DataLoader()
    data = data_loader.generate_sample_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars
    )

    print(f"Loaded {len(data)} bars of data")

    # Run backtest
    backtester = Backtester(initial_balance=args.balance)
    result = backtester.run(strategy, data, args.symbol)

    # Print results
    metrics = PerformanceMetrics()
    report = metrics.format_report(result.metrics)
    print(report)


def cmd_status(args):
    """Check bot status."""
    print("Checking trading bot status...")

    import requests

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API Server: Running")
        else:
            print("✗ API Server: Error")
    except requests.exceptions.ConnectionError:
        print("✗ API Server: Not running")

    # Check MT5 connection
    try:
        from src.core import MT5Handler
        from src.utils import load_config

        config = load_config()
        mt5 = MT5Handler(config.get('mt5', {}))

        if mt5.connect():
            info = mt5.get_account_info()
            print(f"✓ MT5: Connected (Account: {info['login']})")
            print(f"  Balance: ${info['balance']:.2f}")
            print(f"  Equity: ${info['equity']:.2f}")
            mt5.disconnect()
        else:
            print("✗ MT5: Not connected")
    except Exception as e:
        print(f"✗ MT5: Error - {e}")


def cmd_strategies(args):
    """List or manage strategies."""
    from src.strategies import StrategyManager
    from src.utils import load_config

    config = load_config()
    strategies_config = config.get('strategies_config', {})

    manager = StrategyManager(strategies_config)
    manager.load_strategies()

    if args.enable:
        if manager.enable_strategy(args.enable):
            print(f"✓ Strategy '{args.enable}' enabled")
        else:
            print(f"✗ Strategy '{args.enable}' not found")
        return

    if args.disable:
        if manager.disable_strategy(args.disable):
            print(f"✓ Strategy '{args.disable}' disabled")
        else:
            print(f"✗ Strategy '{args.disable}' not found")
        return

    # List strategies
    print("\nAvailable Strategies:")
    print("-" * 50)

    for strategy in manager.get_all_strategies():
        status = "✓ Enabled" if strategy.enabled else "✗ Disabled"
        print(f"  {strategy.name}: {status}")
        print(f"    Symbols: {', '.join(strategy.symbols)}")
        print(f"    Timeframe: {strategy.timeframe}")
        print()


def cmd_backtest_pairs(args):
    """Test all strategy pair combinations."""
    print(f"Testing strategy pairs on {args.symbol} with {args.bars} bars")
    print(f"Mode: {args.mode}")
    print()

    from src.strategies import PairBacktester, PRICE_ACTION_STRATEGIES, CHART_PATTERN_STRATEGIES, CANDLESTICK_STRATEGIES
    from src.backtest import DataLoader

    # Show available strategies
    print(f"Price Action strategies: {len(PRICE_ACTION_STRATEGIES)}")
    print(f"Chart Pattern strategies: {len(CHART_PATTERN_STRATEGIES)}")
    print(f"Candlestick strategies: {len(CANDLESTICK_STRATEGIES)}")
    print()

    # Load data
    data_loader = DataLoader()
    data = data_loader.generate_sample_data(
        symbol=args.symbol,
        timeframe='H1',
        bars=args.bars
    )

    print(f"Loaded {len(data)} bars of data")

    # Create backtester
    pair_tester = PairBacktester(initial_balance=args.balance)

    # Get total pairs count
    pairs = pair_tester.get_all_pairs(args.mode)
    print(f"Testing {len(pairs)} strategy combinations...")
    print()

    # Run tests
    def progress(current, total):
        if current % 10 == 0 or current == total:
            pct = current / total * 100
            print(f"  Progress: {current}/{total} ({pct:.0f}%)")

    results = pair_tester.run_all(data, args.symbol, args.mode, progress)

    # Print report
    print()
    report = pair_tester.get_report(args.top)
    print(report)


def cmd_trade(args):
    """Start live/paper trading with ML model."""
    import os
    from datetime import datetime
    import time

    mode = args.mode
    print(f"=" * 60)
    print(f"ML TRADING BOT - {mode.upper()} MODE")
    print(f"=" * 60)

    # Load environment based on mode
    from dotenv import load_dotenv
    load_dotenv()

    if mode == 'demo':
        login = os.getenv('MT5_DEMO_LOGIN') or os.getenv('MT5_LOGIN')
        password = os.getenv('MT5_DEMO_PASSWORD') or os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_DEMO_SERVER') or os.getenv('MT5_SERVER')
    else:
        login = os.getenv('MT5_REAL_LOGIN')
        password = os.getenv('MT5_REAL_PASSWORD')
        server = os.getenv('MT5_REAL_SERVER')

    if not all([login, password, server]):
        print(f"✗ Missing credentials for {mode} mode")
        print(f"  Set MT5_{mode.upper()}_LOGIN, MT5_{mode.upper()}_PASSWORD, MT5_{mode.upper()}_SERVER in .env")
        return

    print(f"Server: {server}")
    print(f"Mode: {mode}")

    # Connect to MongoDB
    from src.data import Database
    db = Database()
    if not db.connect():
        print("✗ MongoDB connection failed")
        return
    print("✓ MongoDB connected")

    # Connect to MT5
    import MetaTrader5 as mt5
    if not mt5.initialize(login=int(login), password=password, server=server):
        print(f"✗ MT5 connection failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    print(f"✓ MT5 connected - Account: {account.login}")
    print(f"  Balance: ${account.balance:,.2f}")

    # Load ML model
    from src.ml.ensemble_trainer import EnsembleTrainer
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.data_collector import DataCollector

    symbol = 'GBPUSD'
    ensemble = EnsembleTrainer()
    try:
        ensemble.load_model(f'trading_model_{symbol}_H1')
        print(f"✓ ML model loaded")
    except:
        print("✗ Model not found. Run: trading-bot train")
        mt5.shutdown()
        return

    engineer = AdvancedFeatureEngineer()

    # Trading parameters
    RISK_PER_TRADE = 0.02
    MAX_LOT_SIZE = 0.1 if mode == 'demo' else 1.0
    MIN_CONFIDENCE = 0.75
    HOLD_BARS = 48

    print(f"\nTrading Settings:")
    print(f"  Symbol: {symbol}")
    print(f"  Risk: {RISK_PER_TRADE*100}%")
    print(f"  Max Lot: {MAX_LOT_SIZE}")
    print(f"  Min Confidence: {MIN_CONFIDENCE}")

    print(f"\n{'='*60}")
    print("STARTING TRADING LOOP (Ctrl+C to stop)")
    print(f"{'='*60}\n")

    collector = DataCollector([symbol])
    collector.connect()

    try:
        while True:
            now = datetime.now()
            print(f"[{now.strftime('%H:%M:%S')}] Checking signals...")

            # Get data
            df_h1 = collector.get_historical_data(symbol, 'H1', 300)
            df_h4 = collector.get_historical_data(symbol, 'H4', 100)
            df_d1 = collector.get_historical_data(symbol, 'D1', 50)

            if df_h1.empty:
                time.sleep(60)
                continue

            # Create features and predict
            features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
            X = features_df[engineer.get_feature_names()].tail(1).values
            predictions, avg_proba, unanimous = ensemble.predict(X)

            pred = predictions[0]
            proba = avg_proba[0]
            is_unanimous = unanimous[0]
            is_high_conf = proba > MIN_CONFIDENCE or proba < (1 - MIN_CONFIDENCE)

            if is_unanimous and is_high_conf:
                signal_type = 'BUY' if pred == 1 else 'SELL'
                confidence = proba if pred == 1 else 1 - proba

                print(f"  🔔 SIGNAL: {signal_type} | Confidence: {confidence:.2%}")

                # Save signal to MongoDB
                signal_doc = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'price': df_h1['close'].iloc[-1],
                    'mode': mode,
                    'strategy': 'ensemble_ml'
                }
                db.save_signal(signal_doc)

                # Calculate lot size
                balance = mt5.account_info().balance
                lot_size = min((balance * RISK_PER_TRADE) / (40 * 10), MAX_LOT_SIZE)
                lot_size = max(0.01, round(lot_size, 2))

                # Execute trade
                order_type = mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).ask if signal_type == 'BUY' else mt5.symbol_info_tick(symbol).bid

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": f"ML {signal_type}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  ✓ Trade executed: {result.order}")

                    # Save trade to MongoDB
                    trade_doc = {
                        'ticket': result.order,
                        'symbol': symbol,
                        'type': signal_type,
                        'volume': lot_size,
                        'entry_price': price,
                        'mode': mode,
                        'strategy': 'ensemble_ml',
                        'confidence': confidence,
                        'entry_time': datetime.utcnow()
                    }
                    db.save_trade(trade_doc)
                else:
                    print(f"  ✗ Trade failed: {result.comment}")
            else:
                print(f"  No signal (conf={proba:.2%}, unanimous={is_unanimous})")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping trading...")
    finally:
        collector.disconnect()
        mt5.shutdown()
        db.close()


def cmd_train(args):
    """Train ML model."""
    print(f"=" * 60)
    print("TRAINING ML MODEL")
    print(f"=" * 60)

    from src.ml.data_collector import DataCollector
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.ensemble_trainer import EnsembleTrainer

    symbol = args.symbol
    years = args.years
    lookahead = args.lookahead
    threshold = args.threshold

    print(f"Symbol: {symbol}")
    print(f"Years: {years}")
    print(f"Lookahead: {lookahead} bars")
    print(f"Threshold: {threshold} pips")

    # Collect data
    collector = DataCollector([symbol])
    if not collector.connect():
        print("✗ MT5 connection failed")
        return

    try:
        df_h1, labels = collector.prepare_training_data(
            symbol, 'H1', years, lookahead, threshold, binary_only=True
        )
        df_h4 = collector.get_historical_data(symbol, 'H4', 10000)
        df_d1 = collector.get_historical_data(symbol, 'D1', 2000)
    finally:
        collector.disconnect()

    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
    labels = labels.loc[features_df.index]

    feature_cols = [c for c in features_df.columns
                    if c not in ['open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']
                    and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    X = features_df[feature_cols].values
    y = labels.values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Train ensemble
    ensemble = EnsembleTrainer()
    metrics = ensemble.train_ensemble(X, y, feature_cols, top_features=60)
    ensemble.save_model(f'trading_model_{symbol}_H1')

    print(f"\n✓ Model saved: trading_model_{symbol}_H1")
    print(f"  Best accuracy: {metrics.get('accuracy_best', 0):.2%}")


def cmd_ml_backtest(args):
    """Run ML backtest."""
    print(f"=" * 60)
    print("ML BACKTEST")
    print(f"=" * 60)

    from src.ml.data_collector import DataCollector
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.ensemble_trainer import EnsembleTrainer
    import pandas as pd

    symbol = args.symbol
    years = args.years
    initial_balance = args.balance
    monthly_deposit = args.deposit

    print(f"Symbol: {symbol}")
    print(f"Years: {years}")
    print(f"Initial: ${initial_balance}")
    print(f"Monthly deposit: ${monthly_deposit}")

    # Load data
    collector = DataCollector([symbol])
    if not collector.connect():
        print("✗ MT5 connection failed")
        return

    try:
        bars = years * 8760
        df_h1 = collector.get_historical_data(symbol, 'H1', bars)
        df_h4 = collector.get_historical_data(symbol, 'H4', bars // 4)
        df_d1 = collector.get_historical_data(symbol, 'D1', bars // 24)
    finally:
        collector.disconnect()

    # Load model
    ensemble = EnsembleTrainer()
    try:
        ensemble.load_model(f'trading_model_{symbol}_H1')
    except:
        print("✗ Model not found. Run: trading-bot train")
        return

    # Features and predictions
    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
    X = features_df[engineer.get_feature_names()].values
    predictions, avg_proba, unanimous = ensemble.predict(X)

    # Backtest
    high_conf = (avg_proba > 0.75) | (avg_proba < 0.25)
    best_signals = unanimous & high_conf

    balance = initial_balance
    total_deposited = initial_balance
    trades = []
    current_month = None

    HOLD_BARS = 48
    i = 0
    while i < len(features_df) - HOLD_BARS:
        month = features_df.index[i].to_period('M')
        if current_month != month:
            if current_month is not None:
                balance += monthly_deposit
                total_deposited += monthly_deposit
            current_month = month

        if best_signals[i]:
            entry = df_h1.loc[features_df.index[i], 'close']
            exit_price = df_h1.loc[features_df.index[i + HOLD_BARS], 'close']

            pred = predictions[i]
            pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
            pips -= 3  # Spread + slippage

            lot = min((balance * 0.02) / (40 * 10), 10.0)
            lot = max(0.01, round(lot, 2))
            pnl = pips * 10 * lot
            balance += pnl

            trades.append({'pnl': pnl, 'balance': balance})
            i += HOLD_BARS
        else:
            i += 1

    # Results
    wins = sum(1 for t in trades if t['pnl'] > 0)
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total deposited: ${total_deposited:,.2f}")
    print(f"Final balance: ${balance:,.2f}")
    print(f"Net profit: ${balance - total_deposited:,.2f}")
    print(f"Return: {((balance - total_deposited) / total_deposited) * 100:.1f}%")
    print(f"Trades: {len(trades)} | Wins: {wins} | Win rate: {wins/len(trades)*100:.1f}%")


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Execute command
    commands = {
        'run': cmd_run,
        'api': cmd_api,
        'backtest': cmd_backtest,
        'status': cmd_status,
        'strategies': cmd_strategies,
        'backtest-pairs': cmd_backtest_pairs,
        'trade': cmd_trade,
        'train': cmd_train,
        'ml-backtest': cmd_ml_backtest
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

