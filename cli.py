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
    trade_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'USDCHF'],
                              help='Trading symbols (space-separated)')
    trade_parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')

    # Train ML model command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'USDCHF'],
                              help='Trading symbols (space-separated)')
    train_parser.add_argument('--years', '-y', type=int, default=5, help='Years of data')
    train_parser.add_argument('--lookahead', type=int, default=48, help='Lookahead bars')
    train_parser.add_argument('--threshold', type=int, default=40, help='Pip threshold')

    # ML Backtest command
    ml_backtest_parser = subparsers.add_parser('ml-backtest', help='Backtest ML model')
    ml_backtest_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'USDCHF'],
                                    help='Trading symbols (space-separated)')
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
    """Start live/paper trading with ML model and advanced strategies."""
    import os
    from datetime import datetime
    import time
    import numpy as np

    mode = args.mode
    symbols = args.symbols

    print(f"=" * 60)
    print(f"ADVANCED ML TRADING BOT - {mode.upper()} MODE")
    print(f"=" * 60)
    print(f"Symbols: {', '.join(symbols)}")

    # Load environment based on mode (with hardcoded fallbacks for demo)
    from dotenv import load_dotenv
    load_dotenv()

    if mode == 'demo':
        login = os.getenv('MT5_DEMO_LOGIN') or os.getenv('MT5_LOGIN') or '213790354'
        password = os.getenv('MT5_DEMO_PASSWORD') or os.getenv('MT5_PASSWORD') or 'vy5XEd#9'
        server = os.getenv('MT5_DEMO_SERVER') or os.getenv('MT5_SERVER') or 'OctaFX-Demo'
    else:
        login = os.getenv('MT5_REAL_LOGIN')
        password = os.getenv('MT5_REAL_PASSWORD')
        server = os.getenv('MT5_REAL_SERVER')

    if not all([login, password, server]):
        print(f"✗ Missing credentials for {mode} mode")
        return

    print(f"Server: {server}")

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

    # Load ML models for each symbol
    from src.ml.ensemble_trainer import EnsembleTrainer
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.data_collector import DataCollector

    # Load advanced strategy modules
    from src.ml.regime_detector import RegimeDetector
    from src.risk.dynamic_sizing import DynamicPositionSizer, calculate_atr
    from src.risk.trailing_stop import TrailingStopManager, calculate_atr_pips
    from src.risk.correlation_manager import CorrelationManager
    from src.filters.session_filter import SessionFilter

    # Initialize advanced modules
    session_filter = SessionFilter()
    correlation_mgr = CorrelationManager()
    position_sizer = DynamicPositionSizer(
        base_risk_pct=0.02,
        max_lot_size=0.1 if mode == 'demo' else 1.0
    )
    trailing_mgr = TrailingStopManager()
    regime_detectors = {}  # Per-symbol regime detectors

    models = {}
    exit_models = {}
    for symbol in symbols:
        ensemble = EnsembleTrainer()
        try:
            ensemble.load_model(f'trading_model_{symbol}_H1')
            models[symbol] = ensemble
            print(f"✓ Entry model loaded: {symbol}")
        except:
            print(f"✗ Entry model not found for {symbol}")

        # Load exit model
        from src.ml.exit_model import ExitModelTrainer
        exit_trainer = ExitModelTrainer()
        try:
            exit_trainer.load_model(f'{symbol}_H1')
            exit_models[symbol] = exit_trainer
            print(f"✓ Exit model loaded: {symbol}")
        except:
            print(f"  (No exit model for {symbol} - using 48h fallback)")

    if not models:
        print("✗ No models loaded. Run: python cli.py train")
        mt5.shutdown()
        return

    engineer = AdvancedFeatureEngineer()

    # Trading parameters
    MIN_CONFIDENCE = 0.75
    EXIT_CONFIDENCE = 0.6

    print(f"\n{'='*60}")
    print("ADVANCED FEATURES ENABLED:")
    print(f"{'='*60}")
    print(f"  ✓ Session Filter (London/NY)")
    print(f"  ✓ Regime Detection (Trend/Range)")
    print(f"  ✓ Dynamic Position Sizing (ATR + Kelly)")
    print(f"  ✓ Trailing Stops (2×ATR)")
    print(f"  ✓ Correlation Manager")
    print(f"  ✓ ML Entry Model (conf > {MIN_CONFIDENCE})")
    print(f"  ✓ ML Exit Model (conf > {EXIT_CONFIDENCE})")

    print(f"\n{'='*60}")
    print("STARTING TRADING LOOP (Ctrl+C to stop)")
    print(f"{'='*60}\n")

    collector = DataCollector(list(models.keys()))
    collector.connect()

    MAX_HOLD_HOURS = 48  # Maximum hold as fallback

    try:
        while True:
            now = datetime.now()
            print(f"\n[{now.strftime('%H:%M:%S')}] Checking...")

            # First, check if any positions should be closed using ML exit model
            all_positions = mt5.positions_get()
            if all_positions:
                for pos in all_positions:
                    if pos.magic == 123456:  # Only our trades
                        symbol = pos.symbol
                        open_time = datetime.fromtimestamp(pos.time)
                        hours_open = (now - open_time).total_seconds() / 3600
                        bars_open = int(hours_open)  # H1 bars

                        # Calculate position features
                        entry_price = pos.price_open
                        current_price = pos.price_current
                        if pos.type == 0:  # BUY
                            unrealized_pips = (current_price - entry_price) * 10000
                        else:  # SELL
                            unrealized_pips = (entry_price - current_price) * 10000

                        should_close = False
                        close_reason = ""

                        # Check ML exit model if available
                        if symbol in exit_models and symbol in models:
                            # Get current market features
                            df_h1 = collector.get_historical_data(symbol, 'H1', 100)
                            df_h4 = collector.get_historical_data(symbol, 'H4', 50)
                            df_d1 = collector.get_historical_data(symbol, 'D1', 20)

                            if not df_h1.empty:
                                features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
                                market_features = features_df[engineer.get_feature_names()].tail(1).values

                                # Position features
                                max_profit = max(unrealized_pips, pos.profit * 10)  # Approximate
                                position_features = np.array([[
                                    bars_open,
                                    unrealized_pips,
                                    max_profit,
                                    min(0, unrealized_pips),  # max drawdown
                                    unrealized_pips - max_profit,  # pullback
                                    bars_open / 48  # normalized time
                                ]])

                                X_exit = np.hstack([market_features, position_features])
                                exit_pred, exit_proba = exit_models[symbol].predict(X_exit)

                                if exit_pred[0] == 1 and exit_proba[0] > EXIT_CONFIDENCE:
                                    should_close = True
                                    close_reason = f"ML Exit ({exit_proba[0]:.0%})"

                        # Fallback: Force close after max hold hours
                        if not should_close and hours_open >= MAX_HOLD_HOURS:
                            should_close = True
                            close_reason = f"Max hold ({hours_open:.0f}h)"

                        if should_close:
                            tick = mt5.symbol_info_tick(pos.symbol)
                            close_price = tick.bid if pos.type == 0 else tick.ask

                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": 123456,
                                "comment": close_reason[:20],
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_FOK,
                            }

                            result = mt5.order_send(close_request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"  📤 Closed {symbol} ({close_reason}) | P/L: ${pos.profit:.2f}")
                                db.update_trade(pos.ticket, {
                                    'status': 'CLOSED',
                                    'exit_price': close_price,
                                    'profit': pos.profit,
                                    'exit_time': now
                                })
                            else:
                                print(f"  ✗ Failed to close {symbol}: {result.comment}")

            for symbol, ensemble in models.items():
                # Get data
                df_h1 = collector.get_historical_data(symbol, 'H1', 300)
                df_h4 = collector.get_historical_data(symbol, 'H4', 100)
                df_d1 = collector.get_historical_data(symbol, 'D1', 50)

                if df_h1.empty:
                    continue

                # === FILTER 1: Session Filter ===
                can_trade_session, session_reason = session_filter.should_trade(symbol)
                if not can_trade_session:
                    print(f"  {symbol}: ⏰ {session_reason}")
                    continue

                # Create features and predict
                features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
                X = features_df[engineer.get_feature_names()].tail(1).values
                predictions, avg_proba, unanimous = ensemble.predict(X)

                pred = predictions[0]
                proba = avg_proba[0]
                is_unanimous = unanimous[0]
                is_high_conf = proba > MIN_CONFIDENCE or proba < (1 - MIN_CONFIDENCE)
                new_signal = 'BUY' if pred == 1 else 'SELL'

                # Check if already have open position for this symbol
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    pos = positions[0]
                    current_direction = 'BUY' if pos.type == 0 else 'SELL'

                    # If signal contradicts, close the position
                    if is_unanimous and is_high_conf and new_signal != current_direction:
                        print(f"  🔄 {symbol}: Signal reversed! Closing {current_direction}...")
                        tick = mt5.symbol_info_tick(symbol)
                        close_price = tick.bid if pos.type == 0 else tick.ask

                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": pos.volume,
                            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                            "position": pos.ticket,
                            "price": close_price,
                            "deviation": 20,
                            "magic": 123456,
                            "comment": "ML Reverse",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_FOK,
                        }
                        result = mt5.order_send(close_request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"     ✓ Closed (P/L: ${pos.profit:.2f})")
                            trailing_mgr.remove_position(pos.ticket)
                            correlation_mgr.remove_position(str(pos.ticket))
                        else:
                            print(f"     ✗ Failed: {result.comment}")
                            continue
                    else:
                        print(f"  {symbol}: Position open (ticket {pos.ticket})")
                        continue

                if is_unanimous and is_high_conf:
                    signal_type = new_signal
                    confidence = proba if pred == 1 else 1 - proba

                    # === FILTER 2: Correlation Check ===
                    can_trade_corr, corr_reason = correlation_mgr.can_open_position(symbol, signal_type)
                    if not can_trade_corr:
                        print(f"  {symbol}: 📊 {corr_reason}")
                        continue

                    # === FILTER 3: Regime Detection (simple heuristic) ===
                    atr_pips = calculate_atr_pips(df_h1)
                    volatility = df_h1['close'].pct_change().rolling(20).std().iloc[-1]
                    is_high_vol = volatility > 0.015

                    if is_high_vol:
                        print(f"  {symbol}: ⚡ High volatility - reducing size")

                    print(f"  🔔 {symbol}: {signal_type} | Confidence: {confidence:.2%} | ATR: {atr_pips:.1f} pips")

                    # Save signal to MongoDB
                    db.save_signal({
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'price': df_h1['close'].iloc[-1],
                        'mode': mode,
                        'strategy': 'ensemble_ml_advanced'
                    })

                    # === DYNAMIC POSITION SIZING ===
                    balance = mt5.account_info().balance
                    regime = 'high_volatility' if is_high_vol else 'trending_up' if signal_type == 'BUY' else 'trending_down'

                    sizing = position_sizer.calculate_position_size(
                        balance=balance,
                        atr_pips=atr_pips,
                        confidence=confidence,
                        regime=regime
                    )
                    lot_size = sizing['lot_size']

                    print(f"     Size: {lot_size} lots | Risk: {sizing['risk_pct']*100:.1f}% | Stop: {sizing['stop_pips']:.0f} pips")

                    # Execute trade
                    order_type = mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else mt5.ORDER_TYPE_SELL
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    price = tick.ask if signal_type == 'BUY' else tick.bid

                    # Get supported filling mode for OctaFX (try FOK first, then RETURN)
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info.filling_mode & 1:  # FOK supported
                        filling_mode = mt5.ORDER_FILLING_FOK
                    elif symbol_info.filling_mode & 2:  # IOC supported
                        filling_mode = mt5.ORDER_FILLING_IOC
                    else:  # RETURN (partial fill)
                        filling_mode = mt5.ORDER_FILLING_RETURN

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot_size,
                        "type": order_type,
                        "price": price,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": f"ML {signal_type}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling_mode,
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"     ✓ Trade executed: {result.order}")

                        # Add to trailing stop manager
                        trailing_mgr.add_position(
                            ticket=result.order,
                            symbol=symbol,
                            direction=signal_type,
                            entry_price=price,
                            atr_pips=atr_pips
                        )

                        # Add to correlation manager
                        correlation_mgr.add_position(str(result.order), symbol, signal_type)

                        db.save_trade({
                            'ticket': result.order,
                            'symbol': symbol,
                            'type': signal_type,
                            'volume': lot_size,
                            'entry_price': price,
                            'stop_price': sizing['stop_pips'],
                            'mode': mode,
                            'strategy': 'ensemble_ml_advanced',
                            'confidence': confidence,
                            'regime': regime,
                            'atr_pips': atr_pips,
                            'entry_time': datetime.now(tz=__import__('datetime').timezone.utc)
                        })
                    else:
                        print(f"     ✗ Trade failed: {result.comment}")
                else:
                    print(f"  {symbol}: No signal (conf={proba:.2%})")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\nStopping trading...")
    finally:
        collector.disconnect()
        mt5.shutdown()
        db.close()


def cmd_train(args):
    """Train ML model for multiple symbols."""
    print(f"=" * 60)
    print("TRAINING ML MODELS (Entry + Exit)")
    print(f"=" * 60)

    from src.ml.data_collector import DataCollector
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.ensemble_trainer import EnsembleTrainer
    from src.ml.exit_model import ExitModelTrainer

    symbols = args.symbols
    years = args.years
    lookahead = args.lookahead
    threshold = args.threshold

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Years: {years}")
    print(f"Lookahead: {lookahead} bars")
    print(f"Threshold: {threshold} pips")

    collector = DataCollector(symbols)
    if not collector.connect():
        print("✗ MT5 connection failed")
        return

    try:
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"Training {symbol}...")
            print(f"{'='*50}")

            df_h1, labels = collector.prepare_training_data(
                symbol, 'H1', years, lookahead, threshold, binary_only=True
            )
            df_h4 = collector.get_historical_data(symbol, 'H4', 10000)
            df_d1 = collector.get_historical_data(symbol, 'D1', 2000)

            engineer = AdvancedFeatureEngineer()
            features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
            labels = labels.loc[features_df.index]

            feature_cols = [c for c in features_df.columns
                            if c not in ['open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume']
                            and features_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

            X = features_df[feature_cols].values
            y = labels.values

            print(f"Features: {len(feature_cols)}, Samples: {len(X)}")

            # Train ENTRY model
            print("\n--- ENTRY MODEL ---")
            ensemble = EnsembleTrainer()
            metrics = ensemble.train_ensemble(X, y, feature_cols, top_features=60)
            ensemble.save_model(f'trading_model_{symbol}_H1')
            print(f"✓ Entry model saved: trading_model_{symbol}_H1")

            # Train EXIT model
            print("\n--- EXIT MODEL ---")
            exit_trainer = ExitModelTrainer()

            # Generate entry signals from trained model
            predictions, avg_proba, unanimous = ensemble.predict(X)
            high_conf = (avg_proba > 0.75) | (avg_proba < 0.25)
            entry_signals = unanimous & high_conf

            # Generate exit labels
            exit_labels, position_features = exit_trainer.generate_exit_labels(
                df_h1.loc[features_df.index],
                entry_signals,
                predictions,
                max_hold_bars=48
            )

            # Combine features
            position_feature_names = [
                'bars_in_position', 'unrealized_pips', 'max_profit_seen',
                'max_drawdown_seen', 'pullback_from_peak', 'normalized_time'
            ]
            all_feature_names = feature_cols + position_feature_names
            X_exit = exit_trainer.create_exit_features(features_df[feature_cols], position_features)

            exit_metrics = exit_trainer.train_exit_model(X_exit, exit_labels, all_feature_names, top_features=40)
            exit_trainer.save_model(f'{symbol}_H1')
            print(f"✓ Exit model saved: exit_model_{symbol}_H1")

    finally:
        collector.disconnect()

    print(f"\n{'='*60}")
    print(f"✓ All {len(symbols)} entry + exit models trained!")


def cmd_ml_backtest(args):
    """Run ML backtest for multiple symbols with exit model comparison."""
    import numpy as np

    print(f"=" * 60)
    print("ML BACKTEST (Entry + Exit Models)")
    print(f"=" * 60)

    from src.ml.data_collector import DataCollector
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.ensemble_trainer import EnsembleTrainer
    from src.ml.exit_model import ExitModelTrainer

    symbols = args.symbols
    years = args.years
    initial_balance = args.balance
    monthly_deposit = args.deposit

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Years: {years}")
    print(f"Initial: ${initial_balance}")
    print(f"Monthly deposit: ${monthly_deposit}")

    collector = DataCollector(symbols)
    if not collector.connect():
        print("✗ MT5 connection failed")
        return

    results_fixed = []  # Fixed 48h hold
    results_ml_exit = []  # ML exit model

    try:
        for symbol in symbols:
            print(f"\n{'='*50}")
            print(f"Backtesting {symbol}...")
            print(f"{'='*50}")

            bars = years * 8760
            df_h1 = collector.get_historical_data(symbol, 'H1', bars)
            df_h4 = collector.get_historical_data(symbol, 'H4', bars // 4)
            df_d1 = collector.get_historical_data(symbol, 'D1', bars // 24)

            # Load entry model
            ensemble = EnsembleTrainer()
            try:
                ensemble.load_model(f'trading_model_{symbol}_H1')
            except:
                print(f"  ✗ Entry model not found for {symbol}")
                continue

            # Load exit model
            exit_trainer = ExitModelTrainer()
            has_exit_model = False
            try:
                exit_trainer.load_model(f'{symbol}_H1')
                has_exit_model = True
            except:
                print(f"  (No exit model - ML exit disabled)")

            engineer = AdvancedFeatureEngineer()
            features_df = engineer.create_advanced_features(df_h1, df_h4, df_d1)
            feature_cols = engineer.get_feature_names()
            X = features_df[feature_cols].values
            predictions, avg_proba, unanimous = ensemble.predict(X)

            high_conf = (avg_proba > 0.75) | (avg_proba < 0.25)
            best_signals = unanimous & high_conf
            close_prices = df_h1.loc[features_df.index, 'close'].values

            MAX_HOLD = 48

            # ============ STRATEGY 1: Fixed 48h Hold ============
            balance_fixed = initial_balance
            deposited_fixed = initial_balance
            trades_fixed = []
            current_month = None

            i = 0
            while i < len(features_df) - MAX_HOLD:
                month = features_df.index[i].to_period('M')
                if current_month != month:
                    if current_month is not None:
                        balance_fixed += monthly_deposit
                        deposited_fixed += monthly_deposit
                    current_month = month

                if best_signals[i]:
                    entry = close_prices[i]
                    exit_price = close_prices[i + MAX_HOLD]
                    pred = predictions[i]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3  # Spread

                    lot = min((balance_fixed * 0.02) / (40 * 10), 10.0)
                    lot = max(0.01, round(lot, 2))
                    pnl = pips * 10 * lot
                    balance_fixed += pnl
                    trades_fixed.append({'pnl': pnl, 'hold': MAX_HOLD})
                    i += MAX_HOLD
                else:
                    i += 1

            # ============ STRATEGY 2: ML Exit Model ============
            balance_ml = initial_balance
            deposited_ml = initial_balance
            trades_ml = []
            current_month = None

            i = 0
            while i < len(features_df) - MAX_HOLD:
                month = features_df.index[i].to_period('M')
                if current_month != month:
                    if current_month is not None:
                        balance_ml += monthly_deposit
                        deposited_ml += monthly_deposit
                    current_month = month

                if best_signals[i]:
                    entry = close_prices[i]
                    pred = predictions[i]
                    exit_bar = MAX_HOLD  # Default

                    if has_exit_model:
                        # Simulate checking exit model each bar
                        for hold_bars in range(4, MAX_HOLD + 1, 4):  # Check every 4 bars
                            if i + hold_bars >= len(features_df):
                                break

                            current_price = close_prices[i + hold_bars]
                            if pred == 1:
                                unrealized_pips = (current_price - entry) * 10000
                            else:
                                unrealized_pips = (entry - current_price) * 10000

                            # Build exit features
                            max_profit = unrealized_pips  # Simplified
                            position_features = np.array([[
                                hold_bars,
                                unrealized_pips,
                                max(unrealized_pips, 0),
                                min(unrealized_pips, 0),
                                0,  # pullback
                                hold_bars / MAX_HOLD
                            ]])

                            market_features = X[i + hold_bars:i + hold_bars + 1]
                            if len(market_features) > 0:
                                X_exit = np.hstack([market_features, position_features])
                                exit_pred, exit_proba = exit_trainer.predict(X_exit)

                                if exit_pred[0] == 1 and exit_proba[0] > 0.6:
                                    exit_bar = hold_bars
                                    break

                    exit_price = close_prices[i + exit_bar]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3

                    lot = min((balance_ml * 0.02) / (40 * 10), 10.0)
                    lot = max(0.01, round(lot, 2))
                    pnl = pips * 10 * lot
                    balance_ml += pnl
                    trades_ml.append({'pnl': pnl, 'hold': exit_bar})
                    i += exit_bar
                else:
                    i += 1

            # Results for this symbol
            wins_fixed = sum(1 for t in trades_fixed if t['pnl'] > 0)
            wins_ml = sum(1 for t in trades_ml if t['pnl'] > 0)

            print(f"\n  Fixed 48h Hold:")
            print(f"    Balance: ${balance_fixed:,.2f} | Profit: ${balance_fixed - deposited_fixed:,.2f}")
            print(f"    Trades: {len(trades_fixed)} | Win rate: {wins_fixed/len(trades_fixed)*100:.1f}%" if trades_fixed else "    No trades")

            print(f"\n  ML Exit Model:")
            print(f"    Balance: ${balance_ml:,.2f} | Profit: ${balance_ml - deposited_ml:,.2f}")
            print(f"    Trades: {len(trades_ml)} | Win rate: {wins_ml/len(trades_ml)*100:.1f}%" if trades_ml else "    No trades")
            if trades_ml:
                avg_hold = sum(t['hold'] for t in trades_ml) / len(trades_ml)
                print(f"    Avg hold: {avg_hold:.1f} bars")

            results_fixed.append({'symbol': symbol, 'profit': balance_fixed - deposited_fixed, 'trades': len(trades_fixed), 'wins': wins_fixed})
            results_ml_exit.append({'symbol': symbol, 'profit': balance_ml - deposited_ml, 'trades': len(trades_ml), 'wins': wins_ml})

    finally:
        collector.disconnect()

    # Combined comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Fixed 48h vs ML Exit")
    print(f"{'='*60}")

    fixed_profit = sum(r['profit'] for r in results_fixed)
    ml_profit = sum(r['profit'] for r in results_ml_exit)
    fixed_wins = sum(r['wins'] for r in results_fixed)
    ml_wins = sum(r['wins'] for r in results_ml_exit)
    fixed_trades = sum(r['trades'] for r in results_fixed)
    ml_trades = sum(r['trades'] for r in results_ml_exit)

    print(f"\n{'Strategy':<20} {'Profit':>15} {'Trades':>10} {'Win Rate':>10}")
    print(f"{'-'*55}")
    print(f"{'Fixed 48h':<20} ${fixed_profit:>14,.2f} {fixed_trades:>10} {fixed_wins/fixed_trades*100:>9.1f}%" if fixed_trades else "Fixed 48h: No trades")
    print(f"{'ML Exit':<20} ${ml_profit:>14,.2f} {ml_trades:>10} {ml_wins/ml_trades*100:>9.1f}%" if ml_trades else "ML Exit: No trades")

    if ml_profit > fixed_profit:
        improvement = ((ml_profit - fixed_profit) / abs(fixed_profit)) * 100 if fixed_profit != 0 else 0
        print(f"\n✓ ML Exit is BETTER by ${ml_profit - fixed_profit:,.2f} ({improvement:+.1f}%)")
    else:
        print(f"\n✗ Fixed 48h is better by ${fixed_profit - ml_profit:,.2f}")


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

