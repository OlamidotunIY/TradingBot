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
    trade_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'EURUSD'],
                              help='Trading symbols (space-separated)')
    trade_parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')

    # Train ML model command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'EURUSD'],
                              help='Trading symbols (space-separated)')
    train_parser.add_argument('--years', '-y', type=int, default=5, help='Years of data')
    train_parser.add_argument('--lookahead', type=int, default=48, help='Lookahead bars')
    train_parser.add_argument('--threshold', type=int, default=40, help='Pip threshold')

    # ML Backtest command
    ml_backtest_parser = subparsers.add_parser('ml-backtest', help='Backtest ML model')
    ml_backtest_parser.add_argument('--symbols', '-s', nargs='+', default=['GBPUSD', 'EURUSD'],
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
    print("LIVE TRADING MODE (ML Strategy):")
    print(f"{'='*60}")
    print(f"  ✓ ML Entry Model (conf > {MIN_CONFIDENCE})")
    print(f"  ✓ ML Exit Model (conf > {EXIT_CONFIDENCE})")
    print(f"  ✓ Dynamic Position Sizing (Regime-based)")
    print(f"  ✓ Correlation Manager")
    print(f"  ✓ Max Hold Time (48h)")
    print(f"  - Session Filter (DISABLED)")
    print(f"  - Trailing Stops (DISABLED - relying on ML Exit)")

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

                # === FILTER 1: Session Filter (DISABLED for ML Strategy) ===
                # can_trade_session, session_reason = session_filter.should_trade(symbol)
                # if not can_trade_session:
                #     print(f"  {symbol}: ⏰ {session_reason}")
                #     continue

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
                            # trailing_mgr.remove_position(pos.ticket) # Disabled
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

                    # === FILTER 3: FRESH TREND DETECTION ===
                    # Get the latest feature values for freshness checks
                    latest_features = features_df.iloc[-1]

                    # Filter 3.1: Check trend age (avoid exhausted trends)
                    if 'trend_bars_age' in latest_features.index:
                        trend_age = latest_features['trend_bars_age']
                        if trend_age > 100:  # Trend too old
                            print(f"  {symbol}: {signal_type} signal SKIPPED (trend age: {trend_age:.0f} bars - exhausted)")
                            continue

                    # Filter 3.2: Require retracement for better entry
                    if signal_type == 'BUY':
                        # For BUY: prefer pullback in uptrend or bounce from support
                        has_retracement = False
                        if 'pullback_in_uptrend' in latest_features.index and latest_features['pullback_in_uptrend'] == 1:
                            has_retracement = True
                        elif 'in_buy_retracement' in latest_features.index and latest_features['in_buy_retracement'] == 1:
                            has_retracement = True
                        elif 'bouncing_from_low' in latest_features.index and latest_features['bouncing_from_low'] == 1:
                            has_retracement = True

                        if not has_retracement:
                            print(f"  {symbol}: BUY signal SKIPPED (no retracement - price extended)")
                            continue

                    elif signal_type == 'SELL':
                        # For SELL: prefer bounce in downtrend or rejection from resistance
                        has_retracement = False
                        if 'pullback_in_downtrend' in latest_features.index and latest_features['pullback_in_downtrend'] == 1:
                            has_retracement = True
                        elif 'in_sell_retracement' in latest_features.index and latest_features['in_sell_retracement'] == 1:
                            has_retracement = True
                        elif 'rejecting_from_high' in latest_features.index and latest_features['rejecting_from_high'] == 1:
                            has_retracement = True

                        if not has_retracement:
                            print(f"  {symbol}: SELL signal SKIPPED (no retracement - price extended)")
                            continue

                    # Filter 3.3: Check for momentum shift (fresh moves preferred)
                    has_momentum_shift = False
                    if signal_type == 'BUY':
                        if 'momentum_shift_bullish' in latest_features.index and latest_features['momentum_shift_bullish'] == 1:
                            has_momentum_shift = True
                        elif 'macd_turning_bullish' in latest_features.index and latest_features['macd_turning_bullish'] == 1:
                            has_momentum_shift = True
                        elif 'rsi_turning_bullish' in latest_features.index and latest_features['rsi_turning_bullish'] == 1:
                            has_momentum_shift = True
                    else:  # SELL
                        if 'momentum_shift_bearish' in latest_features.index and latest_features['momentum_shift_bearish'] == 1:
                            has_momentum_shift = True
                        elif 'macd_turning_bearish' in latest_features.index and latest_features['macd_turning_bearish'] == 1:
                            has_momentum_shift = True
                        elif 'rsi_turning_bearish' in latest_features.index and latest_features['rsi_turning_bearish'] == 1:
                            has_momentum_shift = True

                    # Momentum shift is a bonus, not required, but we log it
                    momentum_label = "✨ Fresh momentum" if has_momentum_shift else "⚠️ Continuation"

                    # Filter 3.4: Avoid price too extended from MA
                    if 'is_extended_from_ma' in latest_features.index and latest_features['is_extended_from_ma'] == 1:
                        print(f"  {symbol}: {signal_type} signal SKIPPED (price >2% extended from SMA50)")
                        continue

                    # === FILTER 4: Regime Detection (simple heuristic) ===
                    atr_pips = calculate_atr_pips(df_h1)
                    volatility = df_h1['close'].pct_change().rolling(20).std().iloc[-1]
                    is_high_vol = volatility > 0.015

                    if is_high_vol:
                        print(f"  {symbol}: ⚡ High volatility - reducing size")

                    print(f"  🔔 {symbol}: {signal_type} | Confidence: {confidence:.2%} | ATR: {atr_pips:.1f} pips | {momentum_label}")

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
                        # trailing_mgr.add_position(
                        #     ticket=result.order,
                        #     symbol=symbol,
                        #     direction=signal_type,
                        #     entry_price=price,
                        #     atr_pips=atr_pips
                        # )

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
    """Run ML backtest comparing: Fixed 48h vs ML Exit vs Full Advanced."""
    import numpy as np

    print(f"=" * 60)
    print("ML BACKTEST (3-Way Comparison)")
    print(f"=" * 60)

    from src.ml.data_collector import DataCollector
    from src.ml.advanced_features import AdvancedFeatureEngineer
    from src.ml.ensemble_trainer import EnsembleTrainer
    from src.ml.exit_model import ExitModelTrainer
    from src.filters.session_filter import SessionFilter
    from src.risk.dynamic_sizing import DynamicPositionSizer
    from src.risk.trailing_stop import calculate_atr_pips

    import MetaTrader5 as mt5

    if not mt5.initialize():
        print(f"✗ MT5 init failed: {mt5.last_error()}")
        return

    symbols = args.symbols
    years = args.years

    # Get actual demo balance
    account_info = mt5.account_info()
    initial_balance = account_info.balance if account_info else args.balance

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Years: {years}")
    print(f"Account Balance: ${initial_balance:,.2f}")

    print(f"\nStrategies being compared:")
    print(f"  1. Fixed 48h Hold (baseline)")
    print(f"  2. ML Exit Model")
    print(f"  3. ADVANCED (Session + Dynamic Size + ATR Trailing)")

    collector = DataCollector(symbols)
    collector.connected = True

    results_fixed = []
    results_ml_exit = []
    results_advanced = []
    results_hybrid = []

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
            monthly_profits_fixed = {}
            monthly_profits_ml = {}
            monthly_profits_adv = {}
            monthly_profits_hybrid = {}

            # Shared components for all strategies
            session_filter = SessionFilter()
            position_sizer = DynamicPositionSizer(base_risk_pct=0.02, max_lot_size=10.0, max_leverage=20.0)

            # ============ STRATEGY 1: Fixed 48h Hold ============
            balance_fixed = initial_balance
            deposited_fixed = initial_balance
            trades_fixed = []

            i = 0
            current_month = None
            while i < len(features_df) - MAX_HOLD:
                dt = features_df.index[i].to_pydatetime()
                month_key = dt.strftime('%Y-%m')
                if current_month != month_key:
                    current_month = month_key

                # MARGIN CALL CHECK
                if balance_fixed < (initial_balance * 0.3):
                    # Account blown (below 30% margin)
                    break

                if month_key not in monthly_profits_fixed:
                    monthly_profits_fixed[month_key] = 0

                if best_signals[i]:
                    entry = close_prices[i]
                    exit_price = close_prices[i + MAX_HOLD]
                    pred = predictions[i]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3

                    # Use dynamic ATR for sizing
                    atr_pips_fixed = (df_h1.iloc[max(0, i-20):i+1]['high'] - df_h1.iloc[max(0, i-20):i+1]['low']).mean() * 10000

                    sizing = position_sizer.calculate_position_size(
                        balance=balance_fixed, atr_pips=atr_pips_fixed,
                        confidence=avg_proba[i] if pred == 1 else 1-avg_proba[i],
                        regime='trending_up' if pred == 1 else 'trending_down'
                    )
                    lot = sizing['lot_size']
                    pnl = pips * 10 * lot
                    balance_fixed += pnl
                    monthly_profits_fixed[month_key] += pnl
                    trades_fixed.append({'pnl': pnl, 'pips': pips, 'hold': MAX_HOLD})
                    i += MAX_HOLD
                else:
                    i += 1

            # ============ STRATEGY 2: ML Exit Model ============
            balance_ml = initial_balance
            deposited_ml = initial_balance
            trades_ml = []

            i = 0
            current_month = None
            while i < len(features_df) - MAX_HOLD:
                dt = features_df.index[i].to_pydatetime()
                month_key = dt.strftime('%Y-%m')
                if current_month != month_key:
                    current_month = month_key

                # MARGIN CALL CHECK
                if balance_ml < (initial_balance * 0.3):
                    break

                if month_key not in monthly_profits_ml:
                    monthly_profits_ml[month_key] = 0

                if best_signals[i]:
                    entry = close_prices[i]
                    pred = predictions[i]
                    exit_bar = MAX_HOLD

                    if has_exit_model:
                        for hold_bars in range(4, MAX_HOLD + 1, 4):
                            if i + hold_bars >= len(features_df): break
                            current_price = close_prices[i + hold_bars]
                            unrealized = (current_price - entry) * 10000 if pred == 1 else (entry - current_price) * 10000
                            pos_features = np.array([[hold_bars, unrealized, max(unrealized, 0), min(unrealized, 0), 0, hold_bars / MAX_HOLD]])
                            market_feats = X[i + hold_bars:i + hold_bars + 1]
                            if len(market_feats) > 0:
                                X_exit = np.hstack([market_feats, pos_features])
                                exit_pred, exit_proba = exit_trainer.predict(X_exit)
                                if exit_pred[0] == 1 and exit_proba[0] > 0.6:
                                    exit_bar = hold_bars
                                    break

                    exit_price = close_prices[i + exit_bar]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3

                    # Use dynamic ATR for sizing
                    atr_pips_ml = (df_h1.iloc[max(0, i-20):i+1]['high'] - df_h1.iloc[max(0, i-20):i+1]['low']).mean() * 10000

                    sizing = position_sizer.calculate_position_size(
                        balance=balance_ml, atr_pips=atr_pips_ml,
                        confidence=avg_proba[i] if pred == 1 else 1-avg_proba[i],
                        regime='trending_up' if pred == 1 else 'trending_down'
                    )
                    lot = sizing['lot_size']
                    pnl = pips * 10 * lot
                    balance_ml += pnl
                    monthly_profits_ml[month_key] += pnl
                    trades_ml.append({'pnl': pnl, 'pips': pips, 'hold': exit_bar, 'lot': lot})
                    i += exit_bar
                else:
                    i += 1

            # ============ STRATEGY 3: ADVANCED ============
            balance_adv = initial_balance
            deposited_adv = initial_balance
            trades_adv = []

            i = 0
            current_month = None
            while i < len(features_df) - MAX_HOLD:
                dt = features_df.index[i].to_pydatetime()
                month_key = dt.strftime('%Y-%m')
                if current_month != month_key:
                    current_month = month_key

                # MARGIN CALL CHECK
                if balance_adv < (initial_balance * 0.3):
                    break

                if month_key not in monthly_profits_adv:
                    monthly_profits_adv[month_key] = 0

                if best_signals[i]:
                    can_trade, _ = session_filter.should_trade(symbol, dt)
                    if not can_trade:
                        i += 1
                        continue

                    entry = close_prices[i]
                    pred = predictions[i]
                    confidence = avg_proba[i] if pred == 1 else 1 - avg_proba[i]
                    atr_pips = (df_h1.iloc[max(0, i-20):i+1]['high'] - df_h1.iloc[max(0, i-20):i+1]['low']).mean() * 10000

                    sizing = position_sizer.calculate_position_size(
                        balance=balance_adv, atr_pips=atr_pips, confidence=confidence,
                        regime='trending_up' if pred == 1 else 'trending_down'
                    )
                    lot_size = sizing['lot_size']

                    stop_pips = atr_pips * 2
                    target_pips = atr_pips * 3
                    exit_bar = MAX_HOLD
                    max_profit = 0

                    for hold_bars in range(1, MAX_HOLD + 1):
                        if i + hold_bars >= len(features_df): break
                        current_price = close_prices[i + hold_bars]
                        pips = (current_price - entry) * 10000 if pred == 1 else (entry - current_price) * 10000
                        max_profit = max(max_profit, pips)
                        if max_profit - pips >= stop_pips or pips >= target_pips:
                            exit_bar = hold_bars
                            break

                    exit_price = close_prices[i + exit_bar]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3
                    pnl = pips * 10 * lot_size
                    balance_adv += pnl
                    monthly_profits_adv[month_key] += pnl
                    trades_adv.append({'pnl': pnl, 'pips': pips, 'hold': exit_bar, 'lot': lot_size})
                    i += exit_bar
                else:
                    i += 1

            # ============ STRATEGY 4: ULTIMATE HYBRID ============
            balance_hybrid = initial_balance
            deposited_hybrid = initial_balance
            trades_hybrid = []

            i = 0
            current_month = None
            while i < len(features_df) - MAX_HOLD:
                if current_month != month_key:
                    current_month = month_key

                # MARGIN CALL CHECK
                if balance_hybrid < (initial_balance * 0.3):
                    break

                if month_key not in monthly_profits_hybrid:
                    monthly_profits_hybrid[month_key] = 0

                # ENTRY GATE: Unanimous + High Conf + Session
                if best_signals[i]:
                    can_trade, _ = session_filter.should_trade(symbol, dt)
                    if not can_trade:
                        i += 1
                        continue

                    entry = close_prices[i]
                    pred = predictions[i]
                    confidence = avg_proba[i] if pred == 1 else 1 - avg_proba[i]
                    atr_pips_h = (df_h1.iloc[max(0, i-20):i+1]['high'] - df_h1.iloc[max(0, i-20):i+1]['low']).mean() * 10000

                    # SIZING: Dynamic + Confidence Multiplier
                    sizing = position_sizer.calculate_position_size(
                        balance=balance_hybrid, atr_pips=atr_pips_h, confidence=confidence,
                        regime='trending_up' if pred == 1 else 'trending_down'
                    )
                    lot_size_h = sizing['lot_size']

                    # MULTI-LAYER EXIT
                    stop_pips_h = atr_pips_h * 2
                    exit_bar = MAX_HOLD
                    max_profit = 0

                    for hold_bars in range(1, MAX_HOLD + 1):
                        if i + hold_bars >= len(features_df): break
                        current_price = close_prices[i + hold_bars]
                        pips_raw = (current_price - entry) * 10000 if pred == 1 else (entry - current_price) * 10000
                        max_profit = max(max_profit, pips_raw)

                        # Layer 1: ML Exit (Higher speed check every 4h)
                        if has_exit_model and hold_bars % 4 == 0:
                            pos_features = np.array([[hold_bars, pips_raw, max(pips_raw, 0), min(pips_raw, 0), 0, hold_bars / MAX_HOLD]])
                            market_feats = X[i + hold_bars:i + hold_bars + 1]
                            if len(market_feats) > 0:
                                X_exit = np.hstack([market_feats, pos_features])
                                exit_pred, exit_proba = exit_trainer.predict(X_exit)
                                if exit_pred[0] == 1 and exit_proba[0] > 0.65:
                                    exit_bar = hold_bars
                                    break

                        # Layer 2: ATR Trailing Stop
                        if max_profit - pips_raw >= stop_pips_h:
                            exit_bar = hold_bars
                            break

                    exit_price = close_prices[i + exit_bar]
                    pips = (exit_price - entry) * 10000 if pred == 1 else (entry - exit_price) * 10000
                    pips -= 3
                    pnl = pips * 10 * lot_size_h
                    balance_hybrid += pnl
                    monthly_profits_hybrid[month_key] += pnl
                    trades_hybrid.append({'pnl': pnl, 'pips': pips, 'hold': exit_bar, 'lot': lot_size_h})
                    i += exit_bar
                else:
                    i += 1

            # Analysis for Win Rate Explanation
            def analyze_trades(trades, balance, total_deposited):
                if not trades: return None
                wins = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]

                total_profit = balance - total_deposited
                wr = len(wins) / len(trades) if trades else 0
                avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
                avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

                avg_win_pips = sum(t['pips'] for t in wins) / len(wins) if wins else 0
                avg_loss_pips = sum(t['pips'] for t in losses) / len(losses) if losses else 0

                # Leverage analysis
                avg_lot = sum(t.get('lot', 0) for t in trades) / len(trades) if trades else 0
                # Assuming 1 lot = 100k notional
                avg_leverage = (avg_lot * 100000) / (total_deposited + total_profit/2) # Rough avg balance
                max_lot = max(t.get('lot', 0) for t in trades) if trades else 0

                pf = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0
                rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

                return {
                    'wr': wr, 'avg_win': avg_win, 'avg_loss': avg_loss,
                    'avg_win_pips': avg_win_pips, 'avg_loss_pips': avg_loss_pips,
                    'rr': rr, 'pf': pf, 'total_profit': total_profit,
                    'count': len(trades), 'wins_count': len(wins), 'loss_count': len(losses),
                    'avg_leverage': avg_leverage, 'max_lot': max_lot
                }

            stats_fixed = analyze_trades(trades_fixed, balance_fixed, deposited_fixed)
            stats_ml = analyze_trades(trades_ml, balance_ml, deposited_ml)
            stats_advanced = analyze_trades(trades_adv, balance_adv, deposited_adv)
            stats_hybrid = analyze_trades(trades_hybrid, balance_hybrid, deposited_hybrid)

            print(f"\n{'='*75}")
            print(f"TRADING STATS: {symbol}")
            print(f"{'='*75}")
            print(f"{'Metric':<20} {'Fixed 48h':>12} {'ML Exit':>12} {'Advanced':>12} {'Hybrid':>12}")
            print("-" * 75)
            if stats_fixed and stats_ml and stats_advanced and stats_hybrid:
                print(f"{'Total Trades':<20} {stats_fixed['count']:>12} {stats_ml['count']:>12} {stats_advanced['count']:>12} {stats_hybrid['count']:>12}")
                print(f"{'Win Rate':<20} {stats_fixed['wr']*100:>11.1f}% {stats_ml['wr']*100:>11.1f}% {stats_advanced['wr']*100:>11.1f}% {stats_hybrid['wr']*100:>11.1f}%")
                print(f"{'Profit Factor':<20} {stats_fixed['pf']:>12.2f} {stats_ml['pf']:>12.2f} {stats_advanced['pf']:>12.2f} {stats_hybrid['pf']:>12.2f}")
                print(f"{'Avg Win ($)':<20} ${stats_fixed['avg_win']:>11.0f} ${stats_ml['avg_win']:>11.0f} ${stats_advanced['avg_win']:>11.0f} ${stats_hybrid['avg_win']:>11.0f}")
                print(f"{'Avg Loss ($)':<20} ${stats_fixed['avg_loss']:>11.0f} ${stats_ml['avg_loss']:>11.0f} ${stats_advanced['avg_loss']:>11.0f} ${stats_hybrid['avg_loss']:>11.0f}")
                print(f"{'Avg Win (Pips)':<20} {stats_fixed['avg_win_pips']:>12.1f} {stats_ml['avg_win_pips']:>12.1f} {stats_advanced['avg_win_pips']:>12.1f} {stats_hybrid['avg_win_pips']:>12.1f}")
                print(f"{'Avg Loss (Pips)':<20} {stats_fixed['avg_loss_pips']:>12.1f} {stats_ml['avg_loss_pips']:>12.1f} {stats_advanced['avg_loss_pips']:>12.1f} {stats_hybrid['avg_loss_pips']:>12.1f}")
                print(f"{'Reward:Risk':<20} {stats_fixed['rr']:>12.2f} {stats_ml['rr']:>12.2f} {stats_advanced['rr']:>12.2f} {stats_hybrid['rr']:>12.2f}")
                print(f"{'Avg Leverage':<20} {stats_fixed['avg_leverage']:>11.1f}x {stats_ml['avg_leverage']:>11.1f}x {stats_advanced['avg_leverage']:>11.1f}x {stats_hybrid['avg_leverage']:>11.1f}x")
                print(f"{'Max Lot Size':<20} {stats_fixed['max_lot']:>12.2f} {stats_ml['max_lot']:>12.2f} {stats_advanced['max_lot']:>12.2f} {stats_hybrid['max_lot']:>12.2f}")

            print(f"\nMONTHLY PROFIT DETAILS")
            print(f"{'Month':<10} {'Fixed':>10} {'ML Ext':>10} {'Adv':>10} {'Hybrid':>10}")
            print("-" * 60)
            all_months = sorted(list(set(monthly_profits_fixed.keys()) | set(monthly_profits_ml.keys()) | set(monthly_profits_adv.keys()) | set(monthly_profits_hybrid.keys())))
            for m in all_months[-12:]: # Last 12 months
                f_p = monthly_profits_fixed.get(m, 0)
                m_p = monthly_profits_ml.get(m, 0)
                a_p = monthly_profits_adv.get(m, 0)
                h_p = monthly_profits_hybrid.get(m, 0)

                h_pct = (h_p / initial_balance) * 100
                h_icon = "✓" if h_p > 0 else "✗"
                print(f"{m:<10} ${f_p:>7.0f}  ${m_p:>7.0f}  ${a_p:>7.0f}  ${h_p:>7.0f} ({h_pct:>3.1f}%) {h_icon}")

            print(f"\n{'='*60}")
            print(f"ACCOUNT SUMMARY: {symbol}")
            print(f"{'='*60}")
            print(f"  {'Strategy':<15} {'Depository':>12} {'Final Bal':>12} {'Net Profit':>12}")
            print(f"  {'Fixed 48h':<15} ${initial_balance:>11,.0f} ${balance_fixed:>11,.0f} ${balance_fixed - deposited_fixed:>11,.0f}")
            print(f"  {'ML Exit':<15} ${initial_balance:>11,.0f} ${balance_ml:>11,.0f} ${balance_ml - deposited_ml:>11,.0f}")
            print(f"  {'Advanced':<15} ${initial_balance:>11,.0f} ${balance_adv:>11,.0f} ${balance_adv - deposited_adv:>11,.0f}")
            print(f"  {'Hybrid':<15} ${initial_balance:>11,.0f} ${balance_hybrid:>11,.0f} ${balance_hybrid - deposited_hybrid:>11,.0f}")

            results_fixed.append({'symbol': symbol, 'profit': balance_fixed - deposited_fixed, 'trades': len(trades_fixed), 'wins': sum(1 for t in trades_fixed if t['pnl'] > 0)})
            results_ml_exit.append({'symbol': symbol, 'profit': balance_ml - deposited_ml, 'trades': len(trades_ml), 'wins': sum(1 for t in trades_ml if t['pnl'] > 0)})
            results_advanced.append({'symbol': symbol, 'profit': balance_adv - deposited_adv, 'trades': len(trades_adv), 'wins': sum(1 for t in trades_adv if t['pnl'] > 0)})
            results_hybrid.append({'symbol': symbol, 'profit': balance_hybrid - deposited_hybrid, 'trades': len(trades_hybrid), 'wins': sum(1 for t in trades_hybrid if t['pnl'] > 0)})

    finally:
        collector.disconnect()

    # Combined comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON: All 3 Strategies")
    print(f"{'='*60}")

    fixed_profit = sum(r['profit'] for r in results_fixed)
    ml_profit = sum(r['profit'] for r in results_ml_exit)
    adv_profit = sum(r['profit'] for r in results_advanced)
    hybrid_profit = sum(r['profit'] for r in results_hybrid)
    fixed_trades = sum(r['trades'] for r in results_fixed)
    ml_trades = sum(r['trades'] for r in results_ml_exit)
    adv_trades = sum(r['trades'] for r in results_advanced)
    hybrid_trades = sum(r['trades'] for r in results_hybrid)
    fixed_wins = sum(r['wins'] for r in results_fixed)
    ml_wins = sum(r['wins'] for r in results_ml_exit)
    adv_wins = sum(r['wins'] for r in results_advanced)
    hybrid_wins = sum(r['wins'] for r in results_hybrid)

    print(f"\n{'Strategy':<20} {'Profit':>15} {'Trades':>10} {'Win Rate':>10}")
    print(f"{'-'*55}")
    print(f"{'1. Fixed 48h':<20} ${fixed_profit:>14,.2f} {fixed_trades:>10} {fixed_wins/fixed_trades*100:>9.1f}%" if fixed_trades else "Fixed 48h: No trades")
    print(f"{'2. ML Exit':<20} ${ml_profit:>14,.2f} {ml_trades:>10} {ml_wins/ml_trades*100:>9.1f}%" if ml_trades else "ML Exit: No trades")
    print(f"{'3. ADVANCED':<20} ${adv_profit:>14,.2f} {adv_trades:>10} {adv_wins/adv_trades*100:>9.1f}%" if adv_trades else "Advanced: No trades")
    print(f"{'4. HYBRID':<20} ${hybrid_profit:>14,.2f} {hybrid_trades:>10} {hybrid_wins/hybrid_trades*100:>9.1f}%" if hybrid_trades else "Hybrid: No trades")

    profits = [('Fixed 48h', fixed_profit), ('ML Exit', ml_profit), ('Advanced', adv_profit), ('Hybrid', hybrid_profit)]
    best_name, best_profit = max(profits, key=lambda x: x[1])
    print(f"\n🏆 BEST STRATEGY: {best_name} with ${best_profit:,.2f} profit")

    if best_name == 'Advanced':
        improvement = ((adv_profit - fixed_profit) / abs(fixed_profit)) * 100 if fixed_profit != 0 else 0
        print(f"   Advanced beats Fixed 48h by {improvement:+.1f}%")


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

