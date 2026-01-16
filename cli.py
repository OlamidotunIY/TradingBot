"""
MT5 Trading Bot - CLI Interface

Command-line interface for the trading bot.
"""

import argparse
import sys
import uvicorn
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
        'backtest-pairs': cmd_backtest_pairs
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

