"""
Backtester - Strategy Backtesting Engine

This module simulates strategy execution on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid

from ..strategies.base_strategy import BaseStrategy, Signal, SignalType
from .metrics import PerformanceMetrics

logger = logging.getLogger('trading_bot')


@dataclass
class BacktestTrade:
    """Simulated trade data."""
    id: str
    symbol: str
    type: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    volume: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit: float = 0.0
    pips: float = 0.0
    status: str = 'OPEN'  # OPEN, CLOSED, SL_HIT, TP_HIT

    def close(self, exit_price: float, exit_time: datetime, reason: str = 'CLOSED') -> None:
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = reason

        # Calculate profit/pips
        if self.type == 'BUY':
            self.pips = (exit_price - self.entry_price) * 10000
        else:
            self.pips = (self.entry_price - exit_price) * 10000

        # Simplified profit calculation (10$ per pip per lot)
        self.profit = self.pips * 10 * self.volume


@dataclass
class BacktestResult:
    """Backtest result data."""
    id: str
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[BacktestTrade] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)

    @property
    def total_profit(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def return_pct(self) -> float:
        return (self.total_profit / self.initial_balance) * 100

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100


class Backtester:
    """Backtesting engine for strategies."""

    def __init__(self, initial_balance: float = 10000.0, risk_config: Dict[str, Any] = None, verbose: bool = False):
        """
        Initialize Backtester.

        Args:
            initial_balance: Starting balance for simulation
            risk_config: Risk management configuration
            verbose: If True, log every trade in real-time
        """
        self.initial_balance = initial_balance
        self._results: Dict[str, BacktestResult] = {}
        self.verbose = verbose

        # Default risk config
        default_risk = {
            'max_risk_per_trade': 0.02,      # 2% risk per trade
            'max_daily_drawdown': 0.05,      # 5% max daily drawdown
            'max_open_positions': 5,
            'max_positions_per_symbol': 2,
            'default_sl_pips': 50,
            'default_tp_pips': 100
        }
        self.risk_config = risk_config or default_risk

        logger.info(f"Backtester initialized with balance: ${initial_balance}")

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
        commission: float = 0.0,
        spread: float = 0.0002
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance to test
            data: Historical OHLCV data
            symbol: Trading symbol
            commission: Commission per trade
            spread: Spread in price units

        Returns:
            BacktestResult: Backtest results
        """
        result_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting backtest: {strategy.name} on {symbol}")

        # Initialize state
        balance = self.initial_balance
        equity = balance
        equity_curve = [balance]
        trades: List[BacktestTrade] = []
        open_trades: List[BacktestTrade] = []

        # Daily tracking for drawdown
        daily_starting_balance = balance
        current_day = None
        open_trades: List[BacktestTrade] = []

        # Ensure data is sorted
        data = data.sort_index()

        # Minimum bars needed for analysis
        min_bars = 50

        # Iterate through data
        for i in range(min_bars, len(data)):
            current_bar = data.iloc[i]
            historical_data = data.iloc[:i+1]
            current_time = data.index[i]
            current_price = current_bar['close']

            # Track new trading day for drawdown reset
            trade_date = current_time.date() if hasattr(current_time, 'date') else None
            if trade_date and trade_date != current_day:
                current_day = trade_date
                daily_starting_balance = equity

            # Calculate current equity
            unrealized_pnl = sum(self._calculate_unrealized_pnl(t, current_price) for t in open_trades)
            equity = balance + unrealized_pnl

            # Check daily drawdown limit
            daily_drawdown = (daily_starting_balance - equity) / daily_starting_balance if daily_starting_balance > 0 else 0
            max_daily_drawdown = self.risk_config.get('max_daily_drawdown', 0.05)
            hit_daily_limit = daily_drawdown >= max_daily_drawdown

            # Check open trades for SL/TP hits
            for trade in open_trades[:]:
                hit_sl, hit_tp = self._check_sl_tp(trade, current_bar)

                if hit_sl:
                    trade.close(trade.sl, current_time, 'SL_HIT')
                    balance += trade.profit - commission
                    open_trades.remove(trade)
                    if self.verbose:
                        status = "✗ LOSS" if trade.profit < 0 else "✓ WIN"
                        print(f"  CLOSE {trade.type} #{trade.id} @ {trade.sl:.5f} (SL) | P&L: ${trade.profit:,.2f} {status}")
                elif hit_tp:
                    trade.close(trade.tp, current_time, 'TP_HIT')
                    balance += trade.profit - commission
                    open_trades.remove(trade)
                    if self.verbose:
                        status = "✓ WIN" if trade.profit > 0 else "✗ LOSS"
                        print(f"  CLOSE {trade.type} #{trade.id} @ {trade.tp:.5f} (TP) | P&L: ${trade.profit:,.2f} {status}")

            # Get strategy signal
            signal = strategy.get_signal(historical_data, symbol)

            if signal and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # Risk checks
                max_positions = self.risk_config.get('max_open_positions', 5)

                # Skip if hit daily drawdown limit
                if hit_daily_limit:
                    continue

                # Skip if max positions reached
                if len(open_trades) >= max_positions:
                    continue

                # Close opposite trades
                for trade in open_trades[:]:
                    if (signal.signal_type == SignalType.BUY and trade.type == 'SELL') or \
                       (signal.signal_type == SignalType.SELL and trade.type == 'BUY'):
                        trade.close(current_price, current_time)
                        balance += trade.profit - commission
                        open_trades.remove(trade)
                        if self.verbose:
                            status = "✓ WIN" if trade.profit > 0 else "✗ LOSS"
                            print(f"  CLOSE {trade.type} #{trade.id} @ {current_price:.5f} (Signal) | P&L: ${trade.profit:,.2f} {status}")

                # Open new trade
                entry_price = current_price + spread if signal.signal_type == SignalType.BUY else current_price - spread

                # Calculate position size based on risk from config
                risk_pct = self.risk_config.get('max_risk_per_trade', 0.02)
                risk_amount = balance * risk_pct
                default_sl_pips = self.risk_config.get('default_sl_pips', 50)
                default_tp_pips = self.risk_config.get('default_tp_pips', 100)
                min_rr_ratio = 2.0  # Minimum 2:1 risk-reward ratio

                # Calculate SL and TP
                if signal.sl:
                    sl = signal.sl
                    sl_distance = abs(entry_price - sl)
                else:
                    # Default SL
                    sl_distance = default_sl_pips * 0.0001  # pips to price
                    if signal.signal_type == SignalType.BUY:
                        sl = entry_price - sl_distance
                    else:
                        sl = entry_price + sl_distance

                # Enforce minimum 2:1 risk-reward on TP
                if signal.tp:
                    tp = signal.tp
                    tp_distance = abs(tp - entry_price)
                    # Check if RR is at least 2:1, if not, adjust TP
                    if tp_distance < sl_distance * min_rr_ratio:
                        tp_distance = sl_distance * min_rr_ratio
                        if signal.signal_type == SignalType.BUY:
                            tp = entry_price + tp_distance
                        else:
                            tp = entry_price - tp_distance
                else:
                    # Default TP with at least 2:1 RR
                    tp_distance = max(default_tp_pips * 0.0001, sl_distance * min_rr_ratio)
                    if signal.signal_type == SignalType.BUY:
                        tp = entry_price + tp_distance
                    else:
                        tp = entry_price - tp_distance

                # Calculate lot size based on stop loss distance
                pips_at_risk = sl_distance * 10000  # convert to pips
                pip_value = 10.0  # $10 per pip per lot for majors
                if pips_at_risk > 0:
                    lot_size = risk_amount / (pips_at_risk * pip_value)
                else:
                    lot_size = risk_amount / (default_sl_pips * pip_value)

                # Clamp lot size to reasonable range
                lot_size = max(0.1, min(lot_size, 10.0))

                new_trade = BacktestTrade(
                    id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    type=signal.signal_type.value,
                    entry_price=entry_price,
                    entry_time=current_time,
                    volume=signal.volume or lot_size,
                    sl=sl,
                    tp=tp
                )

                trades.append(new_trade)
                open_trades.append(new_trade)

                if self.verbose:
                    sl_str = f"SL: {signal.sl:.5f}" if signal.sl else "No SL"
                    tp_str = f"TP: {signal.tp:.5f}" if signal.tp else "No TP"
                    print(f"[{current_time}] OPEN {new_trade.type} #{new_trade.id} @ {entry_price:.5f} | {lot_size:.2f} lots | {sl_str} | {tp_str}")

            # Update equity curve
            unrealized_pnl = sum(self._calculate_unrealized_pnl(t, current_price) for t in open_trades)
            equity_curve.append(balance + unrealized_pnl)

        # Close remaining open trades
        final_bar = data.iloc[-1]
        final_time = data.index[-1]
        final_price = final_bar['close']

        for trade in open_trades:
            trade.close(final_price, final_time, 'END')
            balance += trade.profit - commission

        # Calculate results
        winning_trades = [t for t in trades if t.profit > 0]
        losing_trades = [t for t in trades if t.profit <= 0]

        # Calculate metrics
        metrics_calculator = PerformanceMetrics()
        metrics = metrics_calculator.calculate(trades, self.initial_balance, equity_curve)

        result = BacktestResult(
            id=result_id,
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=strategy.timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_balance=self.initial_balance,
            final_balance=balance,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve
        )

        self._results[result_id] = result
        self._last_trades = trades  # Store for detailed analysis

        logger.info(f"Backtest complete: {len(trades)} trades, ${result.total_profit:.2f} profit ({result.return_pct:.2f}%)")

        return result

    def _check_sl_tp(self, trade: BacktestTrade, bar: pd.Series) -> tuple:
        """Check if SL or TP was hit."""
        hit_sl = False
        hit_tp = False

        if trade.sl:
            if trade.type == 'BUY' and bar['low'] <= trade.sl:
                hit_sl = True
            elif trade.type == 'SELL' and bar['high'] >= trade.sl:
                hit_sl = True

        if trade.tp:
            if trade.type == 'BUY' and bar['high'] >= trade.tp:
                hit_tp = True
            elif trade.type == 'SELL' and bar['low'] <= trade.tp:
                hit_tp = True

        return hit_sl, hit_tp

    def _calculate_unrealized_pnl(self, trade: BacktestTrade, current_price: float) -> float:
        """Calculate unrealized P&L for an open trade."""
        if trade.type == 'BUY':
            pips = (current_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - current_price) * 10000

        return pips * 10 * trade.volume

    def get_result(self, result_id: str) -> Optional[BacktestResult]:
        """Get a backtest result by ID."""
        return self._results.get(result_id)

    def get_all_results(self) -> List[BacktestResult]:
        """Get all backtest results."""
        return list(self._results.values())

    def to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'id': result.id,
            'strategy': result.strategy_name,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'initial_balance': result.initial_balance,
            'final_balance': result.final_balance,
            'total_profit': result.total_profit,
            'return_pct': result.return_pct,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'metrics': result.metrics
        }
