"""
Performance Metrics - Backtest Performance Calculations

This module calculates various performance metrics for backtesting.
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger('trading_bot')


class PerformanceMetrics:
    """Calculates trading performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize Performance Metrics.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate(
        self,
        trades: List,
        initial_balance: float,
        equity_curve: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            trades: List of BacktestTrade objects
            initial_balance: Starting balance
            equity_curve: Equity values over time

        Returns:
            dict: Performance metrics
        """
        if not trades:
            return self._empty_metrics()

        # Extract profits
        profits = [t.profit for t in trades]
        pips = [t.pips for t in trades]

        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        # Basic metrics
        total_profit = sum(profits)
        total_pips = sum(pips)
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Average metrics
        avg_profit = np.mean(profits) if profits else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0

        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Risk/reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

        # Equity curve metrics
        equity_array = np.array(equity_curve)

        # Maximum drawdown
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_array, initial_balance)

        # Sharpe ratio
        sharpe = self._calculate_sharpe(equity_array)

        # Sortino ratio
        sortino = self._calculate_sortino(equity_array)

        # Calmar ratio
        calmar = self._calculate_calmar(total_profit, initial_balance, max_drawdown_pct)

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive(profits)

        return {
            'total_profit': round(total_profit, 2),
            'total_pips': round(total_pips, 1),
            'return_pct': round((total_profit / initial_balance) * 100, 2),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'risk_reward': round(risk_reward, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
            'expectancy': round(expectancy, 2),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2)
        }

    def _calculate_max_drawdown(self, equity: np.ndarray, initial_balance: float) -> tuple:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = (max_drawdown / initial_balance) * 100

        return max_drawdown, max_drawdown_pct

    def _calculate_sharpe(self, equity: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(equity) < 2:
            return 0

        returns = np.diff(equity) / equity[:-1]

        if np.std(returns) == 0:
            return 0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)

        return sharpe

    def _calculate_sortino(self, equity: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(equity) < 2:
            return 0

        returns = np.diff(equity) / equity[:-1]
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0

        sortino = np.sqrt(periods_per_year) * np.mean(returns) / np.std(downside_returns)

        return sortino

    def _calculate_calmar(self, total_profit: float, initial_balance: float, max_drawdown_pct: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown_pct == 0:
            return 0

        annual_return = (total_profit / initial_balance) * 100
        calmar = annual_return / max_drawdown_pct

        return calmar

    def _calculate_consecutive(self, profits: List[float]) -> tuple:
        """Calculate maximum consecutive wins and losses."""
        if not profits:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dict."""
        return {
            'total_profit': 0,
            'total_pips': 0,
            'return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'risk_reward': 0,
            'avg_profit': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'expectancy': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'gross_profit': 0,
            'gross_loss': 0
        }

    def format_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a readable report."""
        return f"""
╔══════════════════════════════════════════════════════╗
║              BACKTEST PERFORMANCE REPORT              ║
╠══════════════════════════════════════════════════════╣
║  Total Profit:     ${metrics['total_profit']:>12,.2f}  ({metrics['return_pct']:>6.2f}%)   ║
║  Total Pips:       {metrics['total_pips']:>12,.1f}                      ║
╠══════════════════════════════════════════════════════╣
║  Total Trades:     {metrics['total_trades']:>12}                      ║
║  Winning Trades:   {metrics['winning_trades']:>12}                      ║
║  Losing Trades:    {metrics['losing_trades']:>12}                      ║
║  Win Rate:         {metrics['win_rate']:>12.2f}%                     ║
╠══════════════════════════════════════════════════════╣
║  Profit Factor:    {metrics['profit_factor']:>12.2f}                      ║
║  Risk/Reward:      {metrics['risk_reward']:>12.2f}                      ║
║  Expectancy:       ${metrics['expectancy']:>12.2f}                    ║
╠══════════════════════════════════════════════════════╣
║  Max Drawdown:     ${metrics['max_drawdown']:>12,.2f}  ({metrics['max_drawdown_pct']:>6.2f}%)   ║
║  Sharpe Ratio:     {metrics['sharpe_ratio']:>12.2f}                      ║
║  Sortino Ratio:    {metrics['sortino_ratio']:>12.2f}                      ║
║  Calmar Ratio:     {metrics['calmar_ratio']:>12.2f}                      ║
╠══════════════════════════════════════════════════════╣
║  Max Consecutive Wins:   {metrics['max_consecutive_wins']:>6}                    ║
║  Max Consecutive Losses: {metrics['max_consecutive_losses']:>6}                    ║
╚══════════════════════════════════════════════════════╝
"""
