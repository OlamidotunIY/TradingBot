"""
Models - SQLAlchemy Database Models

This module defines database models for trades, positions, and logs.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Trade(Base):
    """Trade history model."""

    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer, unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    type = Column(String(10), nullable=False)  # BUY, SELL
    volume = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    sl = Column(Float)
    tp = Column(Float)
    profit = Column(Float, default=0.0)
    pips = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    magic = Column(Integer, default=0)
    strategy = Column(String(50))
    comment = Column(Text)
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, SL_HIT, TP_HIT
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'type': self.type,
            'volume': self.volume,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'sl': self.sl,
            'tp': self.tp,
            'profit': self.profit,
            'pips': self.pips,
            'commission': self.commission,
            'swap': self.swap,
            'magic': self.magic,
            'strategy': self.strategy,
            'comment': self.comment,
            'status': self.status,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None
        }


class Position(Base):
    """Open position snapshot model."""

    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    type = Column(String(10), nullable=False)
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    current_price = Column(Float)
    sl = Column(Float)
    tp = Column(Float)
    profit = Column(Float, default=0.0)
    magic = Column(Integer, default=0)
    strategy = Column(String(50))
    open_time = Column(DateTime, nullable=False)
    snapshot_time = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'type': self.type,
            'volume': self.volume,
            'open_price': self.open_price,
            'current_price': self.current_price,
            'sl': self.sl,
            'tp': self.tp,
            'profit': self.profit,
            'magic': self.magic,
            'strategy': self.strategy,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'snapshot_time': self.snapshot_time.isoformat() if self.snapshot_time else None
        }


class PerformanceLog(Base):
    """Daily performance log model."""

    __tablename__ = 'performance_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float)
    daily_profit = Column(Float, default=0.0)
    daily_return_pct = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    max_drawdown = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'starting_balance': self.starting_balance,
            'ending_balance': self.ending_balance,
            'daily_profit': self.daily_profit,
            'daily_return_pct': self.daily_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown
        }


class BacktestResult(Base):
    """Backtest result storage model."""

    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(50), unique=True, nullable=False)
    strategy_name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_balance = Column(Float)
    final_balance = Column(Float)
    total_profit = Column(Float)
    return_pct = Column(Float)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    metrics_json = Column(Text)  # Store full metrics as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'result_id': self.result_id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_profit': self.total_profit,
            'return_pct': self.return_pct,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio
        }
