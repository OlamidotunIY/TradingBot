"""
Pydantic Models for Trading Bot

Defines data models for trades, signals, and performance logs.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode."""
    DEMO = "demo"
    REAL = "real"


class SignalType(str, Enum):
    """Signal type."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeStatus(str, Enum):
    """Trade status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    SL_HIT = "SL_HIT"
    TP_HIT = "TP_HIT"


class Signal(BaseModel):
    """Trading signal model."""
    symbol: str
    signal_type: SignalType
    confidence: float = Field(ge=0, le=1)
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: Optional[str] = None
    strategy: Optional[str] = None
    mode: TradingMode = TradingMode.DEMO
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'price': self.price,
            'sl': self.sl,
            'tp': self.tp,
            'reason': self.reason,
            'strategy': self.strategy,
            'mode': self.mode.value,
            'created_at': self.created_at
        }


class Trade(BaseModel):
    """Trade model."""
    ticket: int
    symbol: str
    type: str  # BUY or SELL
    volume: float
    entry_price: float
    exit_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    profit: float = 0.0
    pips: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    magic: int = 0
    strategy: Optional[str] = None
    comment: Optional[str] = None
    status: TradeStatus = TradeStatus.OPEN
    mode: TradingMode = TradingMode.DEMO
    entry_time: datetime
    exit_time: Optional[datetime] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return {
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
            'status': self.status.value,
            'mode': self.mode.value,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'confidence': self.confidence
        }


class PerformanceLog(BaseModel):
    """Daily performance log."""
    date: datetime
    starting_balance: float
    ending_balance: Optional[float] = None
    daily_profit: float = 0.0
    daily_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    mode: TradingMode = TradingMode.DEMO

    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'starting_balance': self.starting_balance,
            'ending_balance': self.ending_balance,
            'daily_profit': self.daily_profit,
            'daily_return_pct': self.daily_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown,
            'mode': self.mode.value
        }
