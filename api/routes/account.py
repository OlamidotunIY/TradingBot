"""
Account Routes - Account Information and Statistics

API endpoints for account info, stats, and trade history.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

router = APIRouter()
logger = logging.getLogger('trading_bot')


# Response Models
class AccountInfo(BaseModel):
    """Account information model."""
    login: int
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: int
    profit: float
    currency: str


class TradingStats(BaseModel):
    """Trading statistics model."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_pips: float
    avg_profit: float
    avg_pips: float
    best_trade: float
    worst_trade: float
    profit_factor: float
    period: str


class TradeHistoryItem(BaseModel):
    """Trade history item model."""
    ticket: int
    symbol: str
    type: str
    volume: float
    entry_price: float
    exit_price: float
    profit: float
    pips: float
    entry_time: str
    exit_time: str
    strategy: Optional[str] = None


# Endpoints
@router.get("/account", response_model=AccountInfo)
async def get_account_info():
    """
    Get account information.
    """
    try:
        from ..app import app_state

        mt5 = app_state.get('mt5_handler')
        if mt5 is None:
            raise HTTPException(status_code=503, detail="MT5 not connected")

        info = mt5.get_account_info()
        if info is None:
            raise HTTPException(status_code=500, detail="Failed to get account info")

        return AccountInfo(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=TradingStats)
async def get_trading_stats(
    period: str = Query(default="all", regex="^(today|week|month|all)$")
):
    """
    Get trading statistics.

    - **period**: Time period (today, week, month, all)
    """
    try:
        from ..app import app_state

        db = app_state.get('database')
        if db is None:
            # Return empty stats if no database
            return TradingStats(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_profit=0,
                total_pips=0,
                avg_profit=0,
                avg_pips=0,
                best_trade=0,
                worst_trade=0,
                profit_factor=0,
                period=period
            )

        # Calculate date filter
        now = datetime.utcnow()
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'week':
            start_date = now - timedelta(days=7)
        elif period == 'month':
            start_date = now - timedelta(days=30)
        else:
            start_date = None

        # Query trades from database
        from src.data.models import Trade

        with db.session_scope() as session:
            query = session.query(Trade).filter(Trade.status != 'OPEN')

            if start_date:
                query = query.filter(Trade.exit_time >= start_date)

            trades = query.all()

        if not trades:
            return TradingStats(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_profit=0,
                total_pips=0,
                avg_profit=0,
                avg_pips=0,
                best_trade=0,
                worst_trade=0,
                profit_factor=0,
                period=period
            )

        # Calculate stats
        profits = [t.profit for t in trades]
        pips = [t.pips for t in trades if t.pips]

        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p < 0]

        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1

        return TradingStats(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=(len(winning) / len(trades) * 100) if trades else 0,
            total_profit=sum(profits),
            total_pips=sum(pips) if pips else 0,
            avg_profit=sum(profits) / len(trades) if trades else 0,
            avg_pips=sum(pips) / len(pips) if pips else 0,
            best_trade=max(profits) if profits else 0,
            worst_trade=min(profits) if profits else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            period=period
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[TradeHistoryItem])
async def get_trade_history(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    """
    Get trade history.

    - **symbol**: Filter by symbol
    - **strategy**: Filter by strategy
    - **limit**: Maximum records to return
    - **offset**: Records to skip
    """
    try:
        from ..app import app_state

        db = app_state.get('database')
        if db is None:
            return []

        from src.data.models import Trade

        with db.session_scope() as session:
            query = session.query(Trade).filter(Trade.status != 'OPEN')

            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if strategy:
                query = query.filter(Trade.strategy == strategy)

            query = query.order_by(Trade.exit_time.desc())
            query = query.offset(offset).limit(limit)

            trades = query.all()

        return [
            TradeHistoryItem(
                ticket=t.ticket,
                symbol=t.symbol,
                type=t.type,
                volume=t.volume,
                entry_price=t.entry_price,
                exit_price=t.exit_price or 0,
                profit=t.profit,
                pips=t.pips or 0,
                entry_time=t.entry_time.isoformat() if t.entry_time else "",
                exit_time=t.exit_time.isoformat() if t.exit_time else "",
                strategy=t.strategy
            )
            for t in trades
        ]

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk")
async def get_risk_settings():
    """
    Get current risk management settings.
    """
    try:
        from ..app import app_state

        risk_manager = app_state.get('risk_manager')
        if risk_manager is None:
            # Return default settings
            return {
                'max_risk_per_trade': 0.02,
                'max_daily_drawdown': 0.05,
                'max_open_positions': 5,
                'max_positions_per_symbol': 2,
                'default_sl_pips': 50,
                'default_tp_pips': 100
            }

        return risk_manager.get_risk_settings()

    except Exception as e:
        logger.error(f"Error getting risk settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-stats")
async def get_daily_stats():
    """
    Get daily trading statistics.
    """
    try:
        from ..app import app_state

        risk_manager = app_state.get('risk_manager')
        if risk_manager is None:
            return {
                'date': datetime.utcnow().date().isoformat(),
                'starting_balance': 0,
                'daily_pnl': 0,
                'max_daily_drawdown': 0.05,
                'max_risk_per_trade': 0.02
            }

        return risk_manager.get_daily_stats()

    except Exception as e:
        logger.error(f"Error getting daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
