"""
Backtest Routes - Backtesting Operations

API endpoints for running and retrieving backtests.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger('trading_bot')

# In-memory storage for backtest results
backtest_results: Dict[str, Any] = {}


# Request/Response Models
class BacktestRequest(BaseModel):
    """Backtest request model."""
    strategy: str = Field(..., example="sma_crossover")
    symbol: str = Field(..., example="EURUSD")
    timeframe: str = Field(default="H1", example="H1")
    start_date: Optional[str] = Field(None, example="2024-01-01")
    end_date: Optional[str] = Field(None, example="2024-12-31")
    initial_balance: float = Field(default=10000.0, ge=100)
    bars: int = Field(default=1000, ge=100, le=10000)


class BacktestSummary(BaseModel):
    """Backtest summary model."""
    id: str
    strategy: str
    symbol: str
    timeframe: str
    status: str
    total_profit: Optional[float] = None
    return_pct: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    created_at: str


class BacktestResult(BaseModel):
    """Full backtest result model."""
    id: str
    strategy: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    metrics: Dict[str, Any]


# Endpoints
@router.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest for a strategy.

    Returns immediately with a backtest ID. Use GET /backtest/{id} to check status.
    """
    import uuid

    backtest_id = str(uuid.uuid4())[:8]

    # Store initial status
    backtest_results[backtest_id] = {
        'id': backtest_id,
        'strategy': request.strategy,
        'symbol': request.symbol,
        'timeframe': request.timeframe,
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'result': None
    }

    # Add background task
    background_tasks.add_task(
        _run_backtest_task,
        backtest_id,
        request
    )

    return {
        'id': backtest_id,
        'status': 'pending',
        'message': 'Backtest started. Check status at /api/backtest/{id}'
    }


async def _run_backtest_task(backtest_id: str, request: BacktestRequest):
    """Background task to run backtest."""
    try:
        backtest_results[backtest_id]['status'] = 'running'

        # Import modules
        from src.backtest import Backtester, DataLoader
        from src.strategies import StrategyManager
        from src.utils import load_config

        # Load config and get strategy
        config = load_config()
        strategies_config = config.get('strategies_config', {})

        # Initialize strategy manager and get strategy
        manager = StrategyManager(strategies_config)
        manager.load_strategies()

        strategy = manager.get_strategy(request.strategy)
        if strategy is None:
            backtest_results[backtest_id]['status'] = 'failed'
            backtest_results[backtest_id]['error'] = f"Strategy '{request.strategy}' not found"
            return

        # Load data
        data_loader = DataLoader()
        data = data_loader.generate_sample_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            bars=request.bars
        )

        # Override strategy symbol for testing
        strategy.symbols = [request.symbol]

        # Run backtest
        backtester = Backtester(initial_balance=request.initial_balance)
        result = backtester.run(strategy, data, request.symbol)

        # Store result
        backtest_results[backtest_id]['status'] = 'completed'
        backtest_results[backtest_id]['result'] = backtester.to_dict(result)

        logger.info(f"Backtest {backtest_id} completed: {result.total_trades} trades")

    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}")
        backtest_results[backtest_id]['status'] = 'failed'
        backtest_results[backtest_id]['error'] = str(e)


@router.get("/backtest", response_model=List[BacktestSummary])
async def list_backtests():
    """
    List all backtests.
    """
    summaries = []

    for bt_id, bt in backtest_results.items():
        result = bt.get('result', {})

        summaries.append(BacktestSummary(
            id=bt_id,
            strategy=bt['strategy'],
            symbol=bt['symbol'],
            timeframe=bt['timeframe'],
            status=bt['status'],
            total_profit=result.get('total_profit') if result else None,
            return_pct=result.get('return_pct') if result else None,
            total_trades=result.get('total_trades') if result else None,
            win_rate=result.get('win_rate') if result else None,
            created_at=bt['created_at']
        ))

    return summaries


@router.get("/backtest/{backtest_id}")
async def get_backtest(backtest_id: str):
    """
    Get backtest status and results.
    """
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    bt = backtest_results[backtest_id]

    return {
        'id': bt['id'],
        'strategy': bt['strategy'],
        'symbol': bt['symbol'],
        'timeframe': bt['timeframe'],
        'status': bt['status'],
        'created_at': bt['created_at'],
        'error': bt.get('error'),
        'result': bt.get('result')
    }


@router.get("/backtest/{backtest_id}/results", response_model=BacktestResult)
async def get_backtest_results(backtest_id: str):
    """
    Get full backtest results.
    """
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    bt = backtest_results[backtest_id]

    if bt['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Backtest status: {bt['status']}")

    result = bt['result']

    return BacktestResult(
        id=backtest_id,
        strategy=bt['strategy'],
        symbol=bt['symbol'],
        timeframe=bt['timeframe'],
        start_date=result['start_date'],
        end_date=result['end_date'],
        initial_balance=result['initial_balance'],
        final_balance=result['final_balance'],
        total_profit=result['total_profit'],
        return_pct=result['return_pct'],
        total_trades=result['total_trades'],
        winning_trades=result['winning_trades'],
        losing_trades=result['losing_trades'],
        win_rate=result['win_rate'],
        metrics=result['metrics']
    )


@router.delete("/backtest/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """
    Delete a backtest result.
    """
    if backtest_id not in backtest_results:
        raise HTTPException(status_code=404, detail="Backtest not found")

    del backtest_results[backtest_id]

    return {"success": True, "message": "Backtest deleted"}
