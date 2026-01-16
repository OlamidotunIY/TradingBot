"""
Strategy Routes - Strategy Management

API endpoints for managing trading strategies.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger('trading_bot')


# Response Models
class StrategyInfo(BaseModel):
    """Strategy information model."""
    name: str
    enabled: bool
    symbols: List[str]
    timeframe: str
    parameters: Dict[str, Any]
    last_analysis: Optional[str] = None


class StrategyListResponse(BaseModel):
    """Strategy list response."""
    total: int
    enabled: int
    strategies: List[StrategyInfo]


class ToggleResponse(BaseModel):
    """Toggle response."""
    name: str
    enabled: bool
    message: str


# Endpoints
@router.get("/strategies", response_model=StrategyListResponse)
async def list_strategies():
    """
    Get all available strategies.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            return StrategyListResponse(total=0, enabled=0, strategies=[])

        status = manager.get_status()
        strategies = []

        for name, info in status.get('strategies', {}).items():
            strategies.append(StrategyInfo(
                name=info['name'],
                enabled=info['enabled'],
                symbols=info['symbols'],
                timeframe=info['timeframe'],
                parameters=info['parameters'],
                last_analysis=info.get('last_analysis')
            ))

        return StrategyListResponse(
            total=status['total_strategies'],
            enabled=status['enabled_strategies'],
            strategies=strategies
        )

    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{name}", response_model=StrategyInfo)
async def get_strategy(name: str):
    """
    Get a specific strategy by name.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            raise HTTPException(status_code=404, detail="Strategy manager not initialized")

        strategy = manager.get_strategy(name)
        if strategy is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

        status = strategy.get_status()

        return StrategyInfo(
            name=status['name'],
            enabled=status['enabled'],
            symbols=status['symbols'],
            timeframe=status['timeframe'],
            parameters=status['parameters'],
            last_analysis=status.get('last_analysis')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{name}/toggle", response_model=ToggleResponse)
async def toggle_strategy(name: str):
    """
    Toggle a strategy's enabled state.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")

        new_state = manager.toggle_strategy(name)

        if new_state is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

        return ToggleResponse(
            name=name,
            enabled=new_state,
            message=f"Strategy {'enabled' if new_state else 'disabled'}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{name}/enable", response_model=ToggleResponse)
async def enable_strategy(name: str):
    """
    Enable a strategy.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")

        success = manager.enable_strategy(name)

        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

        return ToggleResponse(
            name=name,
            enabled=True,
            message="Strategy enabled"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/{name}/disable", response_model=ToggleResponse)
async def disable_strategy(name: str):
    """
    Disable a strategy.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")

        success = manager.disable_strategy(name)

        if not success:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

        return ToggleResponse(
            name=name,
            enabled=False,
            message="Strategy disabled"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{name}/status")
async def get_strategy_status(name: str):
    """
    Get detailed status of a strategy.
    """
    try:
        from ..app import app_state

        manager = app_state.get('strategy_manager')
        if manager is None:
            raise HTTPException(status_code=503, detail="Strategy manager not initialized")

        strategy = manager.get_strategy(name)
        if strategy is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

        status = strategy.get_status()

        # Add last signals if available
        last_signals = {}
        for symbol in strategy.symbols:
            signal = strategy.get_last_signal(symbol)
            if signal:
                last_signals[symbol] = {
                    'type': signal.signal_type.value,
                    'reason': signal.reason,
                    'timestamp': signal.timestamp.isoformat()
                }

        return {
            **status,
            'last_signals': last_signals
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
