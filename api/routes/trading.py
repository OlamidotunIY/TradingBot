"""
Trading Routes - Trade Execution and Position Management

API endpoints for trading operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

router = APIRouter()
logger = logging.getLogger('trading_bot')


# Request/Response Models
class TradeRequest(BaseModel):
    """Trade request model."""
    symbol: str = Field(..., example="EURUSD")
    type: str = Field(..., example="BUY", description="BUY or SELL")
    volume: float = Field(..., ge=0.01, le=100, example=0.1)
    price: Optional[float] = Field(None, description="Price for pending orders")
    sl: Optional[float] = Field(None, description="Stop loss price")
    tp: Optional[float] = Field(None, description="Take profit price")
    comment: Optional[str] = Field(None, max_length=100)


class TradeResponse(BaseModel):
    """Trade response model."""
    success: bool
    ticket: Optional[int] = None
    symbol: str
    type: str
    volume: float
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    message: str = ""


class PositionResponse(BaseModel):
    """Position response model."""
    ticket: int
    symbol: str
    type: str
    volume: float
    open_price: float
    current_price: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    pips: float
    open_time: str


class ClosePositionRequest(BaseModel):
    """Close position request."""
    volume: Optional[float] = Field(None, description="Partial close volume")


# Endpoints
@router.post("/trade", response_model=TradeResponse)
async def execute_trade(request: TradeRequest):
    """
    Execute a new trade.

    - **symbol**: Trading symbol (e.g., EURUSD)
    - **type**: Order type (BUY or SELL)
    - **volume**: Lot size (0.01 - 100)
    - **sl**: Optional stop loss price
    - **tp**: Optional take profit price
    """
    try:
        # Import here to avoid circular imports
        from ..app import app_state

        mt5 = app_state.get('mt5_handler')
        if mt5 is None:
            raise HTTPException(status_code=503, detail="MT5 not connected")

        result = mt5.place_order(
            symbol=request.symbol,
            order_type=request.type,
            volume=request.volume,
            price=request.price,
            sl=request.sl,
            tp=request.tp,
            comment=request.comment or ""
        )

        if result and result.get('success'):
            return TradeResponse(
                success=True,
                ticket=result.get('ticket'),
                symbol=request.symbol,
                type=request.type,
                volume=result.get('volume', request.volume),
                price=result.get('price', 0),
                sl=request.sl,
                tp=request.tp,
                message="Trade executed successfully"
            )
        else:
            return TradeResponse(
                success=False,
                symbol=request.symbol,
                type=request.type,
                volume=request.volume,
                price=0,
                message=result.get('comment', 'Trade execution failed') if result else 'Unknown error'
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(symbol: Optional[str] = None):
    """
    Get all open positions.

    - **symbol**: Optional filter by symbol
    """
    try:
        from ..app import app_state

        mt5 = app_state.get('mt5_handler')
        if mt5 is None:
            # Return empty list if not connected
            return []

        positions = mt5.get_positions(symbol)

        return [
            PositionResponse(
                ticket=pos['ticket'],
                symbol=pos['symbol'],
                type=pos['type'],
                volume=pos['volume'],
                open_price=pos['open_price'],
                current_price=pos['current_price'],
                sl=pos['sl'],
                tp=pos['tp'],
                profit=pos['profit'],
                pips=(pos['current_price'] - pos['open_price']) * 10000 if pos['type'] == 'BUY'
                     else (pos['open_price'] - pos['current_price']) * 10000,
                open_time=pos['open_time'].isoformat()
            )
            for pos in positions
        ]

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/position/{ticket}")
async def close_position(ticket: int, request: Optional[ClosePositionRequest] = None):
    """
    Close an open position.

    - **ticket**: Position ticket number
    - **volume**: Optional partial close volume
    """
    try:
        from ..app import app_state

        mt5 = app_state.get('mt5_handler')
        if mt5 is None:
            raise HTTPException(status_code=503, detail="MT5 not connected")

        result = mt5.close_position(ticket)

        if result and result.get('success'):
            return {
                "success": True,
                "ticket": ticket,
                "profit": result.get('profit', 0),
                "message": "Position closed successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to close position")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-all")
async def close_all_positions(symbol: Optional[str] = None):
    """
    Close all open positions.

    - **symbol**: Optional filter by symbol
    """
    try:
        from ..app import app_state

        mt5 = app_state.get('mt5_handler')
        if mt5 is None:
            raise HTTPException(status_code=503, detail="MT5 not connected")

        positions = mt5.get_positions(symbol)
        results = []

        for pos in positions:
            result = mt5.close_position(pos['ticket'])
            results.append({
                'ticket': pos['ticket'],
                'success': result.get('success', False) if result else False
            })

        closed = sum(1 for r in results if r['success'])

        return {
            "success": True,
            "closed": closed,
            "total": len(positions),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
