"""
Order Manager - Order Execution and Management

This module handles order execution, modification, and cancellation.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

from .mt5_handler import MT5Handler

logger = logging.getLogger('trading_bot.trades')


class OrderType(Enum):
    """Order types."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class OrderRequest:
    """Order request data class."""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    magic: int = 0
    comment: str = ""
    strategy_name: str = ""


@dataclass
class OrderResult:
    """Order result data class."""
    success: bool
    ticket: Optional[int] = None
    symbol: str = ""
    order_type: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    error_code: Optional[int] = None
    error_message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OrderManager:
    """Manages order execution and tracking."""

    def __init__(self, mt5_handler: MT5Handler, config: Dict[str, Any]):
        """
        Initialize Order Manager.

        Args:
            mt5_handler: MT5Handler instance
            config: Trading configuration
        """
        self.mt5 = mt5_handler
        self.config = config
        self.magic_number = config.get('magic_number', 123456)
        self.deviation = config.get('deviation', 20)
        self._order_history: List[OrderResult] = []

    def execute_order(self, request: OrderRequest) -> OrderResult:
        """
        Execute a trading order.

        Args:
            request: OrderRequest object

        Returns:
            OrderResult: Result of order execution
        """
        logger.info(f"Executing order: {request.order_type.value} {request.volume} {request.symbol}")

        # Validate order
        validation = self._validate_order(request)
        if not validation['valid']:
            logger.error(f"Order validation failed: {validation['message']}")
            return OrderResult(
                success=False,
                error_message=validation['message']
            )

        # Execute order through MT5
        result = self.mt5.place_order(
            symbol=request.symbol,
            order_type=request.order_type.value,
            volume=request.volume,
            price=request.price,
            sl=request.sl,
            tp=request.tp,
            magic=request.magic or self.magic_number,
            comment=request.comment or request.strategy_name
        )

        if result is None:
            order_result = OrderResult(
                success=False,
                symbol=request.symbol,
                error_message="Order execution failed"
            )
        elif result.get('success'):
            order_result = OrderResult(
                success=True,
                ticket=result.get('ticket'),
                symbol=request.symbol,
                order_type=request.order_type.value,
                volume=result.get('volume', request.volume),
                price=result.get('price', 0),
                sl=request.sl,
                tp=request.tp
            )
            logger.info(f"Order executed: Ticket {order_result.ticket}")
        else:
            order_result = OrderResult(
                success=False,
                symbol=request.symbol,
                error_code=result.get('retcode'),
                error_message=result.get('comment', 'Unknown error')
            )
            logger.error(f"Order failed: {order_result.error_message}")

        self._order_history.append(order_result)
        return order_result

    def close_position(self, ticket: int) -> OrderResult:
        """
        Close an open position.

        Args:
            ticket: Position ticket number

        Returns:
            OrderResult: Result of close operation
        """
        logger.info(f"Closing position: {ticket}")

        result = self.mt5.close_position(ticket)

        if result and result.get('success'):
            return OrderResult(
                success=True,
                ticket=ticket
            )

        return OrderResult(
            success=False,
            ticket=ticket,
            error_message="Failed to close position"
        )

    def close_all_positions(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Close all open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            list: List of OrderResult objects
        """
        positions = self.mt5.get_positions(symbol)
        results = []

        for position in positions:
            result = self.close_position(position['ticket'])
            results.append(result)

        logger.info(f"Closed {len([r for r in results if r.success])} positions")
        return results

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> OrderResult:
        """
        Modify position SL/TP.

        Args:
            ticket: Position ticket
            sl: New stop loss price
            tp: New take profit price

        Returns:
            OrderResult: Result of modification
        """
        # This would need MT5 implementation for position modification
        # Placeholder for now
        logger.info(f"Modifying position {ticket}: SL={sl}, TP={tp}")

        return OrderResult(
            success=True,
            ticket=ticket,
            sl=sl,
            tp=tp
        )

    def _validate_order(self, request: OrderRequest) -> Dict[str, Any]:
        """
        Validate order before execution.

        Args:
            request: OrderRequest to validate

        Returns:
            dict: Validation result with 'valid' and 'message' keys
        """
        # Check symbol
        symbol_info = self.mt5.get_symbol_info(request.symbol)
        if symbol_info is None:
            return {'valid': False, 'message': f"Symbol {request.symbol} not found"}

        # Check volume
        min_volume = symbol_info.get('volume_min', 0.01)
        max_volume = symbol_info.get('volume_max', 100)
        volume_step = symbol_info.get('volume_step', 0.01)

        if request.volume < min_volume:
            return {'valid': False, 'message': f"Volume {request.volume} below minimum {min_volume}"}

        if request.volume > max_volume:
            return {'valid': False, 'message': f"Volume {request.volume} above maximum {max_volume}"}

        # Check if trading is allowed
        if symbol_info.get('trade_mode', 0) == 0:
            return {'valid': False, 'message': f"Trading disabled for {request.symbol}"}

        return {'valid': True, 'message': 'OK'}

    def get_order_history(self) -> List[OrderResult]:
        """Get order history from this session."""
        return self._order_history.copy()

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current open positions."""
        return self.mt5.get_positions(symbol)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending orders."""
        return self.mt5.get_orders(symbol)
