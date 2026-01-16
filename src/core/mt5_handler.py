"""
MT5 Handler - MetaTrader 5 Connection and Operations

This module handles all interactions with the MetaTrader 5 terminal.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger('trading_bot')


class MT5Handler:
    """Handles connection and communication with MetaTrader 5."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MT5 Handler.

        Args:
            config: MT5 configuration dictionary containing login, password, server, path
        """
        self.config = config
        self.connected = False
        self._login = config.get('login')
        self._password = config.get('password')
        self._server = config.get('server')
        self._path = config.get('path')
        self._timeout = config.get('timeout', 60000)

    def connect(self) -> bool:
        """
        Initialize and connect to MT5 terminal.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize(
                path=self._path,
                login=int(self._login) if self._login else None,
                password=self._password,
                server=self._server,
                timeout=self._timeout
            ):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False

            self.connected = True
            account_info = mt5.account_info()
            logger.info(f"Connected to MT5: {account_info.server}, Account: {account_info.login}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information.

        Returns:
            dict: Account information or None if not connected
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'leverage': info.leverage,
            'profit': info.profit,
            'currency': info.currency
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            dict: Symbol information or None if not found
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found")
            return None

        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'spread': info.spread,
            'digits': info.digits,
            'point': info.point,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step,
            'trade_mode': info.trade_mode
        }

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
        from_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV (candlestick) data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (M1, M5, M15, H1, H4, D1, etc.)
            count: Number of candles to retrieve
            from_date: Start date for historical data

        Returns:
            DataFrame: OHLCV data or None if failed
        """
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        mt5_tf = tf_map.get(timeframe.upper())
        if mt5_tf is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        # Enable symbol if not visible
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol: {symbol}")
            return None

        if from_date:
            rates = mt5.copy_rates_from(symbol, mt5_tf, from_date, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No data received for {symbol}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest tick for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            dict: Tick data or None if failed
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time)
        }

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        magic: int = 0,
        comment: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol
            order_type: 'BUY' or 'SELL'
            volume: Lot size
            price: Price for pending orders (None for market orders)
            sl: Stop loss price
            tp: Take profit price
            magic: Magic number for order identification
            comment: Order comment

        Returns:
            dict: Order result or None if failed
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None

        # Determine order type
        if order_type.upper() == 'BUY':
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask if price is None else price
        elif order_type.upper() == 'SELL':
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid if price is None else price
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "deviation": self.config.get('deviation', 20),
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return {
                'success': False,
                'retcode': result.retcode,
                'comment': result.comment
            }

        logger.info(f"Order placed: {order_type} {volume} {symbol} @ {price}")

        return {
            'success': True,
            'ticket': result.order,
            'volume': result.volume,
            'price': result.price,
            'symbol': symbol,
            'type': order_type
        }

    def close_position(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Close a position by ticket number.

        Args:
            ticket: Position ticket number

        Returns:
            dict: Close result or None if failed
        """
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found")
            return None

        position = position[0]
        symbol = position.symbol
        volume = position.volume

        # Determine close type
        if position.type == mt5.ORDER_TYPE_BUY:
            trade_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            trade_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}")
            return None

        logger.info(f"Position {ticket} closed")
        return {
            'success': True,
            'ticket': ticket,
            'profit': position.profit
        }

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            list: List of open positions
        """
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'volume': pos.volume,
            'open_price': pos.price_open,
            'current_price': pos.price_current,
            'sl': pos.sl,
            'tp': pos.tp,
            'profit': pos.profit,
            'magic': pos.magic,
            'comment': pos.comment,
            'open_time': datetime.fromtimestamp(pos.time)
        } for pos in positions]

    def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pending orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            list: List of pending orders
        """
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        return [{
            'ticket': order.ticket,
            'symbol': order.symbol,
            'type': order.type,
            'volume': order.volume_current,
            'price': order.price_open,
            'sl': order.sl,
            'tp': order.tp,
            'magic': order.magic,
            'comment': order.comment
        } for order in orders]

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
