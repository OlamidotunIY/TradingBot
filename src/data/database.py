"""
MongoDB Database Handler

Handles connections and operations for MongoDB Atlas.
"""
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import os

logger = logging.getLogger('trading_bot')


class Database:
    """MongoDB database handler."""

    def __init__(self, connection_string: str = None, db_name: str = 'trading_bot'):
        """
        Initialize MongoDB connection.

        Args:
            connection_string: MongoDB Atlas connection string
            db_name: Database name
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI',
            'mongodb+srv://dotun:Iyanda1999.@cluster0.ybkkvnq.mongodb.net'
        )
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[MongoDatabase] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            self._connected = True
            logger.info(f"Connected to MongoDB: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False

    def init_db(self) -> None:
        """Initialize database (connect if needed)."""
        if not self._connected:
            self.connect()

    # ==================== TRADES ====================

    def save_trade(self, trade: Dict[str, Any]) -> str:
        """
        Save a trade to the database.

        Args:
            trade: Trade data dictionary

        Returns:
            Inserted document ID
        """
        trade['created_at'] = datetime.utcnow()
        trade['updated_at'] = datetime.utcnow()
        result = self.db.trades.insert_one(trade)
        logger.info(f"Trade saved: {result.inserted_id}")
        return str(result.inserted_id)

    def update_trade(self, ticket: int, updates: Dict[str, Any]) -> bool:
        """Update a trade by ticket number."""
        updates['updated_at'] = datetime.utcnow()
        result = self.db.trades.update_one(
            {'ticket': ticket},
            {'$set': updates}
        )
        return result.modified_count > 0

    def get_trades(self, mode: str = None, limit: int = 100) -> List[Dict]:
        """Get trades, optionally filtered by mode."""
        query = {}
        if mode:
            query['mode'] = mode
        return list(self.db.trades.find(query).sort('created_at', -1).limit(limit))

    def get_trade_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Get a trade by ticket number."""
        return self.db.trades.find_one({'ticket': ticket})

    # ==================== SIGNALS ====================

    def save_signal(self, signal: Dict[str, Any]) -> str:
        """Save a trading signal."""
        signal['created_at'] = datetime.utcnow()
        result = self.db.signals.insert_one(signal)
        return str(result.inserted_id)

    def get_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get signals, optionally filtered by symbol."""
        query = {}
        if symbol:
            query['symbol'] = symbol
        return list(self.db.signals.find(query).sort('created_at', -1).limit(limit))

    # ==================== PERFORMANCE ====================

    def save_performance_log(self, log: Dict[str, Any]) -> str:
        """Save daily performance log."""
        log['created_at'] = datetime.utcnow()
        result = self.db.performance_logs.insert_one(log)
        return str(result.inserted_id)

    def get_performance_logs(self, days: int = 30) -> List[Dict]:
        """Get recent performance logs."""
        return list(self.db.performance_logs.find().sort('date', -1).limit(days))

    # ==================== UTILITY ====================

    def close(self) -> None:
        """Close database connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
