"""
Database - SQLite Database Handler

This module handles database connections and operations.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Optional, Generator
from pathlib import Path
import logging

from .models import Base

logger = logging.getLogger('trading_bot')


class Database:
    """SQLite database handler."""

    def __init__(self, db_path: str = 'data/trading_bot.db'):
        """
        Initialize Database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        self._initialized = False

    def init_db(self) -> None:
        """Initialize database tables."""
        Base.metadata.create_all(bind=self.engine)
        self._initialized = True
        logger.info(f"Database initialized: {self.db_path}")

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            Session: SQLAlchemy session
        """
        if not self._initialized:
            self.init_db()

        return self.SessionLocal()

    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session_scope() as session:
                session.query(...)

        Yields:
            Session: Database session
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.init_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
