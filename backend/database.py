"""
Database configuration and session management for PostgreSQL.
"""
import os
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import logging

logger = logging.getLogger(__name__)

# Get database URL from environment variable
# Default to SQLite for tests/local development without Postgres
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./polymore_test.db"
)

# Create SQLAlchemy engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}  # Required for SQLite
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Enable connection health checks
        pool_size=10,
        max_overflow=20
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

def get_db():
    """
    Dependency function to get database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(max_retries: int = 5, retry_delay: int = 2):
    """
    Initialize database schema with retry logic.
    Creates all tables defined in models.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Initial delay between retries in seconds (exponential backoff)
    
    Raises:
        Exception: If database initialization fails after all retries
    """
    from sqlalchemy import text
    from models import task  # Import models to register them
    
    for attempt in range(max_retries):
        try:
            # Test connection first
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create tables if connection successful
            Base.metadata.create_all(bind=engine)
            logger.info("Database schema initialized successfully")
            return
            
        except OperationalError as e:
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
