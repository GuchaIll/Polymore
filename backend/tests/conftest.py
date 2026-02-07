import pytest
import sys
import os
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set DATABASE_URL to SQLite for tests BEFORE importing
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# Mock tblite module to avoid import errors
mock_tblite = MagicMock()
sys.modules["tblite"] = mock_tblite
sys.modules["tblite.interface"] = mock_tblite

from database import Base, engine
from main import app
from fastapi.testclient import TestClient


# Disable the startup event that tries to initialize the database
app.router.on_startup = []


@pytest.fixture(scope="function", autouse=True)
def test_database():
    """Create database tables for testing."""
    # Recreate tables for each test
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client():
    """Create a test client."""    
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
