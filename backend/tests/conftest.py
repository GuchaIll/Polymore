"""
Pytest configuration for backend tests.
Sets up test environment to avoid database initialization issues.
"""
import os
import sys
from unittest.mock import MagicMock
import pytest

# Set environment variables before importing the app
os.environ["DB_INIT_ON_STARTUP"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./test_polymore.db"

# Mock tblite module for tests (tier-2 analysis dependency)
sys.modules['tblite'] = MagicMock()
sys.modules['tblite.interface'] = MagicMock()
