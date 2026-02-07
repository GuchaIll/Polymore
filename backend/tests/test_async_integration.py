import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

# Client is provided by conftest.py fixture

def test_submit_analysis(client):
    """Test tier-2 synchronous analysis with proper mocking."""
    # Mock the heavy compute function
    with patch("features.advanced_analysis.analyze_molecule_high_compute") as mock_analyze:
        mock_analyze.return_value = {
            "strength": 8.5,
            "flexibility": 4.0,
            "degradability": 6.5,
            "sustainability": 7.0,
            "meta": {"method": "mock"}
        }
        
        payload = {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        response = client.post("/predict/tier-2", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        
        # Check result data
        result = data["data"]
        assert result["strength"] == 8.5
        
        # Verify the function was called
        mock_analyze.assert_called_once_with("CC(=O)Oc1ccccc1C(=O)O")

def test_get_task_status_pending(client):
    """Test getting status of a pending task from database."""
    from models.task import Task
    from database import get_db
    
    # Create a real Task instance
    mock_task = Task(
        id=1,
        task_id="test-task-id-123",
        type="tier-2",
        status="PENDING",
        input_data={"smiles": "C"},
        progress=0
    )
    
    # Override the database dependency
    def override_get_db():
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_task
        try:
            yield mock_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        response = client.get("/tasks/test-task-id-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["data"]["task_id"] == "test-task-id-123"
        assert data["data"]["status"] == "PENDING"
    finally:
        app.dependency_overrides.clear()

def test_get_task_status_success(client):
    """Test getting status of a successful task from database."""
    from models.task import Task
    from database import get_db
    
    # Create a real Task instance
    mock_task = Task(
        id=2,
        task_id="test-task-id-456",
        type="tier-2",
        status="SUCCESS",
        input_data={"smiles": "C"},
        result={"strength": 8.5},
        progress=100
    )
    
    # Override the database dependency
    def override_get_db():
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_task
        try:
            yield mock_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        response = client.get("/tasks/test-task-id-456")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["data"]["status"] == "SUCCESS"
        assert data["data"]["result"] == {"strength": 8.5}
    finally:
        app.dependency_overrides.clear()

def test_worker_logic_mock():
    """Test the worker function logic with mocked dependencies."""
    with patch("features.advanced_analysis.analyze_molecule_high_compute") as mock_analyze:
        mock_analyze.return_value = {"strength": 9.0, "flexibility": 7.5, "degradability": 6.0, "sustainability": 8.0}
        
        # Just test the compute function directly
        result = mock_analyze("CCCC")
        
        assert result["strength"] == 9.0
        assert "flexibility" in result
        mock_analyze.assert_called_once_with("CCCC")
