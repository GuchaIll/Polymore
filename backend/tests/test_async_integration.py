import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from worker import analyze_molecule_task

client = TestClient(app)

@patch("main.analyze_molecule_task.delay")
def test_submit_analysis(mock_delay):
    # Mock the Celery task.delay() call
    mock_task = MagicMock()
    mock_task.id = "test-task-id-123"
    mock_delay.return_value = mock_task
    
    payload = {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
    response = client.post("/api/analyze/high-compute", json=payload)
    
    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "test-task-id-123"
    assert data["status"] == "submitted"
    
    # Verify the task was called with correct arguments
    mock_delay.assert_called_once_with("CC(=O)Oc1ccccc1C(=O)O")

@patch("main.AsyncResult")
def test_get_task_status_pending(mock_async_result):
    # Mock AsyncResult for a pending task
    mock_result = MagicMock()
    mock_result.status = "PENDING"
    mock_result.result = None
    mock_async_result.return_value = mock_result
    
    response = client.get("/api/tasks/test-task-id-123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "test-task-id-123"
    assert data["status"] == "PENDING"
    assert data["result"] is None

@patch("main.AsyncResult")
def test_get_task_status_success(mock_async_result):
    # Mock AsyncResult for a successful task
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"
    mock_result.result = {"strength": 8.5}
    mock_async_result.return_value = mock_result
    
    response = client.get("/api/tasks/test-task-id-123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"] == {"strength": 8.5}

def test_worker_logic_mock():
    # Test the worker function logic by mocking internal call
    with patch("features.advanced_analysis.analyze_molecule_high_compute") as mock_analyze:
        mock_analyze.return_value = {"strength": 9.0}
        
        # Call the worker task directly (not async)
        result = analyze_molecule_task("CCCC")
        
        assert result == {"strength": 9.0}
        mock_analyze.assert_called_once_with("CCCC")
