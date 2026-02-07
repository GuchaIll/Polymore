import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from worker import analyze_molecule_task

# Client is provided by conftest.py fixture

def test_submit_analysis(client):
    # Mock the Celery task.delay() call
    with patch("main.analyze_molecule_task.delay") as mock_delay:
        mock_task = MagicMock()
        mock_task.id = "test-task-id-123"
        mock_delay.return_value = mock_task
        
        payload = {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        response = client.post("/predict/tier-2", json=payload)
        
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == 202
        assert data["data"]["task_id"] == "test-task-id-123"
        assert data["data"]["status"] == "submitted"
        
        # Verify the task was called with correct arguments
        mock_delay.assert_called_once_with("CC(=O)Oc1ccccc1C(=O)O")

def test_get_task_status_pending(client):
    # Create a task in the database first
    from models.task import Task
    from database import SessionLocal
    
    db = SessionLocal()
    db_task = Task(
        task_id="test-task-id-123",
        type="tier-2",
        status="PENDING",
        input_data={"smiles": "C"}
    )
    db.add(db_task)
    db.commit()
    db.close()
    
    response = client.get("/tasks/test-task-id-123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["data"]["task_id"] == "test-task-id-123"
    assert data["data"]["status"] == "PENDING"

def test_get_task_status_success(client):
    # Create a successful task in the database
    from models.task import Task
    from database import SessionLocal
    
    db = SessionLocal()
    db_task = Task(
        task_id="test-task-id-456",
        type="tier-2",
        status="SUCCESS",
        input_data={"smiles": "C"},
        result={"strength": 8.5},
        progress=100
    )
    db.add(db_task)
    db.commit()
    db.close()
    
    response = client.get("/tasks/test-task-id-456")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["data"]["status"] == "SUCCESS"
    assert data["data"]["result"] == {"strength": 8.5}

def test_worker_logic_mock():
    # Test the worker function logic by mocking internal call and database
    with patch("features.advanced_analysis.analyze_molecule_high_compute") as mock_analyze:
        with patch("worker.update_task_in_db") as mock_update_db:
            mock_analyze.return_value = {"strength": 9.0}
            
            # Create a mock bound task with request.id
            mock_task = MagicMock()
            mock_task.request.id = "test-worker-task-id"
            
            # Call the worker task bound to the mock
            result = analyze_molecule_task.__wrapped__(mock_task, "CCCC")
            
            assert result == {"strength": 9.0}
            mock_analyze.assert_called_once()
