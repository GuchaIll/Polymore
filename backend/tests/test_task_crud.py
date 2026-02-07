"""
Tests for task CRUD endpoints.
"""
import pytest
import json
from fastapi.testclient import TestClient
from database import get_db
from models.task import Task


def test_list_tasks_empty(client):
    """Test listing tasks when database is empty."""
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["data"]["tasks"] == []
    assert data["data"]["total"] == 0
    assert data["data"]["limit"] == 50
    assert data["data"]["offset"] == 0


def test_list_tasks_with_data(client):
    """Test listing tasks with data in the database."""
    # Create some test tasks directly in the database
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Create test tasks
        task1 = Task(
            task_id="test-task-1",
            type="tier-2",
            status="SUCCESS",
            input_data={"smiles": "C"}
        )
        task2 = Task(
            task_id="test-task-2",
            type="tier-3",
            status="PENDING",
            input_data={"smiles": "CC"}
        )
        task3 = Task(
            task_id="test-task-3",
            type="tier-2",
            status="FAILURE",
            input_data={"smiles": "CCC"},
            error="Test error"
        )
        
        db.add_all([task1, task2, task3])
        db.commit()
        
        # Test list all
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 3
        assert len(data["data"]["tasks"]) == 3
        
    finally:
        db.close()


def test_list_tasks_with_type_filter(client):
    """Test listing tasks filtered by type."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        task1 = Task(task_id="test-1", type="tier-2", status="SUCCESS", input_data={})
        task2 = Task(task_id="test-2", type="tier-3", status="SUCCESS", input_data={})
        task3 = Task(task_id="test-3", type="tier-2", status="SUCCESS", input_data={})
        
        db.add_all([task1, task2, task3])
        db.commit()
        
        # Filter by tier-2
        response = client.get("/tasks?type=tier-2")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 2
        assert all(task["type"] == "tier-2" for task in data["data"]["tasks"])
        
    finally:
        db.close()


def test_list_tasks_with_status_filter(client):
    """Test listing tasks filtered by status."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        task1 = Task(task_id="test-1", type="tier-2", status="SUCCESS", input_data={})
        task2 = Task(task_id="test-2", type="tier-2", status="PENDING", input_data={})
        task3 = Task(task_id="test-3", type="tier-2", status="SUCCESS", input_data={})
        
        db.add_all([task1, task2, task3])
        db.commit()
        
        # Filter by SUCCESS status
        response = client.get("/tasks?status=SUCCESS")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 2
        assert all(task["status"] == "SUCCESS" for task in data["data"]["tasks"])
        
    finally:
        db.close()


def test_list_tasks_with_pagination(client):
    """Test pagination functionality."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Create 5 tasks
        for i in range(5):
            task = Task(
                task_id=f"test-{i}",
                type="tier-2",
                status="SUCCESS",
                input_data={}
            )
            db.add(task)
        db.commit()
        
        # Test limit
        response = client.get("/tasks?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["tasks"]) == 2
        assert data["data"]["total"] == 5
        assert data["data"]["limit"] == 2
        
        # Test offset
        response = client.get("/tasks?limit=2&offset=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["tasks"]) == 2
        assert data["data"]["offset"] == 2
        
    finally:
        db.close()


def test_update_task_success(client):
    """Test updating a task successfully."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Create a test task
        task = Task(
            task_id="update-test",
            type="tier-2",
            status="PENDING",
            input_data={"smiles": "C"},
            progress=0
        )
        db.add(task)
        db.commit()
        
        # Update the task
        update_data = {
            "status": "SUCCESS",
            "progress": 100,
            "result": {"prediction": 42.0}
        }
        response = client.patch("/tasks/update-test", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["data"]["status"] == "SUCCESS"
        assert data["data"]["progress"] == 100
        assert data["data"]["result"] == {"prediction": 42.0}
        
    finally:
        db.close()


def test_update_task_partial(client):
    """Test updating only some fields of a task."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        task = Task(
            task_id="partial-update",
            type="tier-2",
            status="PENDING",
            input_data={"smiles": "C"},
            progress=0
        )
        db.add(task)
        db.commit()
        
        # Update only progress
        update_data = {"progress": 50}
        response = client.patch("/tasks/partial-update", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["progress"] == 50
        assert data["data"]["status"] == "PENDING"  # Should remain unchanged
        
    finally:
        db.close()


def test_update_task_not_found(client):
    """Test updating a non-existent task."""
    update_data = {"status": "SUCCESS"}
    response = client.patch("/tasks/non-existent-id", json=update_data)
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["error"].lower()


def test_delete_task_success(client):
    """Test deleting a task successfully."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Create a test task
        task = Task(
            task_id="delete-test",
            type="tier-2",
            status="SUCCESS",
            input_data={"smiles": "C"}
        )
        db.add(task)
        db.commit()
        
        # Delete the task
        response = client.delete("/tasks/delete-test")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["data"]["deleted"] is True
        assert data["data"]["task_id"] == "delete-test"
        
        # Verify task is actually deleted
        db.expire_all()  # Clear session cache
        deleted_task = db.query(Task).filter(Task.task_id == "delete-test").first()
        assert deleted_task is None
        
    finally:
        db.close()


def test_delete_task_not_found(client):
    """Test deleting a non-existent task."""
    response = client.delete("/tasks/non-existent-id")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["error"].lower()


def test_crud_workflow(client):
    """Test a complete CRUD workflow."""
    from database import SessionLocal
    db = SessionLocal()
    
    try:
        # Create
        task = Task(
            task_id="workflow-test",
            type="tier-3",
            status="PENDING",
            input_data={"smiles": "CCCC"}
        )
        db.add(task)
        db.commit()
        
        # Read (List)
        response = client.get("/tasks")
        assert response.status_code == 200
        assert response.json()["data"]["total"] >= 1
        
        # Read (Get specific)
        response = client.get("/tasks/workflow-test")
        assert response.status_code == 200
        assert response.json()["data"]["task_id"] == "workflow-test"
        
        # Update
        response = client.patch("/tasks/workflow-test", json={"status": "SUCCESS"})
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "SUCCESS"
        
        # Delete
        response = client.delete("/tasks/workflow-test")
        assert response.status_code == 200
        assert response.json()["data"]["deleted"] is True
        
        # Verify deletion
        response = client.get("/tasks/workflow-test")
        assert response.status_code == 404
        
    finally:
        db.close()
