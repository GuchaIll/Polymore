
from fastapi.testclient import TestClient
from main import app

# Client is provided by conftest.py fixture

def test_health_standardized(client):
    """Test that /health returns standardized ResponseModel."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Health check successful"
    assert data["data"] == {"status": "ok"}
    assert data["error"] is None

def test_404_standardized(client):
    """Test that 404 errors return standardized ResponseModel."""
    response = client.get("/non_existent_endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == 404
    assert data["message"] == "Resource not found"
    assert data["data"] is None
    assert data["error"] == "Resource not found"

def test_predict_success(client):
    """Test that tier-1 prediction returns standardized ResponseModel."""
    response = client.post("/predict/tier-1", json={"smiles": "C"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Heuristic conversion successful"
    assert data["data"] is not None
    assert data["error"] is None

def test_predict_error(client):
    """Test that tier-1 errors return standardized ResponseModel."""
    response = client.post("/predict/tier-1", json={"smiles": "InvalidSMILES"})
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == 500
    assert "Invalid SMILES" in data["message"]
    assert data["data"] is None
    assert data["error"] is not None

def test_validation_error(client):
    """Test that validation errors return standardized 422 response."""
    # Sending missing required field 'smiles'
    response = client.post("/predict/tier-1", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == 422
    assert data["message"] == "Validation Error"
    assert data["data"] is None
    assert data["error"] is not None
