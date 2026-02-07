
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)

def test_health_standardized():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Health check successful"
    assert data["data"] == {"status": "ok"}
    assert data["error"] is None

def test_404_standardized():
    response = client.get("/non_existent_endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == 404
    assert data["message"] == "Resource not found"
    assert data["data"] is None
    assert data["error"] == "Resource not found"

def test_predict_success():
    response = client.post("/api/predict", json={"smiles": "C"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Heuristic conversion successful"
    assert data["data"] is not None
    assert data["error"] is None

def test_predict_error():
    response = client.post("/api/predict", json={"smiles": "InvalidSMILES"})
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == 500
    assert "Invalid SMILES" in data["message"]
    assert data["data"] is None
    assert data["error"] is not None
