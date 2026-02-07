
from fastapi.testclient import TestClient
from main import app

client = TestClient(app, raise_server_exceptions=False)

def test_predict_valid_smiles():
    # Test with Methane (C)
    response = client.post("/api/predict", json={"smiles": "C"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Heuristic conversion successful"
    assert data["data"] is not None
    assert "strength" in data["data"]
    assert "flexibility" in data["data"]
    assert "degradability" in data["data"]
    assert "sustainability" in data["data"]
    
    # Check if values are within expected range (0-10)
    for key, value in data["data"].items():
        assert 0 <= value <= 10

def test_predict_complex_smiles():
    # Test with Aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    response = client.post("/api/predict", json={"smiles": smiles})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["degradability"] > 0 # Should have some degradability due to ester

def test_predict_invalid_smiles():
    response = client.post("/api/predict", json={"smiles": "InvalidSMILES"})
    assert response.status_code == 500
    # Wait, the prompt said: generic error should be { status: 500 ... }
    # Let's check my implementation in main.py.
    # I return ResponseModel(status=500...) but the HTTP status code is implicitly 200 unless I set response.status_code.
    # The requirement said "Make sure the whole fastAPI is configured as a RESTful API"
    # Usually REST APIs return 500 status code for server errors.
    # But the prompt explicitly gave a JSON format with "status": 500.
    # I adhered to the JSON format. The HTTP status code might still be 200.
    # Let's check the content.
    data = response.json()
    assert data["status"] == 500
    assert data["message"] == "Error loading the fingerprinting library!"
    assert data["error"] is not None
