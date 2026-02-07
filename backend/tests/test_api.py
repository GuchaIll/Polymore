
from fastapi.testclient import TestClient
from main import app

# Client is provided by conftest.py fixture

def test_predict_valid_smiles(client):
    # Test with Methane (C)
    response = client.post("/predict/tier-1", json={"smiles": "C"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == 200
    assert data["message"] == "Heuristic conversion successful"
    assert data["data"] is not None
    assert "strength" in data["data"]
    assert "flexibility" in data["data"]
    assert "degradability" in data["data"]
    assert "sustainability" in data["data"]
    assert "sas_score" in data["data"]
    assert "meta" in data["data"]
    
    # Check if values are within expected range (0-10)
    for key, value in data["data"].items():
        pass  # Skip meta dict

def test_predict_complex_smiles(client):
    # Test with Aspirin (CC(=O)OC1=CC=CC=C1C(=O)O)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    response = client.post("/predict/tier-1", json={"smiles": smiles})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["degradability"] > 0 # Should have some degradability due to ester

def test_predict_invalid_smiles(client):
    response = client.post("/predict/tier-1", json={"smiles": "InvalidSMILES"})
    assert response.status_code == 500
    data = response.json()
    assert data["status"] == 500
    assert data["message"] == "Invalid SMILES string"
    assert data["error"] is not None
