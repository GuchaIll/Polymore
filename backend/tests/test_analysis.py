import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app
from features.advanced_analysis import analyze_molecule_high_compute

client = TestClient(app)

def test_analyze_endpoint_mocked():
    """
    Test the endpoint with mocked results to ensure API plumbing works 
    even if tblite is not installed.
    """
    mock_results = {
        "strength": 8.5,
        "flexibility": 2.1,
        "degradability": 0.5,
        "sustainability": 9.0,
        "meta": {
            "method": "GFN2-xTB (Semi-Empirical QM)",
            "dipole_debye": 2.5,
            "total_energy_hartree": -75.1234
        }
    }
    
    with patch('main.analyze_molecule_high_compute', return_value=mock_results):
        response = client.post("/api/analyze/high-compute", json={"smiles": "C"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == 200
        assert data["data"]["strength"] == 8.5
        assert data["data"]["flexibility"] == 2.1

def test_analyze_feature_import_error():
    """
    Test that the feature raises ImportError if tblite is missing.
    """
    # We force Calculator to be None
    with patch('features.advanced_analysis.Calculator', None):
        with pytest.raises(ImportError):
            analyze_molecule_high_compute("C")

# Note: We can't easily test the actual tblite calculation without the binary installed.
