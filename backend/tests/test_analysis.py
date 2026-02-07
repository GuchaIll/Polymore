import pytest
import sys
from unittest.mock import MagicMock, patch

# 1. Mock tblite module BEFORE importing features.advanced_analysis
# This prevents ImportError if tblite is not installed in the test environment
mock_tblite = MagicMock()
sys.modules["tblite"] = mock_tblite
sys.modules["tblite.interface"] = mock_tblite

# Now we can safely import
from fastapi.testclient import TestClient
from main import app
from features.advanced_analysis import analyze_molecule_high_compute

# Client is provided by conftest.py fixture

def test_analyze_endpoint_submission(client):
    """
    Test that the tier-2 analysis endpoint correctly submits a Celery task.
    """
    with patch("main.analyze_molecule_task.delay") as mock_delay:
        mock_task = MagicMock()
        mock_task.id = "test-task-id"
        mock_delay.return_value = mock_task
        
        response = client.post("/predict/tier-2", json={"smiles": "C"})
        
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == 202
        assert data["data"]["task_id"] == "test-task-id"
        assert data["data"]["status"] == "submitted"
        
        # Verify it was called with the SMILES string
        mock_delay.assert_called_once_with("C")

def test_analyze_compute_logic():
    """
    Test the computational logic of analyze_molecule_high_compute
    by mocking the internal Calculator class and RDKit embedding.
    """
    # Mock the Calculator class that was imported in advanced_analysis
    with patch("features.advanced_analysis.Calculator") as MockCalculator:
        # Setup the mock calculator instance
        mock_calc_instance = MockCalculator.return_value
        
        # Mock singlepoint() return value
        mock_calc_instance.singlepoint.return_value = {
            "energy": -40.0,                  # Hartree
            "dipole": [0.0, 0.0, 1.0],        # Bohr
            "orbital_energies": [-0.4, -0.1], # Hartree
            "orbital_occupations": [2.0, 0.0],
            "gradient": [0.01, 0.01, 0.01]
        }
        
        # Mock RDKit parts to avoid expensive/random conformer generation
        with patch("rdkit.Chem.AllChem.EmbedMultipleConfs", return_value=[0, 1, 2]):
            with patch("rdkit.Chem.Mol.GetConformer") as mock_conf:
                 mock_conf.return_value.GetPositions.return_value = [[0.0, 0.0, 0.0]] # Dummy positions
                 
                 # Run the function
                 result = analyze_molecule_high_compute("C")
                 
                 # detailed checks
                 assert result["strength"] >= 0
                 assert result["flexibility"] >= 0 
                 assert result["degradability"] >= 0
                 assert result["sustainability"] >= 0
                 
                 # Check metadata
                 assert "meta" in result
                 assert result["meta"]["method"] == "GFN2-xTB (Ensemble)"
                 assert result["meta"]["conformers_scanned"] == 3
