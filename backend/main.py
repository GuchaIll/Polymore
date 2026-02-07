"""
Module: main.py
Purpose: FastAPI backend for polymer GNN designer with RDKit validation
Inputs: SMILES strings, placed molecule data
Outputs: Validation results, canonical SMILES, property predictions
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import logging

from schemas.common import ResponseModel
from schemas.molecule import (
    SmiRequest,
    SmilesValidationRequest,
    SmilesValidationResponse,
)
from schemas.polymer import (
    HeuristicPredictedProperties,
    ValidatePolymerRequest,
    ValidationResponse,
    MoleculeData,
)
from core.exceptions import (
    BaseAPIException,
    BadRequestException,
    NotFoundException,
    ServerException,
)
from features.heuristics import predict_properties

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Polymore Backend API",
    description="API for polymer SMILES validation and property prediction using RDKit.",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://spotme.life", "https://spotme.life", "http://api.spotme.life", "https://api.spotme.life", "http://localhost:8000", "https://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": exc.status_code,
            "message": exc.detail,
            "data": None,
            "error": exc.detail
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": 500,
            "message": "Internal Server Error",
            "data": None,
            "error": str(exc)
        }
    )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
     return JSONResponse(
        status_code=404,
        content={
            "status": 404,
            "message": "Resource not found",
            "data": None,
            "error": "Resource not found"
        }
    )

@app.get("/health", response_model=ResponseModel[dict])
def read_health():
    return ResponseModel(
        status=200,
        message="Health check successful",
        data={"status": "ok"}
    )

@app.post("/api/predict", response_model=ResponseModel[HeuristicPredictedProperties])
def predict_heuristics(request: SmiRequest):
    """Predict polymer properties from SMILES using heuristics."""
    try:
        properties = predict_properties(request.smiles)
        return ResponseModel(
            status=200,
            message="Heuristic conversion successful",
            data=HeuristicPredictedProperties(**properties)
        )
    except Exception as e:
        logger.error(f"Prediction failed for SMILES {request.smiles}: {e}")
        raise ServerException(detail=str(e))

@app.post("/api/validate-smiles", response_model=ResponseModel[SmilesValidationResponse])
def validate_smiles(request: SmilesValidationRequest):
    """
    Validate a single SMILES string using RDKit.
    """
    mol = Chem.MolFromSmiles(request.smiles)
    
    if mol is None:
        return ResponseModel(
            status=200, 
            message="Validation completed",
            data=SmilesValidationResponse(
                isValid=False,
                error="Invalid SMILES: could not parse molecule"
            )
        )
    
    # Get canonical SMILES
    canonical = Chem.MolToSmiles(mol, canonical=True)
    
    # Calculate basic properties
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    
    return ResponseModel(
        status=200,
        message="Validation successful",
        data=SmilesValidationResponse(
            isValid=True,
            canonicalSmiles=canonical,
            molecularWeight=round(mw, 2),
            formula=formula
        )
    )

@app.post("/api/validate-polymer", response_model=ResponseModel[ValidationResponse])
def validate_polymer(request: ValidatePolymerRequest):
    """
    Validate a polymer configuration from placed molecules.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if not request.molecules:
        return ResponseModel(
            status=400,
            message="No molecules provided",
            data=ValidationResponse(
                isValid=False,
                canonicalSmiles="",
                errors=["No molecules provided"],
                warnings=[],
                polymerType="unknown"
            )
        )
    
    # Validate each molecule's SMILES individually
    validated_mols: Dict[int, Any] = {}
    
    for mol_data in request.molecules:
        try:
            rdkit_mol = Chem.MolFromSmiles(mol_data.smiles)
            
            if rdkit_mol is None:
                errors.append(
                    f"Invalid SMILES for {mol_data.name}: '{mol_data.smiles}'"
                )
                continue
                
            validated_mols[mol_data.id] = {
                "mol": rdkit_mol,
                "data": mol_data,
                "canonical": Chem.MolToSmiles(rdkit_mol, canonical=True)
            }
            
        except Exception as e:
            errors.append(f"Error parsing {mol_data.name}: {str(e)}")
    
    # Check connectivity validity
    mol_ids = {m.id for m in request.molecules}
    for mol_data in request.molecules:
        for conn_id in mol_data.connections:
            if conn_id not in mol_ids:
                errors.append(
                    f"{mol_data.name} connected to non-existent molecule ID {conn_id}"
                )
    
    # Determine polymer type based on connectivity
    polymer_type = _classify_polymer_type(request.molecules)
    
    # Try to validate the combined SMILES
    canonical_smiles = ""
    mw = None
    aromatic_rings = None
    
    if request.generatedSmiles:
        try:
            combined_mol = Chem.MolFromSmiles(request.generatedSmiles)
            
            if combined_mol is not None:
                canonical_smiles = Chem.MolToSmiles(combined_mol, canonical=True)
                mw = round(Descriptors.MolWt(combined_mol), 2)
                aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(combined_mol)
            else:
                # Fallback: join individual canonical SMILES
                canonical_parts = [
                    v["canonical"] for v in validated_mols.values()
                ]
                canonical_smiles = ".".join(canonical_parts)
                warnings.append(
                    "Combined SMILES invalid, using individual fragments"
                )
                
        except Exception as e:
            warnings.append(f"Could not validate combined SMILES: {str(e)}")
            # Use individual SMILES as fallback
            canonical_parts = [v["canonical"] for v in validated_mols.values()]
            canonical_smiles = ".".join(canonical_parts)
    
    # Check for unusual valence states (from RDKit docs)
    for mol_id, mol_info in validated_mols.items():
        rdkit_mol = mol_info["mol"]
        try:
            # This will catch invalid valence states
            Chem.SanitizeMol(rdkit_mol)
        except Exception as e:
            warnings.append(
                f"Potential valence issue in {mol_info['data'].name}: {str(e)}"
            )
    
    # Check for aromaticity consistency
    _check_aromaticity(validated_mols, warnings)
    
    return ResponseModel(
        status=200,
        message="Polymer validation complete",
        data=ValidationResponse(
            isValid=len(errors) == 0,
            canonicalSmiles=canonical_smiles,
            errors=errors,
            warnings=warnings,
            polymerType=polymer_type,
            molecularWeight=mw,
            aromaticRings=aromatic_rings
        )
    )


def _classify_polymer_type(molecules: List[MoleculeData]) -> str:
    """
    Classify polymer type based on connectivity graph.
    """
    if not molecules:
        return "unknown"
    
    # Build adjacency for cycle detection
    adjacency: Dict[int, List[int]] = {m.id: m.connections for m in molecules}
    
    # Check for cycles using DFS
    visited = set()
    
    def has_cycle(node: int, parent: Optional[int]) -> bool:
        visited.add(node)
        
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    # Check all components for cycles
    for mol in molecules:
        if mol.id not in visited:
            if has_cycle(mol.id, None):
                return "cyclic"
    
    # Check for branching (any node with > 2 connections)
    if any(len(m.connections) > 2 for m in molecules):
        return "branched"
    
    # Check for linear chain (exactly 2 endpoints with 1 connection each)
    endpoints = [m for m in molecules if len(m.connections) == 1]
    if len(endpoints) == 2:
        return "linear"
    
    if len(molecules) == 1:
        return "linear"
    
    return "unknown"


def _check_aromaticity(
    validated_mols: Dict[int, Any], 
    warnings: List[str]
) -> None:
    """
    Check aromaticity consistency across connected molecules.
    """
    for mol_id, mol_info in validated_mols.items():
        rdkit_mol = mol_info["mol"]
        
        # Count aromatic atoms
        aromatic_atoms = sum(
            1 for atom in rdkit_mol.GetAtoms() if atom.GetIsAromatic()
        )
        
        # Check for broken aromaticity (aromatic atoms not in aromatic rings)
        ring_info = rdkit_mol.GetRingInfo()
        aromatic_ring_atoms = set()
        
        for ring in ring_info.AtomRings():
            ring_atoms = list(ring)
            if all(rdkit_mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms):
                aromatic_ring_atoms.update(ring_atoms)
        
        isolated_aromatic = aromatic_atoms - len(aromatic_ring_atoms)
        if isolated_aromatic > 0:
            warnings.append(
                f"{mol_info['data'].name} has aromatic atoms outside aromatic rings"
            )


@app.post("/api/generate-psmiles", response_model=ResponseModel[dict])
def generate_polymer_smiles(request: ValidatePolymerRequest):
    """
    Generate polymer SMILES (pSMILES) notation from placed molecules.
    """
    validation_response_model = validate_polymer(request)
    validation = validation_response_model.data
    
    if not validation.isValid:
        return ResponseModel(
            status=400,
            message="Validation failed",
            data={
                "errors": validation.errors
            }
        )
        
    # For linear polymers, add star atoms at ends
    canonical = validation.canonicalSmiles
    
    if validation.polymerType == "linear" and canonical:
        # Simple pSMILES notation: [*]...monomer...[*]
        psmiles = f"[*]{canonical}[*]"
    else:
        # For branched/cyclic, use standard notation
        psmiles = canonical
    
    return ResponseModel(
        status=200,
        message="pSMILES generation successful",
        data={
            "psmiles": psmiles,
            "smiles": canonical,
            "polymerType": validation.polymerType,
            "molecularWeight": validation.molecularWeight
        }
    )
