"""
Module: main.py
Purpose: FastAPI backend for polymer GNN designer with RDKit validation
Inputs: SMILES strings, placed molecule data
Outputs: Validation results, canonical SMILES, property predictions
"""

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os
from sqlalchemy.orm import Session

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

import logging

from schemas.common import ResponseModel
from schemas.molecule import (
    SmiRequest,
    SmilesValidationRequest,
    SmilesValidationResponse,
)
from schemas.tier3 import (
    Tier3Request,
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
# from features.advanced_analysis import analyze_molecule_high_compute  # Moved to worker
from schemas.analysis import AnalysisRequest, AnalysisResponse
from schemas.task import TaskSubmissionResponse, TaskStatusResponse
from worker import celery_app, analyze_molecule_task, predict_tier_3_task
from celery.result import AsyncResult
from database import get_db, init_db
from models.task import Task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Polymore Backend API",
    description="API for polymer SMILES validation and property prediction using RDKit.",
    version="1.0.0"
)

# Startup event to initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database schema on application startup."""
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Database schema initialized")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://spotme.life", "https://spotme.life", "http://api.spotme.life", "https://api.spotme.life", "http://localhost:8000", "https://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": 422,
            "message": "Validation Error",
            "data": None,
            "error": str(exc.errors())
        }
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

@app.post("/predict/tier-1", response_model=ResponseModel[HeuristicPredictedProperties])
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

@app.post("/predict/tier-2", status_code=202, response_model=ResponseModel[TaskSubmissionResponse])
def analyze_high_compute(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    Submit a high-compute analysis task to the Celery queue.
    
    This endpoint accepts a molecule SMILES string and offloads the intensive
    quantum mechanical analysis (GFN2-xTB) to a background worker.
    
    Returns:
        ResponseModel containing the task_id and submission status.
    """
    try:
        task = analyze_molecule_task.delay(request.smiles)
        
        # Create task record in database
        db_task = Task(
            task_id=task.id,
            type="tier-2",
            status="PENDING",
            input_data={"smiles": request.smiles}
        )
        db.add(db_task)
        db.commit()
        
        return ResponseModel(
            status=202,
            message="Analysis submitted to queue",
            data=TaskSubmissionResponse(
                task_id=task.id,
                status="submitted",
                message="Analysis submitted successfully"
            )
        )
    except Exception as e:
        logger.error(f"Failed to submit analysis task: {e}")
        db.rollback()
        raise ServerException(detail=str(e))

@app.get("/tasks/{task_id}", response_model=ResponseModel[TaskStatusResponse])
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the status and result of a background task.
    
    Poll this endpoint using the `task_id` returned from the submission endpoint.
    It returns the current status (PENDING, PROGRESS, SUCCESS, FAILURE) and
    any available progress metadata or final results.
    
    This endpoint first queries the database for task information, then falls back
    to the Celery result backend if needed.
    """
    logger.info(f"Checking task status for: {task_id}")
    
    # Try to get task from database first
    db_task = db.query(Task).filter(Task.task_id == task_id).first()
    
    if db_task:
        # Use database record
        response_data = TaskStatusResponse(
            task_id=task_id,
            status=db_task.status,
            result=db_task.result,
            error=db_task.error,
            progress=db_task.progress,
            message=db_task.progress_message
        )
        
        return ResponseModel(
            status=200,
            message=f"Task status: {db_task.status}",
            data=response_data
        )
    else:
        # Fallback to Celery result backend (for backward compatibility)
        task_result = AsyncResult(task_id, app=celery_app)
        
        response_data = TaskStatusResponse(
            task_id=task_id,
            status=task_result.status,
        )
        
        if task_result.status == 'SUCCESS':
            response_data.result = task_result.result if isinstance(task_result.result, dict) else {"data": task_result.result}
            response_data.progress = 100
            
        elif task_result.status == 'FAILURE':
            response_data.error = str(task_result.result)
            
        elif task_result.status == 'PROGRESS':
            meta = task_result.result if isinstance(task_result.result, dict) else {}
            response_data.progress = meta.get('current', 0)
            response_data.message = meta.get('status', '')
            
        return ResponseModel(
            status=200,
            message=f"Task status: {task_result.status}",
            data=response_data
        )

@app.post("/predict/tier-3", status_code=202, response_model=ResponseModel[TaskSubmissionResponse])
def predict_tier_3(request: Tier3Request, db: Session = Depends(get_db)):
    """
    Submit a tier-3 prediction task to the Celery queue.
    
    This endpoint accepts a SMILES string and offloads the prediction to a
    background worker that communicates with the serverless GPU.
    
    Args:
        request: Contains the SMILES string to analyze
        
    Returns:
        ResponseModel containing the task_id and submission status.
    """
    try:
        task = predict_tier_3_task.delay(request.smiles)
        
        # Create task record in database
        db_task = Task(
            task_id=task.id,
            type="tier-3",
            status="PENDING",
            input_data={"smiles": request.smiles}
        )
        db.add(db_task)
        db.commit()
        
        return ResponseModel(
            status=202,
            message="Tier-3 prediction submitted to queue",
            data=TaskSubmissionResponse(
                task_id=task.id,
                status="submitted",
                message="Tier-3 prediction submitted successfully"
            )
        )
    except Exception as e:
        logger.error(f"Failed to submit tier-3 prediction task: {e}")
        db.rollback()
        raise ServerException(detail=str(e))

@app.post("/validate-smiles", response_model=ResponseModel[SmilesValidationResponse])
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

@app.post("/validate-polymer", response_model=ResponseModel[ValidationResponse])
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
