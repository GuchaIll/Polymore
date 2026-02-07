from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class HeuristicPredictedProperties(BaseModel):
    strength: float
    flexibility: float
    degradability: float
    sustainability: float
    sas_score: float
    meta: Dict[str, Any]

class Position(BaseModel):
    """3D position in canvas space."""
    x: float
    y: float
    z: float

class MoleculeData(BaseModel):
    """Data for a placed molecule from the frontend."""
    id: int
    smiles: str
    name: str
    position: Position
    connections: List[int]

class ValidatePolymerRequest(BaseModel):
    """Request body for polymer validation endpoint."""
    molecules: List[MoleculeData]
    generatedSmiles: str

class ValidationResponse(BaseModel):
    """Response body for validation results."""
    isValid: bool
    canonicalSmiles: str
    errors: List[str]
    warnings: List[str]
    polymerType: str
    molecularWeight: Optional[float] = None
    aromaticRings: Optional[int] = None
