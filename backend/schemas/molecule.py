from typing import Optional
from pydantic import BaseModel

class SmiRequest(BaseModel):
    smiles: str

class SmilesValidationRequest(BaseModel):
    """Request for simple SMILES validation."""
    smiles: str

class SmilesValidationResponse(BaseModel):
    """Response for SMILES validation."""
    isValid: bool
    canonicalSmiles: Optional[str] = None
    error: Optional[str] = None
    molecularWeight: Optional[float] = None
    formula: Optional[str] = None
