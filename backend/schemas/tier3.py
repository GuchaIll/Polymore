from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class Tier3PredictionResponse(BaseModel):
    """Response from tier-3 serverless prediction."""
    smiles: str
    predictions: Dict[str, float]

class Tier3Request(BaseModel):
    """Request for tier-3 prediction."""
    smiles: str

class Tier3AnalysisBase(BaseModel):
    """Base schema for Tier 3 analysis."""
    smiles: str
    result: Dict[str, Any]

class Tier3AnalysisCreate(Tier3AnalysisBase):
    """Schema for creating a Tier 3 analysis result."""
    pass

class Tier3AnalysisUpdate(BaseModel):
    """Schema for updating a Tier 3 analysis result."""
    result: Optional[Dict[str, Any]] = None

class Tier3AnalysisResult(Tier3AnalysisBase):
    """Schema for returning a Tier 3 analysis result."""
    id: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
