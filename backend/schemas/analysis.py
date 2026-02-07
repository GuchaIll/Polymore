from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    """
    Request model for high-compute analysis.
    """
    smiles: str = Field(..., description="The SMILES string of the molecule to analyze")

class AnalysisResponse(BaseModel):
    """
    Response model for high-compute analysis results.
    """
    strength: float = Field(..., description="Calculated strength score (0-10)")
    flexibility: float = Field(..., description="Calculated flexibility score (0-10)")
    degradability: float = Field(..., description="Calculated degradability score (0-10)")
    sustainability: float = Field(..., description="Calculated sustainability score (0-10)")
    meta: Dict[str, Any] = Field(..., description="Metadata including methodology and raw values")
