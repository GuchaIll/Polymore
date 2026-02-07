from pydantic import BaseModel
from typing import Optional, Dict

class Tier3PredictionResponse(BaseModel):
    """Response from tier-3 serverless prediction."""
    smiles: str
    predictions: Dict[str, float]

class Tier3Request(BaseModel):
    """Request for tier-3 prediction."""
    smiles: str
