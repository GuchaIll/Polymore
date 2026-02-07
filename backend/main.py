from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any, TypeVar, Generic
from heuristics import predict_properties

app = FastAPI(
    title="Polymore Backend API",
    description="API for predicting polymer properties using RDKit heuristics.",
    version="1.0.0"
)

T = TypeVar('T')

class HeuristicPredictedProperties(BaseModel):
    strength: float
    flexibility: float
    degradability: float
    sustainability: float

class ResponseModel(BaseModel, Generic[T]):
    status: int
    message: str
    data: Optional[T] = None
    error: Optional[str] = None

class SmiRequest(BaseModel):
    smiles: str

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": 500,
            "message": "Error loading the fingerprinting library!",
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
    properties = predict_properties(request.smiles)
    return ResponseModel(
        status=200,
        message="Heuristic conversion successful",
        data=HeuristicPredictedProperties(**properties)
    )
