"""
Router for Tier 3 analysis results operations.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import get_db
from models.tier3_analysis import Tier3Analysis
from schemas.tier3 import Tier3AnalysisCreate, Tier3AnalysisResult, Tier3AnalysisUpdate
from schemas.common import ResponseModel

router = APIRouter(
    prefix="/tier3",
    tags=["tier3"]
)

@router.get("/", response_model=ResponseModel[List[Tier3AnalysisResult]])
def list_tier3_results(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all Tier 3 analysis results.
    """
    results = db.query(Tier3Analysis).offset(skip).limit(limit).all()
    return ResponseModel(
        status=200,
        message="Retrieved Tier 3 results successfully",
        data=[Tier3AnalysisResult.model_validate(r) for r in results]
    )

@router.post("/", response_model=ResponseModel[Tier3AnalysisResult])
def create_tier3_result(
    analysis: Tier3AnalysisCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new Tier 3 analysis result.
    """
    db_analysis = Tier3Analysis(
        smiles=analysis.smiles,
        result=analysis.result
    )
    try:
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        return ResponseModel(
            status=201,
            message="Created Tier 3 result successfully",
            data=Tier3AnalysisResult.model_validate(db_analysis)
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Analysis result for SMILES '{analysis.smiles}' already exists")

@router.get("/{smiles}", response_model=ResponseModel[Tier3AnalysisResult])
def get_tier3_result(
    smiles: str,
    db: Session = Depends(get_db)
):
    """
    Get a Tier 3 analysis result by SMILES string.
    """
    db_analysis = db.query(Tier3Analysis).filter(Tier3Analysis.smiles == smiles).first()
    if db_analysis is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    return ResponseModel(
        status=200,
        message="Retrieved Tier 3 result successfully",
        data=Tier3AnalysisResult.model_validate(db_analysis)
    )

@router.put("/{smiles}", response_model=ResponseModel[Tier3AnalysisResult])
def update_tier3_result(
    smiles: str,
    analysis_update: Tier3AnalysisUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a Tier 3 analysis result.
    """
    db_analysis = db.query(Tier3Analysis).filter(Tier3Analysis.smiles == smiles).first()
    if db_analysis is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    if analysis_update.result is not None:
        db_analysis.result = analysis_update.result
    
    db.commit()
    db.refresh(db_analysis)
    
    return ResponseModel(
        status=200,
        message="Updated Tier 3 result successfully",
        data=Tier3AnalysisResult.model_validate(db_analysis)
    )

@router.delete("/{smiles}", response_model=ResponseModel[dict])
def delete_tier3_result(
    smiles: str,
    db: Session = Depends(get_db)
):
    """
    Delete a Tier 3 analysis result.
    """
    db_analysis = db.query(Tier3Analysis).filter(Tier3Analysis.smiles == smiles).first()
    if db_analysis is None:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    
    db.delete(db_analysis)
    db.commit()
    
    return ResponseModel(
        status=200,
        message="Deleted Tier 3 result successfully",
        data={"smiles": smiles}
    )
