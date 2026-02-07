"""
SQLAlchemy model for Tier 3 analysis results.
"""
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, Index
from sqlalchemy.sql import func
from database import Base

class Tier3Analysis(Base):
    """
    Model for storing Tier 3 analysis results.
    """
    __tablename__ = "tier3_analysis"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # SMILES string (unique identifier for the molecule analysis)
    smiles = Column(String, unique=True, index=True, nullable=False)
    
    # Analysis result data
    result = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_tier3_smiles', 'smiles'),
    )
    
    def to_dict(self):
        """Convert analysis result to dictionary."""
        return {
            "id": self.id,
            "smiles": self.smiles,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
