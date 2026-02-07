"""
SQLAlchemy model for task persistence.
"""
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, Index
from sqlalchemy.sql import func
from database import Base

class Task(Base):
    """
    Model for storing task information and results.
    """
    __tablename__ = "tasks"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Celery task ID (UUID)
    task_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Task type (tier-2, tier-3, etc.)
    type = Column(String(50), nullable=False, index=True)
    
    # Task status (PENDING, PROGRESS, SUCCESS, FAILURE)
    status = Column(String(50), nullable=False, default="PENDING")
    
    # Input data (SMILES string and other params as JSON)
    input_data = Column(JSON, nullable=False)
    
    # Result data (predictions, analysis results as JSON)
    result = Column(JSON, nullable=True)
    
    # Error message if task failed
    error = Column(Text, nullable=True)
    
    # Progress percentage (0-100)
    progress = Column(Integer, default=0)
    
    # Progress message
    progress_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_task_type_status', 'type', 'status'),
        Index('idx_created_at', 'created_at'),
    )
    
    def to_dict(self):
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "type": self.type,
            "status": self.status,
            "input_data": self.input_data,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
