from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime

class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None

class TaskDetail(BaseModel):
    """Detailed task information including metadata."""
    id: int
    task_id: str
    type: str
    status: str
    input_data: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0
    progress_message: Optional[str] = None
    created_at: str
    updated_at: str

class TaskListResponse(BaseModel):
    """Response model for listing tasks with pagination."""
    tasks: List[TaskDetail]
    total: int
    limit: int
    offset: int

class TaskUpdateRequest(BaseModel):
    """Request model for updating task fields."""
    status: Optional[str] = Field(None, description="Task status (PENDING, PROGRESS, SUCCESS, FAILURE)")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message")
    progress_message: Optional[str] = Field(None, description="Progress message")

class TaskDeleteResponse(BaseModel):
    """Response model for task deletion."""
    task_id: str
    deleted: bool
    message: str
