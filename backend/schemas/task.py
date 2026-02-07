from pydantic import BaseModel
from typing import Optional, Any, Dict

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
