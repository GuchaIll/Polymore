import os
from celery import Celery
from features.advanced_analysis import analyze_molecule_high_compute
import logging
import httpx
from database import SessionLocal
from models.task import Task

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Redis URL from environment or default
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "polymore_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

def update_task_in_db(task_id: str, status: str, result=None, error=None, progress=None, progress_message=None):
    """
    Update task status in database.
    """
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if task:
            task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if progress is not None:
                task.progress = progress
            if progress_message is not None:
                task.progress_message = progress_message
            db.commit()
            logger.info(f"Updated task {task_id} in database: status={status}")
        else:
            logger.warning(f"Task {task_id} not found in database")
    except Exception as e:
        logger.error(f"Failed to update task in database: {e}")
        db.rollback()
    finally:
        db.close()

@celery_app.task(bind=True, name="analyze_molecule_task")
def analyze_molecule_task(self, smiles: str):
    """
    Celery task wrapper for high-compute analysis (tier-2).
    """
    logger.info(f"Starting tier-2 analysis for SMILES: {smiles}")
    
    # Update database status to PROGRESS
    update_task_in_db(self.request.id, "PROGRESS", progress=0, progress_message="Starting analysis")
    
    def progress_callback(percent: int, message: str):
        self.update_state(
            state='PROGRESS',
            meta={
                'current': percent,
                'total': 100,
                'status': message
            }
        )
        # Update database with progress
        update_task_in_db(self.request.id, "PROGRESS", progress=percent, progress_message=message)
        logger.info(f"Progress: {percent}% - {message}")

    try:
        result = analyze_molecule_high_compute(smiles, progress_callback=progress_callback)
        logger.info("Tier-2 analysis completed successfully")
        
        # Update database with success
        update_task_in_db(self.request.id, "SUCCESS", result=result, progress=100)
        
        return result
    except Exception as e:
        logger.error(f"Tier-2 analysis failed: {e}")
        
        # Update database with failure
        update_task_in_db(self.request.id, "FAILURE", error=str(e))
        
        # Re-raise to let Celery handle failure state
        raise e

@celery_app.task(bind=True, name="predict_tier_3_task")
def predict_tier_3_task(self, smiles: str):
    """
    Celery task for tier-3 predictions using serverless GPU.
    """
    logger.info(f"Starting tier-3 prediction for SMILES: {smiles}")
    
    # Update database status to PROGRESS
    update_task_in_db(self.request.id, "PROGRESS", progress=10, progress_message="Sending request to serverless GPU")
    
    # Get serverless URL from environment variable
    serverless_url = os.getenv(
        "SERVERLESS_GPU_URL",
        "https://xxxx-serving-api.modal.run"
    )
    
    try:
        # Make request to serverless GPU with extended timeout
        with httpx.Client(timeout=60.0) as client:
            logger.info(f"Calling serverless API at {serverless_url}")
            update_task_in_db(self.request.id, "PROGRESS", progress=30, progress_message="Waiting for serverless GPU response")
            
            response = client.post(
                serverless_url,
                json={"smiles": smiles},
                headers={"Content-Type": "application/json"}
            )
            
            update_task_in_db(self.request.id, "PROGRESS", progress=70, progress_message="Processing response")
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Check if the serverless API returned an error
                if "error" in result:
                    error_msg = result["error"]
                    logger.error(f"Serverless API returned error: {error_msg}")
                    update_task_in_db(self.request.id, "FAILURE", error=error_msg)
                    raise Exception(error_msg)
                
                # Validate that we have the expected structure
                if "predictions" not in result:
                    error_msg = "Invalid response from serverless API: missing predictions"
                    logger.error(error_msg)
                    update_task_in_db(self.request.id, "FAILURE", error=error_msg)
                    raise Exception(error_msg)
                
                logger.info("Tier-3 prediction completed successfully")
                
                # SAVE TO LEADERBOARD (Tier3Analysis)
                try:
                    from models.tier3_analysis import Tier3Analysis
                    db = SessionLocal()
                    
                    # Log the result structure for debugging
                    logger.info(f"Tier 3 RAW RESULT: {result}")
                    
                    # Create analysis record
                    # Extract result from API response. 
                    # Assuming result structure: {"predictions": [...], "meta": ...}
                    # We store the whole JSON in the 'result' column
                    analysis_record = Tier3Analysis(
                        smiles=smiles,
                        result=result
                    )
                    db.add(analysis_record)
                    db.commit()
                    logger.info(f"Successfully saved Tier 3 analysis to leaderboard for SMILES: {smiles}")
                    db.close()
                except Exception as db_err:
                    logger.error(f"Failed to save Tier 3 analysis to leaderboard: {db_err}", exc_info=True)
                    # Don't fail the task if DB save fails, just log it
                
                update_task_in_db(self.request.id, "SUCCESS", result=result, progress=100, progress_message="Prediction complete")
                
                return result
            else:
                # Handle non-200 status codes from serverless API
                try:
                    error_detail = response.json().get("error", response.text)
                except Exception:
                    error_detail = response.text or f"HTTP {response.status_code}"
                
                error_msg = f"Serverless API error: {error_detail}"
                logger.error(error_msg)
                update_task_in_db(self.request.id, "FAILURE", error=error_msg)
                raise Exception(error_msg)
                
    except httpx.TimeoutException:
        error_msg = "Request to serverless GPU timed out"
        logger.error(error_msg)
        update_task_in_db(self.request.id, "FAILURE", error=error_msg)
        raise Exception(error_msg)
        
    except httpx.RequestError as e:
        error_msg = f"Failed to connect to serverless GPU: {str(e)}"
        logger.error(error_msg)
        update_task_in_db(self.request.id, "FAILURE", error=error_msg)
        raise Exception(error_msg)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tier-3 prediction failed: {error_msg}")
        update_task_in_db(self.request.id, "FAILURE", error=error_msg)
        raise e
