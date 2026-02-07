import os
from celery import Celery
from features.advanced_analysis import analyze_molecule_high_compute
import logging

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

@celery_app.task(bind=True, name="analyze_molecule_task")
def analyze_molecule_task(self, smiles: str):
    """
    Celery task wrapper for high-compute analysis.
    """
    logger.info(f"Starting analysis for SMILES: {smiles}")
    
    def progress_callback(percent: int, message: str):
        self.update_state(
            state='PROGRESS',
            meta={
                'current': percent,
                'total': 100,
                'status': message
            }
        )
        logger.info(f"Progress: {percent}% - {message}")

    try:
        result = analyze_molecule_high_compute(smiles, progress_callback=progress_callback)
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        # Re-raise to let Celery handle failure state
        raise e
