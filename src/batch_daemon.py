"""
Continuous batch prediction daemon that runs periodically.
"""
import time
import logging
import sys
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import schedule

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.batch import BatchPredictor, main as run_batch_once
from src.alerts import alert_manager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchDaemon:
    def __init__(self, interval_minutes: Optional[int] = None):
        self.interval_minutes = interval_minutes or settings.BATCH_INTERVAL_MIN
        self.predictor = BatchPredictor()
        self.running = True
        self.stats = {
            "total_runs": 0,
            "total_processed": 0,
            "total_errors": 0,
            "last_run_time": None,
            "last_run_duration": None,
            "last_run_processed": 0,
            "last_run_skipped": 0,
        }
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def _run_job(self):
        """Execute one batch processing job"""
        start_time = datetime.now()
        logger.info(f"Starting batch job at {start_time.isoformat()}")
        
        try:
            result = run_batch_once()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.stats["total_runs"] += 1
            self.stats["last_run_time"] = start_time.isoformat()
            self.stats["last_run_duration"] = duration
            self.stats["last_run_processed"] = result.get("processed", 0)
            self.stats["last_run_skipped"] = result.get("skipped", 0)
            
            if result.get("processed", 0) > 0:
                self.stats["total_processed"] += result["processed"]
                
            if result.get("error"):
                self.stats["total_errors"] += 1
                logger.error(f"Batch job failed: {result['error']}")
            else:
                logger.info(f"Batch job completed: {result['processed']} processed, {result['skipped']} skipped, {duration:.2f}s")
                
        except Exception as e:
            self.stats["total_errors"] += 1
            logger.error(f"Batch job exception: {e}", exc_info=True)
            alert_manager.send(
                "Batch Daemon Exception",
                f"Error: {str(e)}\nStats: {self.stats}",
                severity="ERROR"
            )
    
    def run_forever(self):
        """Run the daemon continuously"""
        logger.info(f"Starting batch daemon - interval: {self.interval_minutes} minutes")
        logger.info(f"Batch limit: {settings.BATCH_LIMIT}")
        logger.info(f"Drift monitoring: {'enabled' if settings.ENABLE_DRIFT_MONITORING else 'disabled'}")
        
        self._run_job()
        
        schedule.every(self.interval_minutes).minutes.do(self._run_job)
        
        while self.running:
            schedule.run_pending()
            time.sleep(30)
        
        logger.info(f"Batch daemon stopped. Final stats: {self.stats}")
        
    def run_once(self):
        """Run a single batch job and exit"""
        self._run_job()
        return self.stats

def main():
    """Entry point for batch daemon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch prediction daemon")
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Run once and exit instead of running continuously"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        help=f"Interval between runs in minutes (default: {settings.BATCH_INTERVAL_MIN})"
    )
    
    args = parser.parse_args()
    
    daemon = BatchDaemon(interval_minutes=args.interval)
    
    if args.once:
        daemon.run_once()
    else:
        daemon.run_forever()

if __name__ == "__main__":
    main()