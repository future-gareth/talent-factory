"""
Feedback and usage metrics module for Talent Factory
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feedback router
feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])

# Pydantic models
class UsageMetrics(BaseModel):
    talent_id: str
    persona_id: Optional[str] = None
    usage_count: int
    success_count: int
    failure_count: int
    avg_response_time_ms: float
    last_used: str
    user_satisfaction: Optional[float] = None

class FeedbackData(BaseModel):
    talent_id: str
    persona_id: Optional[str] = None
    feedback_type: str  # "usage", "performance", "safety", "general"
    rating: Optional[int] = None  # 1-5 scale
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class FeedbackService:
    """Service for managing feedback and usage metrics"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.feedback_db = base_dir / "feedback.db"
        self.metrics_db = base_dir / "metrics.db"
        
        # Initialize databases
        self.init_databases()
        
        logger.info("FeedbackService initialized")
    
    def init_databases(self):
        """Initialize feedback and metrics databases"""
        # Feedback database
        import sqlite3
        conn = sqlite3.connect(self.feedback_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                talent_id TEXT NOT NULL,
                persona_id TEXT,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
        # Metrics database
        conn = sqlite3.connect(self.metrics_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                talent_id TEXT NOT NULL,
                persona_id TEXT,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_response_time_ms REAL DEFAULT 0.0,
                last_used TIMESTAMP,
                user_satisfaction REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def submit_feedback(self, feedback: FeedbackData) -> Dict[str, Any]:
        """Submit feedback for a talent"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.feedback_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (talent_id, persona_id, feedback_type, rating, comment, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                feedback.talent_id,
                feedback.persona_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.comment,
                json.dumps(feedback.metadata) if feedback.metadata else None
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Feedback submitted for talent {feedback.talent_id}: {feedback.feedback_type}")
            
            return {
                "feedback_id": feedback_id,
                "status": "submitted",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def update_usage_metrics(self, metrics: UsageMetrics) -> Dict[str, Any]:
        """Update usage metrics for a talent"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.metrics_db)
            cursor = conn.cursor()
            
            # Check if metrics exist for this talent
            cursor.execute('''
                SELECT id FROM usage_metrics WHERE talent_id = ? AND persona_id = ?
            ''', (metrics.talent_id, metrics.persona_id))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing metrics
                cursor.execute('''
                    UPDATE usage_metrics 
                    SET usage_count = ?, success_count = ?, failure_count = ?,
                        avg_response_time_ms = ?, last_used = ?, user_satisfaction = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE talent_id = ? AND persona_id = ?
                ''', (
                    metrics.usage_count,
                    metrics.success_count,
                    metrics.failure_count,
                    metrics.avg_response_time_ms,
                    metrics.last_used,
                    metrics.user_satisfaction,
                    metrics.talent_id,
                    metrics.persona_id
                ))
            else:
                # Insert new metrics
                cursor.execute('''
                    INSERT INTO usage_metrics 
                    (talent_id, persona_id, usage_count, success_count, failure_count,
                     avg_response_time_ms, last_used, user_satisfaction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.talent_id,
                    metrics.persona_id,
                    metrics.usage_count,
                    metrics.success_count,
                    metrics.failure_count,
                    metrics.avg_response_time_ms,
                    metrics.last_used,
                    metrics.user_satisfaction
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Usage metrics updated for talent {metrics.talent_id}")
            
            return {
                "status": "updated",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update usage metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_talent_feedback(self, talent_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a specific talent"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.feedback_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, talent_id, persona_id, feedback_type, rating, comment, 
                       metadata, created_at
                FROM feedback 
                WHERE talent_id = ?
                ORDER BY created_at DESC
            ''', (talent_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for row in results:
                feedback_list.append({
                    "id": row[0],
                    "talent_id": row[1],
                    "persona_id": row[2],
                    "feedback_type": row[3],
                    "rating": row[4],
                    "comment": row[5],
                    "metadata": json.loads(row[6]) if row[6] else None,
                    "created_at": row[7]
                })
            
            return feedback_list
            
        except Exception as e:
            logger.error(f"Failed to get talent feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_talent_metrics(self, talent_id: str) -> Dict[str, Any]:
        """Get usage metrics for a specific talent"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.metrics_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT usage_count, success_count, failure_count, avg_response_time_ms,
                       last_used, user_satisfaction, updated_at
                FROM usage_metrics 
                WHERE talent_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            ''', (talent_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "talent_id": talent_id,
                    "usage_count": result[0],
                    "success_count": result[1],
                    "failure_count": result[2],
                    "avg_response_time_ms": result[3],
                    "last_used": result[4],
                    "user_satisfaction": result[5],
                    "updated_at": result[6]
                }
            else:
                return {
                    "talent_id": talent_id,
                    "usage_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_response_time_ms": 0.0,
                    "last_used": None,
                    "user_satisfaction": None,
                    "updated_at": None
                }
            
        except Exception as e:
            logger.error(f"Failed to get talent metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get usage metrics for all talents"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.metrics_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT talent_id, persona_id, usage_count, success_count, failure_count,
                       avg_response_time_ms, last_used, user_satisfaction, updated_at
                FROM usage_metrics 
                ORDER BY updated_at DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            metrics_list = []
            for row in results:
                metrics_list.append({
                    "talent_id": row[0],
                    "persona_id": row[1],
                    "usage_count": row[2],
                    "success_count": row[3],
                    "failure_count": row[4],
                    "avg_response_time_ms": row[5],
                    "last_used": row[6],
                    "user_satisfaction": row[7],
                    "updated_at": row[8]
                })
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Failed to get all metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_feedback_summary(self, talent_id: str) -> Dict[str, Any]:
        """Get feedback summary for a talent"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.feedback_db)
            cursor = conn.cursor()
            
            # Get feedback counts by type
            cursor.execute('''
                SELECT feedback_type, COUNT(*) as count, AVG(rating) as avg_rating
                FROM feedback 
                WHERE talent_id = ?
                GROUP BY feedback_type
            ''', (talent_id,))
            
            type_counts = cursor.fetchall()
            
            # Get overall stats
            cursor.execute('''
                SELECT COUNT(*) as total_feedback, AVG(rating) as avg_rating
                FROM feedback 
                WHERE talent_id = ?
            ''', (talent_id,))
            
            overall_stats = cursor.fetchone()
            conn.close()
            
            summary = {
                "talent_id": talent_id,
                "total_feedback": overall_stats[0] if overall_stats else 0,
                "average_rating": overall_stats[1] if overall_stats and overall_stats[1] else None,
                "feedback_by_type": {}
            }
            
            for row in type_counts:
                summary["feedback_by_type"][row[0]] = {
                    "count": row[1],
                    "average_rating": row[2] if row[2] else None
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get feedback summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize feedback service
feedback_service = None

def init_feedback_service(base_dir: Path):
    """Initialize feedback service"""
    global feedback_service
    feedback_service = FeedbackService(base_dir)

# Feedback endpoints

@feedback_router.post("/submit")
async def submit_feedback(feedback: FeedbackData):
    """Submit feedback for a talent"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        result = feedback_service.submit_feedback(feedback)
        return result
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.post("/metrics")
async def update_metrics(metrics: UsageMetrics):
    """Update usage metrics for a talent"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        result = feedback_service.update_usage_metrics(metrics)
        return result
    except Exception as e:
        logger.error(f"Failed to update metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.get("/talent/{talent_id}")
async def get_talent_feedback(talent_id: str):
    """Get feedback for a specific talent"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        feedback = feedback_service.get_talent_feedback(talent_id)
        return {"feedback": feedback}
    except Exception as e:
        logger.error(f"Failed to get talent feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.get("/talent/{talent_id}/metrics")
async def get_talent_metrics(talent_id: str):
    """Get usage metrics for a specific talent"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        metrics = feedback_service.get_talent_metrics(talent_id)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get talent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.get("/talent/{talent_id}/summary")
async def get_feedback_summary(talent_id: str):
    """Get feedback summary for a talent"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        summary = feedback_service.get_feedback_summary(talent_id)
        return summary
    except Exception as e:
        logger.error(f"Failed to get feedback summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@feedback_router.get("/metrics/all")
async def get_all_metrics():
    """Get usage metrics for all talents"""
    if not feedback_service:
        raise HTTPException(status_code=500, detail="Feedback service not initialized")
    
    try:
        metrics = feedback_service.get_all_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to get all metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
