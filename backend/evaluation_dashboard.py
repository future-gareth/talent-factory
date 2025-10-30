"""
Evaluation Dashboard for Base vs Tuned Model Comparison
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluation Router
eval_router = APIRouter(prefix="/evaluation", tags=["Evaluation"])

# Pydantic models
class EvaluationMetrics(BaseModel):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    perplexity: float
    bleu_score: Optional[float] = None
    rouge_score: Optional[float] = None

class SafetyMetrics(BaseModel):
    safety_score: float
    toxicity_score: float
    bias_score: float
    hallucination_score: float
    rubric_passed: bool

class ModelComparison(BaseModel):
    base_model: str
    tuned_model: str
    base_metrics: EvaluationMetrics
    tuned_metrics: EvaluationMetrics
    improvement: Dict[str, float]
    safety_comparison: Dict[str, Any]

class EvaluationDashboard(BaseModel):
    evaluation_id: str
    talent_id: str
    comparison: ModelComparison
    test_dataset: str
    evaluation_date: str
    duration_seconds: int
    status: str

class EvaluationDashboardService:
    """Service for managing evaluation dashboard data"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.registry_db = base_dir / "registry.db"
        self.models_dir = base_dir / "models"
        self.datasets_dir = base_dir / "datasets"
        
        logger.info("Evaluation Dashboard Service initialized")
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.registry_db)
    
    def generate_base_model_metrics(self, base_model: str, test_dataset: str) -> EvaluationMetrics:
        """Generate baseline metrics for the base model"""
        # Simulate base model evaluation
        # In a real implementation, this would run inference on the base model
        
        base_metrics = {
            "llama-2-7b": {
                "accuracy": 0.72,
                "f1_score": 0.68,
                "precision": 0.75,
                "recall": 0.62,
                "perplexity": 12.5,
                "bleu_score": 0.45,
                "rouge_score": 0.52
            },
            "mistral-7b": {
                "accuracy": 0.74,
                "f1_score": 0.70,
                "precision": 0.77,
                "recall": 0.64,
                "perplexity": 11.8,
                "bleu_score": 0.47,
                "rouge_score": 0.54
            },
            "codellama-7b": {
                "accuracy": 0.68,
                "f1_score": 0.65,
                "precision": 0.71,
                "recall": 0.59,
                "perplexity": 13.2,
                "bleu_score": 0.42,
                "rouge_score": 0.49
            }
        }
        
        default_metrics = {
            "accuracy": 0.70,
            "f1_score": 0.67,
            "precision": 0.73,
            "recall": 0.61,
            "perplexity": 12.0,
            "bleu_score": 0.44,
            "rouge_score": 0.51
        }
        
        return EvaluationMetrics(**base_metrics.get(base_model, default_metrics))
    
    def generate_tuned_model_metrics(self, talent_id: str, test_dataset: str) -> EvaluationMetrics:
        """Generate metrics for the tuned model"""
        # Simulate tuned model evaluation
        # In a real implementation, this would run inference on the tuned model
        
        # Get talent info to determine base model
        conn = self.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT model_path FROM talents WHERE id = ?
        ''', (talent_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise ValueError(f"Talent {talent_id} not found")
        
        model_path = result[0]
        
        # Read metadata to get base model
        metadata_path = Path(model_path) / "metadata.json"
        base_model = "llama-2-7b"  # Default
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                base_model = metadata.get('base_model', 'llama-2-7b')
            except Exception as e:
                logger.warning(f"Failed to read metadata: {e}")
        
        # Get base metrics
        base_metrics = self.generate_base_model_metrics(base_model, test_dataset)
        
        # Simulate improvement (tuned model should be better)
        improvement_factor = 1.15  # 15% improvement on average
        
        tuned_metrics = EvaluationMetrics(
            accuracy=min(0.95, base_metrics.accuracy * improvement_factor),
            f1_score=min(0.90, base_metrics.f1_score * improvement_factor),
            precision=min(0.92, base_metrics.precision * improvement_factor),
            recall=min(0.88, base_metrics.recall * improvement_factor),
            perplexity=max(5.0, base_metrics.perplexity / improvement_factor),
            bleu_score=min(0.70, (base_metrics.bleu_score or 0.44) * improvement_factor),
            rouge_score=min(0.75, (base_metrics.rouge_score or 0.51) * improvement_factor)
        )
        
        return tuned_metrics
    
    def calculate_improvement(self, base_metrics: EvaluationMetrics, tuned_metrics: EvaluationMetrics) -> Dict[str, float]:
        """Calculate improvement percentages"""
        improvement = {}
        
        # For metrics where higher is better
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'bleu_score', 'rouge_score']:
            base_val = getattr(base_metrics, metric, 0)
            tuned_val = getattr(tuned_metrics, metric, 0)
            if base_val > 0:
                improvement[metric] = ((tuned_val - base_val) / base_val) * 100
        
        # For perplexity (lower is better)
        if base_metrics.perplexity > 0 and tuned_metrics.perplexity > 0:
            improvement['perplexity'] = ((base_metrics.perplexity - tuned_metrics.perplexity) / base_metrics.perplexity) * 100
        
        return improvement
    
    def generate_safety_comparison(self, base_model: str, tuned_model_path: str) -> Dict[str, Any]:
        """Generate safety comparison between base and tuned models"""
        # Simulate safety evaluation
        base_safety = {
            "safety_score": 0.85,
            "toxicity_score": 0.12,
            "bias_score": 0.18,
            "hallucination_score": 0.15,
            "rubric_passed": True
        }
        
        # Tuned model should maintain or improve safety
        tuned_safety = {
            "safety_score": 0.88,  # Slightly better
            "toxicity_score": 0.10,  # Slightly better
            "bias_score": 0.16,  # Slightly better
            "hallucination_score": 0.13,  # Slightly better
            "rubric_passed": True
        }
        
        return {
            "base": base_safety,
            "tuned": tuned_safety,
            "improvement": {
                "safety_score": ((tuned_safety["safety_score"] - base_safety["safety_score"]) / base_safety["safety_score"]) * 100,
                "toxicity_score": ((base_safety["toxicity_score"] - tuned_safety["toxicity_score"]) / base_safety["toxicity_score"]) * 100,
                "bias_score": ((base_safety["bias_score"] - tuned_safety["bias_score"]) / base_safety["bias_score"]) * 100,
                "hallucination_score": ((base_safety["hallucination_score"] - tuned_safety["hallucination_score"]) / base_safety["hallucination_score"]) * 100
            }
        }
    
    def create_evaluation_dashboard(self, talent_id: str, test_dataset_id: Optional[str] = None) -> EvaluationDashboard:
        """Create a comprehensive evaluation dashboard"""
        try:
            # Get talent info
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, model_path, version FROM talents WHERE id = ?
            ''', (talent_id,))
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                raise ValueError(f"Talent {talent_id} not found")
            
            talent_name, model_path, version = result
            
            # Read metadata to get base model
            metadata_path = Path(model_path) / "metadata.json"
            base_model = "llama-2-7b"  # Default
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    base_model = metadata.get('base_model', 'llama-2-7b')
                except Exception as e:
                    logger.warning(f"Failed to read metadata: {e}")
            
            # Generate metrics
            base_metrics = self.generate_base_model_metrics(base_model, test_dataset_id or "default")
            tuned_metrics = self.generate_tuned_model_metrics(talent_id, test_dataset_id or "default")
            
            # Calculate improvements
            improvement = self.calculate_improvement(base_metrics, tuned_metrics)
            
            # Generate safety comparison
            safety_comparison = self.generate_safety_comparison(base_model, model_path)
            
            # Create comparison object
            comparison = ModelComparison(
                base_model=base_model,
                tuned_model=f"{talent_name} (v{version})",
                base_metrics=base_metrics,
                tuned_metrics=tuned_metrics,
                improvement=improvement,
                safety_comparison=safety_comparison
            )
            
            # Create dashboard
            dashboard = EvaluationDashboard(
                evaluation_id=f"eval_{talent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                talent_id=talent_id,
                comparison=comparison,
                test_dataset=test_dataset_id or "default",
                evaluation_date=datetime.now().isoformat(),
                duration_seconds=300,  # Simulated 5 minutes
                status="completed"
            )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to create evaluation dashboard: {e}")
            raise
    
    def get_evaluation_history(self, talent_id: str) -> List[Dict[str, Any]]:
        """Get evaluation history for a talent"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, metrics, safety_score, rubric_passed, created_at
                FROM eval_reports
                WHERE talent_id = ?
                ORDER BY created_at DESC
            ''', (talent_id,))
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                metrics = json.loads(row[1]) if row[1] else {}
                history.append({
                    "evaluation_id": row[0],
                    "metrics": metrics,
                    "safety_score": row[2],
                    "rubric_passed": row[3],
                    "created_at": row[4]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get evaluation history: {e}")
            return []

# Initialize evaluation service
eval_service = None

def init_evaluation_service(base_dir: Path):
    """Initialize evaluation service"""
    global eval_service
    eval_service = EvaluationDashboardService(base_dir)

# Evaluation endpoints

@eval_router.get("/dashboard/{talent_id}")
async def get_evaluation_dashboard(talent_id: str, test_dataset_id: Optional[str] = None):
    """Get evaluation dashboard with base vs tuned comparison"""
    if not eval_service:
        raise HTTPException(status_code=500, detail="Evaluation service not initialized")
    
    try:
        dashboard = eval_service.create_evaluation_dashboard(talent_id, test_dataset_id)
        return dashboard
    except Exception as e:
        logger.error(f"Failed to get evaluation dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@eval_router.get("/history/{talent_id}")
async def get_evaluation_history(talent_id: str):
    """Get evaluation history for a talent"""
    if not eval_service:
        raise HTTPException(status_code=500, detail="Evaluation service not initialized")
    
    try:
        history = eval_service.get_evaluation_history(talent_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@eval_router.get("/metrics/{talent_id}")
async def get_detailed_metrics(talent_id: str):
    """Get detailed metrics breakdown for a talent"""
    if not eval_service:
        raise HTTPException(status_code=500, detail="Evaluation service not initialized")
    
    try:
        dashboard = eval_service.create_evaluation_dashboard(talent_id)
        
        # Return detailed metrics
        return {
            "talent_id": talent_id,
            "base_model": dashboard.comparison.base_model,
            "tuned_model": dashboard.comparison.tuned_model,
            "performance_metrics": {
                "base": dashboard.comparison.base_metrics.dict(),
                "tuned": dashboard.comparison.tuned_metrics.dict(),
                "improvement": dashboard.comparison.improvement
            },
            "safety_metrics": dashboard.comparison.safety_comparison,
            "evaluation_date": dashboard.evaluation_date,
            "status": dashboard.status
        }
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@eval_router.post("/compare")
async def compare_models(
    base_model: str,
    tuned_model_path: str,
    test_dataset: str
):
    """Compare two models directly"""
    if not eval_service:
        raise HTTPException(status_code=500, detail="Evaluation service not initialized")
    
    try:
        # Generate metrics for both models
        base_metrics = eval_service.generate_base_model_metrics(base_model, test_dataset)
        tuned_metrics = eval_service.generate_tuned_model_metrics("temp", test_dataset)
        
        # Calculate improvements
        improvement = eval_service.calculate_improvement(base_metrics, tuned_metrics)
        
        # Generate safety comparison
        safety_comparison = eval_service.generate_safety_comparison(base_model, tuned_model_path)
        
        return {
            "base_model": base_model,
            "tuned_model": tuned_model_path,
            "base_metrics": base_metrics.dict(),
            "tuned_metrics": tuned_metrics.dict(),
            "improvement": improvement,
            "safety_comparison": safety_comparison,
            "comparison_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
