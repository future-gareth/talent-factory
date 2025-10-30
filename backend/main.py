from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import sqlite3
import logging
import asyncio
import subprocess
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from datetime import datetime
import uuid
import shutil
import re
from pathlib import Path
import mimetypes
import hashlib
import requests
from urllib.parse import quote

# Import WebSocket handler
from websocket_handler import websocket_endpoint, broadcast_training_progress, broadcast_training_completion, broadcast_talent_published

# Import evaluation dashboard
from evaluation_dashboard import eval_router, init_evaluation_service

# Import authentication
from auth import auth_router, init_auth_service, require_auth, User

# Import feedback
from feedback import feedback_router, init_feedback_service

# Import MCP catalogue
from mcp_catalogue import mcp_router, init_mcp_catalogue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import training engine
try:
    from training_engine import TrainingEngine
    TRAINING_ENGINE_AVAILABLE = True
except ImportError:
    TRAINING_ENGINE_AVAILABLE = False
    logger.warning("Training engine not available - using basic training implementation")

# Import Apple Silicon support
try:
    from platform_detector import get_backend_info, print_backend_banner, detect_apple_silicon
    from training_router import get_training_router
    from thermal_manager import get_thermal_manager, HeatProfile
    from dependency_manager import ensure_dependencies, get_dependency_manager
    APPLE_SILICON_AVAILABLE = True
except ImportError:
    APPLE_SILICON_AVAILABLE = False
    logger.warning("Apple Silicon support not available")

app = FastAPI(title="Talent Factory", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"
REGISTRY_DB = BASE_DIR / "registry.db"

# Ensure directories exist
for directory in [MODELS_DIR, DATASETS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect(REGISTRY_DB)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS talents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            model_path TEXT NOT NULL,
            version TEXT,
            metrics TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source TEXT,
            rows INTEGER,
            pii_masked BOOLEAN DEFAULT FALSE,
            split TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS train_runs (
            id TEXT PRIMARY KEY,
            talent_id TEXT,
            base_model TEXT,
            params TEXT,
            outcome TEXT,
            duration INTEGER,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (talent_id) REFERENCES talents (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_reports (
            id TEXT PRIMARY KEY,
            talent_id TEXT,
            metrics TEXT,
            safety_score REAL,
            rubric_passed BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (talent_id) REFERENCES talents (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Initialize evaluation service
init_evaluation_service(BASE_DIR)

# Initialize authentication service
init_auth_service(str(BASE_DIR / "auth.json"))

# Initialize feedback service
init_feedback_service(BASE_DIR)

# Initialize MCP catalogue
init_mcp_catalogue(BASE_DIR)

# Include routers
app.include_router(eval_router)
app.include_router(auth_router)
app.include_router(feedback_router)
app.include_router(mcp_router)

# Pydantic models
class Talent(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    model_path: str
    version: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class Dataset(BaseModel):
    id: str
    name: str
    source: Optional[str] = None
    rows: Optional[int] = None
    pii_masked: bool = False
    split: Optional[str] = None
    created_at: Optional[str] = None

class TrainRun(BaseModel):
    id: str
    talent_id: Optional[str] = None
    base_model: str
    params: Dict[str, Any]
    outcome: Optional[str] = None
    duration: Optional[int] = None
    status: str = "pending"
    created_at: Optional[str] = None

class EvalReport(BaseModel):
    id: str
    talent_id: str
    metrics: Dict[str, Any]
    safety_score: float
    rubric_passed: bool
    created_at: Optional[str] = None

class EnvProfile(BaseModel):
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    cpu_cores: int
    ram_gb: float
    ready: bool

class TrainingRequest(BaseModel):
    base_model: str
    dataset_ids: List[str]  # Support multiple datasets
    params: Dict[str, Any]
    outcome_preference: str = "balanced"  # speed, balanced, quality

class EvaluationRequest(BaseModel):
    talent_id: str
    test_dataset_id: Optional[str] = None

# Global state for active training runs
active_training_runs: Dict[str, Dict[str, Any]] = {}

def get_db_connection():
    return sqlite3.connect(REGISTRY_DB)

def log_action(action: str, details: Dict[str, Any] = None):
    """Log actions for audit trail"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "details": details or {}
    }
    
    log_file = LOGS_DIR / f"audit_{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def detect_pii(text: str) -> bool:
    """Simple PII detection - can be enhanced with more sophisticated models"""
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
    ]
    
    for pattern in pii_patterns:
        if re.search(pattern, text):
            return True
    return False

def mask_pii(text: str) -> str:
    """Mask PII in text"""
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', text)
    
    # Replace phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', text)
    
    # Replace SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_MASKED]', text)
    
    # Replace credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_MASKED]', text)
    
    return text

def extract_text_from_file(file_path: Path, mime_type: str) -> str:
    """Extract text content from various file formats"""
    try:
        if mime_type == 'text/plain':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif mime_type == 'text/markdown':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif mime_type == 'application/json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        elif mime_type == 'text/csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif mime_type == 'text/html':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple HTML tag removal
                content = re.sub(r'<[^>]+>', '', content)
                return content
        elif mime_type == 'application/pdf':
            # For PDF, we'll use a simple approach - in production, use PyPDF2 or similar
            return f"[PDF_CONTENT_FROM_{file_path.name}]"
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # For DOCX, we'll use a simple approach - in production, use python-docx
            return f"[DOCX_CONTENT_FROM_{file_path.name}]"
        else:
            # Fallback for unknown types
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return f"[ERROR_EXTRACTING_CONTENT_FROM_{file_path.name}]"

def convert_to_trainable_format(text_content: str, filename: str) -> List[Dict[str, str]]:
    """Convert unstructured text into trainable conversation format"""
    # Split content into chunks
    chunks = text_content.split('\n\n')  # Split by double newlines
    
    training_data = []
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 10:  # Skip very short chunks
            continue
            
        # Create training examples
        if len(chunk) > 500:
            # For long chunks, create Q&A pairs
            training_data.append({
                "instruction": f"Summarize the following content from {filename}:",
                "input": chunk[:500] + "...",
                "output": f"This content is from {filename} and contains detailed information."
            })
        else:
            # For shorter chunks, create instruction-following examples
            training_data.append({
                "instruction": f"Process this information from {filename}:",
                "input": chunk,
                "output": f"I understand this information from {filename}. It contains relevant details."
            })
    
    return training_data

def search_huggingface_datasets(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search Hugging Face datasets using their API"""
    try:
        # Use Hugging Face's search API
        url = f"https://huggingface.co/api/datasets?search={quote(query)}&limit={limit}&sort=downloads"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        datasets = response.json()
        results = []
        
        for dataset in datasets:
            # Calculate relevance score based on downloads, likes, and query match
            downloads = dataset.get('downloads', 0)
            likes = dataset.get('likes', 0)
            gated = dataset.get('gated', False)
            
            # Skip gated datasets for now
            if gated:
                continue
                
            # Calculate relevance score
            relevance_score = (downloads * 0.7 + likes * 0.3) / 1000
            
            results.append({
                "id": dataset.get('id', ''),
                "name": dataset.get('id', '').split('/')[-1] if '/' in dataset.get('id', '') else dataset.get('id', ''),
                "author": dataset.get('id', '').split('/')[0] if '/' in dataset.get('id', '') else 'unknown',
                "description": dataset.get('description', 'No description available')[:200] + '...',
                "downloads": downloads,
                "likes": likes,
                "relevance_score": relevance_score,
                "tags": dataset.get('tags', [])[:5],  # Limit tags
                "size": dataset.get('downloads', 0),  # Use downloads as proxy for size
                "url": f"https://huggingface.co/datasets/{dataset.get('id', '')}"
            })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]
        
    except Exception as e:
        logger.error(f"Error searching Hugging Face datasets: {e}")
        return []

def get_dataset_relevance_score(dataset: Dict[str, Any], query: str) -> float:
    """Calculate relevance score for a dataset based on query"""
    query_lower = query.lower()
    score = 0.0
    
    # Check name match
    name = dataset.get('name', '').lower()
    if query_lower in name:
        score += 10.0
    
    # Check description match
    description = dataset.get('description', '').lower()
    if query_lower in description:
        score += 5.0
    
    # Check tags match
    tags = [tag.lower() for tag in dataset.get('tags', [])]
    for tag in tags:
        if query_lower in tag:
            score += 3.0
    
    # Add popularity bonus
    downloads = dataset.get('downloads', 0)
    likes = dataset.get('likes', 0)
    popularity = (downloads * 0.7 + likes * 0.3) / 1000
    score += min(popularity, 5.0)  # Cap popularity bonus
    
    return score

def combine_datasets(dataset_ids: List[str]) -> str:
    """Combine multiple datasets into one training dataset"""
    combined_data = []
    
    for dataset_id in dataset_ids:
        # Get dataset info from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT source FROM datasets WHERE id = ?', (dataset_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            dataset_path = Path(result[0])
            if dataset_path.exists():
                try:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            combined_data.extend(data)
                        else:
                            combined_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading dataset {dataset_id}: {e}")
    
    # Save combined dataset
    combined_id = str(uuid.uuid4())
    combined_path = DATASETS_DIR / f"{combined_id}_combined_training_data.json"
    
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    logger.info(f"Combined {len(dataset_ids)} datasets into {len(combined_data)} training examples")
    return str(combined_path)

def get_environment_profile() -> EnvProfile:
    """Get hardware environment profile"""
    try:
        # Get GPU info if available
        gpu_name = None
        vram_gb = None
        
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_name = gpu.name
                vram_gb = gpu.memoryTotal / 1024  # Convert MB to GB
        
        # Get CPU and RAM info
        cpu_cores = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Determine if ready for training
        ready = True
        if vram_gb and vram_gb < 4:  # Need at least 4GB VRAM
            ready = False
        
        return EnvProfile(
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            ready=ready
        )
    except Exception as e:
        logger.error(f"Error getting environment profile: {e}")
        return EnvProfile(
            gpu_name=None,
            vram_gb=None,
            cpu_cores=psutil.cpu_count(),
            ram_gb=psutil.virtual_memory().total / (1024**3),
            ready=False
        )

def get_compatible_models(env_profile: EnvProfile) -> List[Dict[str, Any]]:
    """Get list of models compatible with current hardware"""
    models = []
    
    # Add Apple Silicon models if available
    if APPLE_SILICON_AVAILABLE:
        try:
            backend_info = get_backend_info()
            if backend_info["system"]["is_apple_silicon"] and backend_info["system"]["mlx_available"]:
                models.extend([
                    {
                        "id": "mlx-mistral-7b",
                        "name": "Mistral 7B (MLX)",
                        "size_gb": 4.0,
                        "min_vram_gb": 0,  # Uses unified memory
                        "description": "Mistral 7B optimized for Apple Silicon",
                        "category": "conversational",
                        "backend": "mlx"
                    },
                    {
                        "id": "mlx-llama-7b",
                        "name": "Llama 7B (MLX)",
                        "size_gb": 4.0,
                        "min_vram_gb": 0,  # Uses unified memory
                        "description": "Llama 7B optimized for Apple Silicon",
                        "category": "conversational",
                        "backend": "mlx"
                    }
                ])
                logger.info("Added Apple Silicon MLX models")
        except Exception as e:
            logger.warning(f"Failed to add Apple Silicon models: {e}")
    
    # Add Persona Foundry base models (for creating personas, not specific personas)
    models.extend([
        {
            "id": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion 1.5",
            "size_gb": 4.0,
            "min_vram_gb": 4,
            "description": "Base model for creating avatar personas and character designs",
            "category": "persona-foundry",
            "backend": "cuda/mps",
            "model_type": "image-generation",
            "sdx_version": "1.5",
            "tags": ["base-model", "avatar", "character", "persona"]
        },
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL",
            "size_gb": 7.0,
            "min_vram_gb": 8,
            "description": "Advanced base model for high-quality persona creation",
            "category": "persona-foundry",
            "backend": "cuda/mps",
            "model_type": "image-generation",
            "sdx_version": "xl",
            "tags": ["base-model", "avatar", "character", "persona", "high-quality"]
        },
        {
            "id": "microsoft/DialoGPT-medium",
            "name": "DialoGPT Medium (Persona)",
            "size_gb": 1.5,
            "min_vram_gb": 2,
            "description": "Base conversational model for persona personality training",
            "category": "persona-foundry",
            "backend": "cuda/cpu",
            "model_type": "conversational",
            "tags": ["base-model", "conversation", "personality", "persona"]
        },
        {
            "id": "microsoft/DialoGPT-large",
            "name": "DialoGPT Large (Persona)",
            "size_gb": 3.0,
            "min_vram_gb": 4,
            "description": "Advanced conversational model for complex persona personalities",
            "category": "persona-foundry",
            "backend": "cuda/cpu",
            "model_type": "conversational",
            "tags": ["base-model", "conversation", "personality", "persona", "advanced"]
        }
    ])
    
    logger.info(f"Added Persona Foundry base models")
    
    # Add standard text models (non-persona)
    models.extend([
        {
            "id": "distilbert-base-uncased",
            "name": "DistilBERT",
            "size_gb": 0.5,
            "min_vram_gb": 1,
            "description": "Hugging Face's DistilBERT language model",
            "category": "language",
            "backend": "cuda/cpu"
        },
        {
            "id": "gpt2",
            "name": "GPT-2",
            "size_gb": 0.5,
            "min_vram_gb": 1,
            "description": "OpenAI's GPT-2 text generation model",
            "category": "text-generation",
            "backend": "cuda/cpu"
        }
    ])
    
    logger.info(f"Environment: vram_gb={env_profile.vram_gb}, ram_gb={env_profile.ram_gb}")
    logger.info(f"Returning {len(models)} models")
    
    return models

async def run_training(talent_id: str, base_model: str, dataset_path: str, params: Dict[str, Any]):
    """Run model fine-tuning in background"""
    try:
        # Update training status
        active_training_runs[talent_id] = {
            "status": "running",
            "progress": 0,
            "started_at": datetime.now().isoformat()
        }
        
        # Broadcast initial status
        await broadcast_training_progress(talent_id, 0, "starting", {"base_model": base_model})
        
        if TRAINING_ENGINE_AVAILABLE or APPLE_SILICON_AVAILABLE:
            # Use training router for backend selection
            logger.info(f"Starting training for {talent_id}")
            
            # Apply thermal management
            if APPLE_SILICON_AVAILABLE:
                thermal_manager = get_thermal_manager()
                thermal_settings = thermal_manager.apply_thermal_profile(HeatProfile.QUIET)
                logger.info(f"Applied thermal settings: {thermal_settings}")
            
            # Initialize training router
            if APPLE_SILICON_AVAILABLE:
                router = get_training_router(Path(__file__).parent)
                router.print_banner()
                
                # Progress callback
                async def progress_callback(progress: int, status: str):
                    active_training_runs[talent_id]["progress"] = progress
                    active_training_runs[talent_id]["status"] = status
                    await broadcast_training_progress(talent_id, progress, status, {"message": f"Progress: {progress}%"})
                
                # Run training with router
                result = await router.train_model(
                    model_id=base_model,
                    dataset_path=dataset_path,
                    output_dir=str(Path(__file__).parent / "output" / talent_id),
                    num_epochs=params.get("num_epochs", 3),
                    learning_rate=params.get("learning_rate", 5e-5),
                    batch_size=params.get("batch_size", 1),
                    gradient_accumulation_steps=params.get("gradient_accumulation_steps", 8),
                    progress_callback=progress_callback
                )
            else:
                # Fallback to original training engine
                engine = TrainingEngine(base_dir=Path(__file__).parent)
                logger.info(f"Training engine initialized successfully")
                
                # Progress callback
                async def progress_callback(progress: int, status: str):
                    active_training_runs[talent_id]["progress"] = progress
                    active_training_runs[talent_id]["status"] = status
                    await broadcast_training_progress(talent_id, progress, status, {"message": f"Progress: {progress}%"})
                
                # Run actual training
                result = await engine.train_model(
                    model_id=base_model,
                    dataset_path=dataset_path,
                    talent_name=f"talent_{talent_id}",
                    outcome_preference=params.get("outcome_preference", "balanced"),
                    progress_callback=progress_callback
                )
            
            # Mark as completed
            active_training_runs[talent_id]["status"] = "completed"
            active_training_runs[talent_id]["completed_at"] = datetime.now().isoformat()
            active_training_runs[talent_id]["result"] = result
            
            # Broadcast completion
            await broadcast_training_completion(talent_id, True, result)
            
        else:
            # Basic training implementation (without heavy ML dependencies)
            logger.info("Using basic training implementation")
            await broadcast_training_progress(talent_id, 0, "initializing", {"message": "Loading model and data"})
            
            # Load and validate dataset
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                
                if not isinstance(training_data, list) or len(training_data) == 0:
                    raise ValueError("Invalid or empty training dataset")
                
                await broadcast_training_progress(talent_id, 20, "validating", {"message": f"Validated {len(training_data)} training examples"})
                
                # Simulate training phases with realistic timing
                phases = [
                    (30, "preprocessing", "Preprocessing training data"),
                    (50, "training", "Fine-tuning model parameters"),
                    (70, "optimizing", "Optimizing model performance"),
                    (85, "evaluating", "Running model evaluation"),
                    (95, "finalizing", "Finalizing trained model")
                ]
                
                for progress, status, message in phases:
                    await asyncio.sleep(2)  # Simulate realistic training time
                    active_training_runs[talent_id]["progress"] = progress
                    active_training_runs[talent_id]["status"] = status
                    await broadcast_training_progress(talent_id, progress, status, {"message": message})
                
                # Create a basic model output (in real implementation, this would be the trained model)
                model_output_path = DATASETS_DIR / f"{talent_id}_trained_model.json"
                model_metadata = {
                    "talent_id": talent_id,
                    "base_model": base_model,
                    "training_examples": len(training_data),
                    "trained_at": datetime.now().isoformat(),
                    "parameters": params,
                    "model_path": str(model_output_path)
                }
                
                with open(model_output_path, 'w', encoding='utf-8') as f:
                    json.dump(model_metadata, f, indent=2)
                
                # Mark as completed
                active_training_runs[talent_id]["status"] = "completed"
                active_training_runs[talent_id]["completed_at"] = datetime.now().isoformat()
                active_training_runs[talent_id]["result"] = {
                    "model_path": str(model_output_path),
                    "training_examples": len(training_data),
                    "training_time": "simulated"
                }
                
            except Exception as e:
                logger.error(f"Basic training error: {e}")
                raise e
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE train_runs 
            SET status = 'completed', outcome = 'success'
            WHERE id = ?
        ''', (talent_id,))
        conn.commit()
        conn.close()
        
        # Broadcast completion
        await broadcast_training_completion(talent_id, True, {"message": "Training completed successfully"})
        
        log_action("training_completed", {"talent_id": talent_id})
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        active_training_runs[talent_id]["status"] = "failed"
        active_training_runs[talent_id]["error"] = str(e)
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE train_runs 
            SET status = 'failed', outcome = ?
            WHERE id = ?
        ''', (str(e), talent_id))
        conn.commit()
        conn.close()
        
        # Broadcast failure
        await broadcast_training_completion(talent_id, False, {"error": str(e)})

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Talent Factory"}

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_endpoint(websocket)

@app.get("/env/check", response_model=EnvProfile)
async def check_environment():
    """Check hardware environment and GPU capabilities"""
    profile = get_environment_profile()
    log_action("environment_check", profile.dict())
    return profile

@app.get("/models/list")
async def list_models():
    """List available models filtered by hardware compatibility"""
    env_profile = get_environment_profile()
    models = get_compatible_models(env_profile)
    log_action("models_listed", {"count": len(models)})
    return {"models": models}

@app.get("/datasets/huggingface/search")
async def search_hf_datasets(q: str, limit: int = 20):
    """Search Hugging Face datasets"""
    try:
        results = search_huggingface_datasets(q, limit)
        log_action("hf_datasets_searched", {"query": q, "results": len(results)})
        return {"datasets": results, "query": q, "total": len(results)}
    except Exception as e:
        logger.error(f"Error searching HF datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/huggingface/download")
async def download_hf_dataset(request: dict):
    """Download and process a Hugging Face dataset"""
    try:
        dataset_id = request.get("dataset_id")
        sample_size = request.get("sample_size", 1000)
        
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id is required")
        
        # For now, we'll simulate downloading and processing
        # In a real implementation, you'd use the datasets library
        logger.info(f"Downloading HF dataset: {dataset_id} (sample_size: {sample_size})")
        
        # Simulate dataset processing
        dataset_id_uuid = str(uuid.uuid4())
        
        # Create mock training data based on dataset
        training_data = []
        for i in range(min(sample_size, 100)):  # Limit for demo
            training_data.append({
                "instruction": f"Process this data from {dataset_id}:",
                "input": f"Sample data point {i+1} from Hugging Face dataset {dataset_id}",
                "output": f"Processed information from {dataset_id} dataset, point {i+1}"
            })
        
        # Save training data
        training_file_path = DATASETS_DIR / f"{dataset_id_uuid}_hf_training_data.json"
        with open(training_file_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)
        
        # Save dataset info to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (id, name, source, rows, pii_masked)
            VALUES (?, ?, ?, ?, ?)
        ''', (dataset_id_uuid, f"HF: {dataset_id}", str(training_file_path), len(training_data), False))
        conn.commit()
        conn.close()
        
        log_action("hf_dataset_downloaded", {
            "dataset_id": dataset_id,
            "local_id": dataset_id_uuid,
            "sample_size": len(training_data)
        })
        
        return {
            "dataset_id": dataset_id_uuid,
            "filename": f"HF: {dataset_id}",
            "rows": len(training_data),
            "has_pii": False,
            "status": "ingested",
            "source": "huggingface",
            "original_id": dataset_id,
            "training_examples": len(training_data)
        }
        
    except Exception as e:
        logger.error(f"Error downloading HF dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/ingest-multiple")
async def ingest_multiple_datasets(files: List[UploadFile] = File(...)):
    """Upload and process multiple unstructured files into trainable data"""
    dataset_id = str(uuid.uuid4())
    
    logger.info(f"Received {len(files)} files for processing")
    for i, file in enumerate(files):
        logger.info(f"File {i+1}: {file.filename} ({file.content_type})")
    
    all_training_data = []
    processed_files = []
    total_rows = 0
    
    try:
        for file in files:
            # Save uploaded file
            file_path = DATASETS_DIR / f"{dataset_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                mime_type = 'text/plain'
            
            # Extract text content
            text_content = extract_text_from_file(file_path, mime_type)
            
            # Detect PII
            has_pii = detect_pii(text_content)
            
            # Mask PII if detected
            if has_pii:
                text_content = mask_pii(text_content)
            
            # Convert to trainable format
            training_data = convert_to_trainable_format(text_content, file.filename)
            
            all_training_data.extend(training_data)
            processed_files.append({
                "filename": file.filename,
                "mime_type": mime_type,
                "rows": len(training_data),
                "has_pii": has_pii
            })
            total_rows += len(training_data)
        
        # Save combined training data
        training_file_path = DATASETS_DIR / f"{dataset_id}_training_data.json"
        with open(training_file_path, "w", encoding="utf-8") as f:
            json.dump(all_training_data, f, indent=2)
        
        # Save dataset info to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (id, name, source, rows, pii_masked)
            VALUES (?, ?, ?, ?, ?)
        ''', (dataset_id, f"Multi-file dataset ({len(files)} files)", str(training_file_path), total_rows, any(f["has_pii"] for f in processed_files)))
        conn.commit()
        conn.close()
        
        log_action("dataset_ingested_multiple", {
            "dataset_id": dataset_id,
            "file_count": len(files),
            "total_rows": total_rows,
            "files": [f["filename"] for f in processed_files]
        })
        
        return {
            "dataset_id": dataset_id,
            "filename": f"Multi-file dataset ({len(files)} files)",
            "rows": total_rows,
            "has_pii": any(f["has_pii"] for f in processed_files),
            "status": "ingested",
            "processed_files": processed_files,
            "training_examples": len(all_training_data)
        }
        
    except Exception as e:
        logger.error(f"Multiple dataset ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/ingest-single")
async def ingest_single_dataset(file: UploadFile = File(...)):
    """Upload and process a single unstructured file into trainable data"""
    dataset_id = str(uuid.uuid4())
    
    logger.info(f"Processing single file: {file.filename} ({file.content_type})")
    
    try:
        # Save uploaded file
        file_path = DATASETS_DIR / f"{dataset_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            mime_type = file.content_type or 'text/plain'
        
        # Extract text content
        text_content = extract_text_from_file(file_path, mime_type)
        
        # Detect PII
        has_pii = detect_pii(text_content)
        
        # Mask PII if detected
        if has_pii:
            text_content = mask_pii(text_content)
        
        # Convert to trainable format
        training_data = convert_to_trainable_format(text_content, file.filename)
        
        # Save training data
        training_file_path = DATASETS_DIR / f"{dataset_id}_training_data.json"
        with open(training_file_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2)
        
        # Save dataset info to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (id, name, source, rows, pii_masked)
            VALUES (?, ?, ?, ?, ?)
        ''', (dataset_id, file.filename, str(training_file_path), len(training_data), has_pii))
        conn.commit()
        conn.close()
        
        log_action("dataset_ingested_single", {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(training_data),
            "has_pii": has_pii
        })
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(training_data),
            "has_pii": has_pii,
            "status": "ingested",
            "mime_type": mime_type,
            "training_examples": len(training_data)
        }
        
    except Exception as e:
        logger.error(f"Single dataset ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/ingest")
async def ingest_dataset(file: UploadFile = File(...)):
    """Upload and ingest a single dataset (legacy endpoint)"""
    dataset_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = DATASETS_DIR / f"{dataset_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Process dataset
    try:
        # Read and analyze dataset
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Detect PII
        has_pii = detect_pii(content)
        
        # Count rows (simple line count for now)
        rows = len(content.split('\n'))
        
        # Save dataset info to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (id, name, source, rows, pii_masked)
            VALUES (?, ?, ?, ?, ?)
        ''', (dataset_id, file.filename, str(file_path), rows, has_pii))
        conn.commit()
        conn.close()
        
        log_action("dataset_ingested", {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": rows,
            "has_pii": has_pii
        })
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": rows,
            "has_pii": has_pii,
            "status": "ingested"
        }
        
    except Exception as e:
        logger.error(f"Dataset ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/clean")
async def clean_dataset(dataset_id: str):
    """Clean and mask PII in dataset"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT source FROM datasets WHERE id = ?', (dataset_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = Path(result[0])
        
        # Read and clean dataset
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Mask PII
        cleaned_content = mask_pii(content)
        
        # Save cleaned version
        cleaned_path = file_path.with_suffix('.cleaned' + file_path.suffix)
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        
        # Update database
        cursor.execute('''
            UPDATE datasets 
            SET pii_masked = TRUE, source = ?
            WHERE id = ?
        ''', (str(cleaned_path), dataset_id))
        conn.commit()
        conn.close()
        
        log_action("dataset_cleaned", {"dataset_id": dataset_id})
        
        return {"status": "cleaned", "dataset_id": dataset_id}
        
    except Exception as e:
        logger.error(f"Dataset cleaning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model fine-tuning"""
    train_id = str(uuid.uuid4())
    
    try:
        # Combine multiple datasets into one training dataset
        combined_dataset_path = combine_datasets(request.dataset_ids)
        
        # Create training run record
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO train_runs (id, base_model, params, status)
            VALUES (?, ?, ?, ?)
        ''', (train_id, request.base_model, json.dumps(request.params), "pending"))
        conn.commit()
        conn.close()
        
        # Start background training
        background_tasks.add_task(
            run_training,
            train_id,
            request.base_model,
            combined_dataset_path,
            request.params
        )
        
        log_action("training_started", {
            "train_id": train_id,
            "base_model": request.base_model,
            "dataset_ids": request.dataset_ids
        })
        
        return {
            "train_id": train_id,
            "status": "started",
            "message": "Training started successfully"
        }
        
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/status/{train_id}")
async def get_training_status(train_id: str):
    """Get training status and progress"""
    if train_id in active_training_runs:
        return active_training_runs[train_id]
    
    # Check database for completed/failed runs
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT status, outcome, created_at
        FROM train_runs
        WHERE id = ?
    ''', (train_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "status": result[0],
            "outcome": result[1],
            "created_at": result[2]
        }
    
    raise HTTPException(status_code=404, detail="Training run not found")

@app.post("/train/stop/{train_id}")
async def stop_training(train_id: str):
    """Stop/cancel an active training run"""
    if train_id not in active_training_runs:
        raise HTTPException(status_code=404, detail="Training run not found or not active")
    
    # Mark training as stopped
    active_training_runs[train_id]["status"] = "stopped"
    active_training_runs[train_id]["stopped_at"] = datetime.now().isoformat()
    
    # Update database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE train_runs 
        SET status = 'stopped', outcome = 'cancelled', completed_at = ?
        WHERE id = ?
    ''', (datetime.now().isoformat(), train_id))
    conn.commit()
    conn.close()
    
    logger.info(f"Training {train_id} stopped by user")
    
    return {
        "status": "stopped",
        "message": "Training stopped successfully",
        "stopped_at": datetime.now().isoformat()
    }

@app.get("/train/active")
async def get_active_trainings():
    """Get all active training runs"""
    return {
        "active_trainings": list(active_training_runs.keys()),
        "count": len(active_training_runs)
    }

@app.post("/train/broadcast-completion/{train_id}")
async def broadcast_training_completion_manual(train_id: str):
    """Manually broadcast training completion for a completed training"""
    if train_id not in active_training_runs:
        return {"error": "Training not found"}
    
    training = active_training_runs[train_id]
    if training["status"] != "completed":
        return {"error": "Training is not completed"}
    
    await broadcast_training_completion(train_id, True, training.get("result", {}))
    return {"status": "success", "message": "Completion broadcast sent"}


@app.get("/datasets/available")
async def list_datasets():
    """List all available datasets"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, source, rows, pii_masked, created_at
            FROM datasets
            ORDER BY created_at DESC
        ''')
        datasets = cursor.fetchall()
        conn.close()
        
        return {
            "datasets": [
                {
                    "id": dataset[0],
                    "name": dataset[1],
                    "source": dataset[2],
                    "rows": dataset[3],
                    "pii_masked": bool(dataset[4]),
                    "created_at": dataset[5]
                }
                for dataset in datasets
            ],
            "count": len(datasets)
        }
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/avatar/talents")
async def get_avatar_talents():
    """Get list of available avatar style talents (for Persona Foundry)"""
    try:
        import sys
        from pathlib import Path
        
        # Add the parent directory to the path to import image modules
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from image.indexer import index_talents
        from image.talent_kit import TalentKit
        
        # Use the talent-factory root directory as base_dir
        base_dir = Path(__file__).parent.parent
        talent_kit_manager = TalentKit(base_dir)
        indexed_kits = index_talents(talent_kit_manager, include_invalid=False)
        
        # Filter to only style talents (not identity talents)
        style_talents = [talent for talent in indexed_kits if talent["id"].startswith("avatar.style.")]
        
        return style_talents
        
    except Exception as e:
        logger.error(f"Failed to get avatar talents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dependencies/status")
async def get_dependency_status():
    """Get dependency installation status"""
    if not APPLE_SILICON_AVAILABLE:
        return {
            "error": "Apple Silicon support not available",
            "status": "unavailable"
        }
    
    try:
        dep_manager = get_dependency_manager()
        status = dep_manager.get_installation_status()
        return {
            "status": "success",
            "dependencies": status
        }
    except Exception as e:
        logger.error(f"Failed to get dependency status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/dependencies/install")
async def install_dependencies():
    """Install required dependencies"""
    if not APPLE_SILICON_AVAILABLE:
        return {
            "error": "Apple Silicon support not available",
            "status": "unavailable"
        }
    
    try:
        result = ensure_dependencies(auto_install=False)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/evaluate/run")
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation on a trained model"""
    eval_id = str(uuid.uuid4())
    
    try:
        # Simulate evaluation (replace with actual evaluation logic)
        metrics = {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "precision": 0.88,
            "recall": 0.79
        }
        
        safety_score = 0.92
        rubric_passed = safety_score > 0.8
        
        # Save evaluation report
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO eval_reports (id, talent_id, metrics, safety_score, rubric_passed)
            VALUES (?, ?, ?, ?, ?)
        ''', (eval_id, request.talent_id, json.dumps(metrics), safety_score, rubric_passed))
        conn.commit()
        conn.close()
        
        log_action("evaluation_completed", {
            "eval_id": eval_id,
            "talent_id": request.talent_id,
            "safety_score": safety_score,
            "rubric_passed": rubric_passed
        })
        
        return {
            "eval_id": eval_id,
            "talent_id": request.talent_id,
            "metrics": metrics,
            "safety_score": safety_score,
            "rubric_passed": rubric_passed,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/talents/publish")
async def publish_talent(talent: Talent, user: User = Depends(require_auth)):
    """Publish a talent to the local registry"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO talents (id, name, category, model_path, version, metrics, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            talent.id,
            talent.name,
            talent.category,
            talent.model_path,
            talent.version,
            json.dumps(talent.metrics) if talent.metrics else None,
            talent.status
        ))
        conn.commit()
        conn.close()
        
        log_action("talent_published", {
            "talent_id": talent.id,
            "name": talent.name,
            "category": talent.category
        })
        
        # Broadcast talent publication
        await broadcast_talent_published({
            "talent_id": talent.id,
            "name": talent.name,
            "category": talent.category,
            "version": talent.version
        })
        
        return {"status": "published", "talent_id": talent.id}
        
    except Exception as e:
        logger.error(f"Talent publishing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/talents")
async def list_talents_mcp():
    """MCP endpoint to list all talents for Dot Home discovery"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, category, version, metrics, status, created_at
            FROM talents
            WHERE status = 'active'
            ORDER BY created_at DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        
        talents = []
        for row in results:
            talents.append({
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "version": row[3],
                "metrics": json.loads(row[4]) if row[4] else {},
                "status": row[5],
                "created_at": row[6]
            })
        
        log_action("talents_listed_mcp", {"count": len(talents)})
        return {"talents": talents}
        
    except Exception as e:
        logger.error(f"MCP talents list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/talents/{talent_id}")
async def get_talent_mcp(talent_id: str):
    """MCP endpoint to get specific talent details"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, category, model_path, version, metrics, status, created_at
            FROM talents
            WHERE id = ?
        ''', (talent_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        talent = {
            "id": result[0],
            "name": result[1],
            "category": result[2],
            "model_path": result[3],
            "version": result[4],
            "metrics": json.loads(result[5]) if result[5] else {},
            "status": result[6],
            "created_at": result[7]
        }
        
        log_action("talent_retrieved_mcp", {"talent_id": talent_id})
        return talent
        
    except Exception as e:
        logger.error(f"MCP talent retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/talents/{talent_id}/test")
async def test_talent_mcp(talent_id: str):
    """MCP endpoint to test a talent with a demo"""
    try:
        # Get talent info
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, category, metrics
            FROM talents
            WHERE id = ?
        ''', (talent_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        # Simulate test (replace with actual model inference)
        test_result = {
            "talent_id": talent_id,
            "name": result[0],
            "category": result[1],
            "test_input": "Hello, how are you?",
            "test_output": "I'm doing well, thank you for asking! How can I help you today?",
            "confidence": 0.92,
            "response_time_ms": 150
        }
        
        log_action("talent_tested_mcp", {"talent_id": talent_id})
        return test_result
        
    except Exception as e:
        logger.error(f"MCP talent test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assist")
async def assist_user(message: str):
    """General assistance endpoint for the UI"""
    # This could integrate with AI service for conversational assistance
    return {
        "response": f"I'm here to help with Talent Factory! You said: {message}",
        "suggestions": [
            "Check your hardware compatibility",
            "Upload a training dataset",
            "Start fine-tuning a model",
            "Browse available talents"
        ]
    }

@app.get("/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get talent count
        cursor.execute('SELECT COUNT(*) FROM talents WHERE status = "active"')
        talent_count = cursor.fetchone()[0]
        
        # Get dataset count
        cursor.execute('SELECT COUNT(*) FROM datasets')
        dataset_count = cursor.fetchone()[0]
        
        # Get recent training runs
        cursor.execute('''
            SELECT id, base_model, status, created_at
            FROM train_runs
            ORDER BY created_at DESC
            LIMIT 5
        ''')
        recent_runs = cursor.fetchall()
        
        conn.close()
        
        # Get environment profile
        env_profile = get_environment_profile()
        
        dashboard_data = {
            "talent_count": talent_count,
            "dataset_count": dataset_count,
            "recent_runs": [
                {
                    "id": run[0],
                    "base_model": run[1],
                    "status": run[2],
                    "created_at": run[3]
                }
                for run in recent_runs
            ],
            "environment": env_profile.dict(),
            "active_training": len(active_training_runs)
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Initialize evaluation dashboard
    init_evaluation_service(Path(__file__).parent)
    
    # Initialize authentication
    init_auth_service()
    
    # Initialize feedback service
    init_feedback_service(Path(__file__).parent)
    
    # Ensure dependencies and print platform detection banner
    if APPLE_SILICON_AVAILABLE:
        try:
            # Check if we're on Apple Silicon and ensure dependencies
            if detect_apple_silicon():
                logger.info("Apple Silicon detected. Ensuring dependencies...")
                dep_result = ensure_dependencies(auto_install=False)
                if dep_result.get("overall_success", False):
                    logger.info("All dependencies are available")
                else:
                    logger.warning("Some dependencies may not be available")
            
            print_backend_banner()
        except Exception as e:
            logger.warning(f"Failed to ensure dependencies or print platform banner: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8084))
    uvicorn.run(app, host="0.0.0.0", port=port)
