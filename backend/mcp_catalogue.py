"""
MCP Talent Catalogue API
Provides read-only MCP endpoints for Dot Home integration
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Router
mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])

# Pydantic models for MCP responses
class MCPTalent(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    metrics: Dict[str, Any] = {}
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    model_path: Optional[str] = None
    base_model: Optional[str] = None
    safety_score: Optional[float] = None
    rubric_passed: Optional[bool] = None

class MCPTalentTest(BaseModel):
    talent_id: str
    name: str
    category: Optional[str] = None
    test_input: str
    test_output: str
    confidence: float
    response_time_ms: int
    timestamp: str

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    talent_id: str
    prompt: str
    response: str
    confidence: float
    response_time_ms: int
    tokens_generated: Optional[int] = None
    timestamp: str

class MCPDiscovery(BaseModel):
    service_name: str = "Talent Factory"
    service_type: str = "talent_catalogue"
    version: str = "1.0.0"
    endpoints: List[str] = [
        "/mcp/talents",
        "/mcp/talents/{id}",
        "/mcp/talents/{id}/test",
        "/mcp/talents/{id}/infer",
        "/mcp/discovery"
    ]
    capabilities: List[str] = [
        "talent_discovery",
        "talent_testing",
        "model_inference"
    ]

class MCPCatalogue:
    """MCP Talent Catalogue service"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.registry_db = base_dir / "registry.db"
        self.models_dir = base_dir / "models"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("MCP Catalogue initialized")
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.registry_db)
    
    def get_talent_by_id(self, talent_id: str) -> Optional[Dict[str, Any]]:
        """Get talent by ID from database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get talent info
            cursor.execute('''
                SELECT id, name, category, model_path, version, metrics, status, created_at, updated_at
                FROM talents
                WHERE id = ? AND status = 'active'
            ''', (talent_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return None
            
            # Get evaluation results
            cursor.execute('''
                SELECT metrics, safety_score, rubric_passed
                FROM eval_reports
                WHERE talent_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (talent_id,))
            
            eval_result = cursor.fetchone()
            conn.close()
            
            # Parse metrics
            metrics = json.loads(result[5]) if result[5] else {}
            
            # Add evaluation metrics
            if eval_result:
                eval_metrics = json.loads(eval_result[0]) if eval_result[0] else {}
                metrics.update(eval_metrics)
                safety_score = eval_result[1]
                rubric_passed = eval_result[2]
            else:
                safety_score = None
                rubric_passed = None
            
            # Get base model from metadata if available
            base_model = None
            if result[3]:  # model_path
                metadata_path = Path(result[3]) / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        base_model = metadata.get('base_model')
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {talent_id}: {e}")
            
            return {
                "id": result[0],
                "name": result[1],
                "category": result[2],
                "model_path": result[3],
                "version": result[4],
                "metrics": metrics,
                "status": result[6],
                "created_at": result[7],
                "updated_at": result[8],
                "base_model": base_model,
                "safety_score": safety_score,
                "rubric_passed": rubric_passed
            }
            
        except Exception as e:
            logger.error(f"Failed to get talent {talent_id}: {e}")
            return None
    
    def get_all_talents(self) -> List[Dict[str, Any]]:
        """Get all active talents"""
        try:
            conn = self.get_db_connection()
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
            for result in results:
                metrics = json.loads(result[4]) if result[4] else {}
                
                talents.append({
                    "id": result[0],
                    "name": result[1],
                    "category": result[2],
                    "version": result[3],
                    "metrics": metrics,
                    "status": result[5],
                    "created_at": result[6]
                })
            
            return talents
            
        except Exception as e:
            logger.error(f"Failed to get all talents: {e}")
            return []
    
    def test_talent(self, talent_id: str, test_input: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Test a talent with sample input"""
        try:
            # Get talent info
            talent = self.get_talent_by_id(talent_id)
            if not talent:
                return None
            
            # Use default test input if not provided
            if not test_input:
                test_input = "Hello, how are you today?"
            
            # Simple test implementation (replace with actual model inference)
            test_output = self._generate_test_response(talent, test_input)
            
            return {
                "talent_id": talent_id,
                "name": talent["name"],
                "category": talent["category"],
                "test_input": test_input,
                "test_output": test_output,
                "confidence": 0.92,  # Simulated confidence
                "response_time_ms": 150,  # Simulated response time
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to test talent {talent_id}: {e}")
            return None
    
    def _generate_test_response(self, talent: Dict[str, Any], test_input: str) -> str:
        """Generate a test response based on talent category"""
        
        category = talent.get("category", "general")
        name = talent.get("name", "Unknown")
        
        # Category-specific responses
        responses = {
            "coding": f"I'm {name}, your coding assistant. I can help you with programming questions, code review, and debugging. What would you like to work on?",
            "writing": f"I'm {name}, your writing assistant. I can help you with content creation, editing, and improving your writing style. What would you like to write about?",
            "analysis": f"I'm {name}, your analysis assistant. I can help you analyze data, create reports, and provide insights. What would you like me to analyze?",
            "creative": f"I'm {name}, your creative assistant. I can help you with creative writing, brainstorming, and artistic projects. What creative project can I help you with?",
            "general": f"I'm {name}, your AI assistant. I'm here to help you with various tasks and questions. How can I assist you today?"
        }
        
        return responses.get(category, responses["general"])
    
    def run_inference(self, talent_id: str, prompt: str, max_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.9) -> Optional[Dict[str, Any]]:
        """Run actual inference on a talent model"""
        try:
            # Get talent info
            talent = self.get_talent_by_id(talent_id)
            if not talent:
                return None
            
            model_path = talent.get("model_path")
            if not model_path:
                logger.error(f"Talent {talent_id} has no model_path")
                return None
            
            start_time = time.time()
            
            # Try to load and run the model
            # Note: This is a simplified implementation - actual model loading would depend on the model type
            # For now, we'll simulate inference with a context-aware response
            response = self._run_actual_inference(talent, prompt, max_tokens, temperature, top_p)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "talent_id": talent_id,
                "prompt": prompt,
                "response": response,
                "confidence": 0.85,  # Simulated confidence
                "response_time_ms": response_time_ms,
                "tokens_generated": len(response.split()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to run inference on talent {talent_id}: {e}")
            return None
    
    def _run_actual_inference(self, talent: Dict[str, Any], prompt: str, 
                              max_tokens: int, temperature: float, top_p: float) -> str:
        """Actually run inference on the model - placeholder for real implementation"""
        
        # For now, generate a context-aware response based on the talent
        category = talent.get("category", "general")
        name = talent.get("name", "Talent")
        base_model = talent.get("base_model", "unknown")
        
        # Category-specific response templates
        templates = {
            "coding": f"I'm {name}, your coding assistant trained on {base_model}. I understand you're asking about: {prompt}. Let me help you with that! Based on the context, here's what I can suggest...",
            "writing": f"I'm {name}, your writing assistant. Regarding '{prompt}', I can help you craft a compelling response or piece of content. Let me assist you with that.",
            "analysis": f"As {name}, your analysis assistant, I can help analyze: {prompt}. Let me provide insights based on the data and patterns I've learned.",
            "creative": f"I'm {name}, your creative companion. About '{prompt}' - let me help you explore creative possibilities and ideas!",
            "general": f"I'm {name}, your AI assistant. I understand you're asking about: {prompt}. Let me provide a helpful response..."
        }
        
        # Generate a response based on category
        base_response = templates.get(category, templates["general"])
        
        # Add some variation based on the prompt
        if "?" in prompt:
            response = base_response + " I'd be happy to answer your question in detail."
        elif "help" in prompt.lower() or "how" in prompt.lower():
            response = base_response + " I can guide you through the process step by step."
        else:
            response = base_response + " Here's my perspective on this topic."
        
        return response

# Initialize MCP Catalogue
mcp_catalogue = None

def init_mcp_catalogue(base_dir: Path):
    """Initialize MCP Catalogue"""
    global mcp_catalogue
    mcp_catalogue = MCPCatalogue(base_dir)

# MCP Endpoints

@mcp_router.get("/discovery", response_model=MCPDiscovery)
async def mcp_discovery():
    """MCP service discovery endpoint"""
    return MCPDiscovery()

@mcp_router.get("/talents")
async def list_talents_mcp():
    """MCP endpoint to list all talents for Dot Home discovery"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        talents = mcp_catalogue.get_all_talents()
        
        logger.info(f"MCP: Listed {len(talents)} talents")
        return {"talents": talents}
        
    except Exception as e:
        logger.error(f"MCP: Failed to list talents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp_router.get("/talents/{talent_id}", response_model=MCPTalent)
async def get_talent_mcp(talent_id: str):
    """MCP endpoint to get specific talent details"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        talent = mcp_catalogue.get_talent_by_id(talent_id)
        
        if not talent:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        logger.info(f"MCP: Retrieved talent {talent_id}")
        return MCPTalent(**talent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP: Failed to get talent {talent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp_router.get("/talents/{talent_id}/test", response_model=MCPTalentTest)
async def test_talent_mcp(talent_id: str, test_input: Optional[str] = None):
    """MCP endpoint to test a talent with a demo"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        test_result = mcp_catalogue.test_talent(talent_id, test_input)
        
        if not test_result:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        logger.info(f"MCP: Tested talent {talent_id}")
        return MCPTalentTest(**test_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP: Failed to test talent {talent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp_router.get("/talents/{talent_id}/metadata")
async def get_talent_metadata(talent_id: str):
    """Get detailed metadata for a talent"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        talent = mcp_catalogue.get_talent_by_id(talent_id)
        
        if not talent:
            raise HTTPException(status_code=404, detail="Talent not found")
        
        # Get additional metadata from model files
        metadata = {}
        if talent.get("model_path"):
            metadata_path = Path(talent["model_path"]) / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read metadata file: {e}")
        
        # Combine talent info with metadata
        result = {
            "talent": talent,
            "metadata": metadata,
            "capabilities": [
                "text_generation",
                "conversation",
                "task_completion"
            ],
            "usage_limits": {
                "max_tokens": 2048,
                "rate_limit": "100 requests/hour"
            }
        }
        
        logger.info(f"MCP: Retrieved metadata for talent {talent_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP: Failed to get metadata for talent {talent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp_router.get("/health")
async def mcp_health():
    """MCP service health check"""
    return {
        "status": "healthy",
        "service": "MCP Talent Catalogue",
        "version": "1.0.0",
        "talents_count": len(mcp_catalogue.get_all_talents()) if mcp_catalogue else 0
    }

# Subscription endpoints for real-time updates
@mcp_router.get("/talents/events")
async def get_talent_events():
    """Get recent talent events for subscription"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        # Get recent events from audit logs
        events = []
        logs_dir = mcp_catalogue.base_dir / "logs"
        
        if logs_dir.exists():
            # Get today's audit log
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = logs_dir / f"audit_{today}.log"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 10 events
                    for line in lines[-10:]:
                        try:
                            event = json.loads(line.strip())
                            if event.get('action') in ['talent_published', 'talent_updated']:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
        
        return {
            "events": events,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"MCP: Failed to get talent events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mcp_router.websocket("/talents/subscribe")
async def subscribe_to_talent_events(websocket: WebSocket):
    """WebSocket endpoint for real-time talent event subscriptions"""
    await websocket.accept()
    
    try:
        while True:
            # Wait for subscription message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Client wants to subscribe to talent events
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "status": "subscribed",
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Keep connection alive and send events
                while True:
                    # Check for new events every 5 seconds
                    await asyncio.sleep(5)
                    
                    # Get recent events
                    try:
                        events = []
                        logs_dir = mcp_catalogue.base_dir / "logs"
                        
                        if logs_dir.exists():
                            today = datetime.now().strftime('%Y-%m-%d')
                            log_file = logs_dir / f"audit_{today}.log"
                            
                            if log_file.exists():
                                with open(log_file, 'r') as f:
                                    lines = f.readlines()
                                    # Get last 5 events
                                    for line in lines[-5:]:
                                        try:
                                            event = json.loads(line.strip())
                                            if event.get('action') in ['talent_published', 'talent_updated']:
                                                events.append(event)
                                        except json.JSONDecodeError:
                                            continue
                        
                        if events:
                            await websocket.send_text(json.dumps({
                                "type": "talent_event",
                                "events": events,
                                "timestamp": datetime.now().isoformat()
                            }))
                    
                    except Exception as e:
                        logger.error(f"Error in talent event subscription: {e}")
                        break
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message.get('type')}"
                }))
    
    except WebSocketDisconnect:
        logger.info("Talent event subscription disconnected")
    except Exception as e:
        logger.error(f"Talent event subscription error: {e}")
        try:
            await websocket.close()
        except:
            pass

@mcp_router.post("/talents/{talent_id}/infer", response_model=InferenceResponse)
async def infer_talent_mcp(talent_id: str, request: InferenceRequest):
    """MCP endpoint to run inference on a talent with a custom prompt"""
    if not mcp_catalogue:
        raise HTTPException(status_code=500, detail="MCP Catalogue not initialized")
    
    try:
        result = mcp_catalogue.run_inference(
            talent_id=talent_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Talent not found or inference failed")
        
        logger.info(f"MCP: Ran inference on talent {talent_id}")
        return InferenceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP: Failed to run inference on talent {talent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
