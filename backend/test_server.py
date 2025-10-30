#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime
from typing import List

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscribers: dict = {}  # train_id -> list of websockets

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Remove from all subscriber lists
        for train_id, connections in self.subscribers.items():
            if websocket in connections:
                connections.remove(websocket)

    async def subscribe_to_training(self, websocket: WebSocket, train_id: str):
        if train_id not in self.subscribers:
            self.subscribers[train_id] = []
        if websocket not in self.subscribers[train_id]:
            self.subscribers[train_id].append(websocket)
        
        # Send subscription confirmation
        await websocket.send_text(json.dumps({
            "type": "subscription_confirmed",
            "train_id": train_id,
            "status": "subscribed"
        }))

    async def broadcast_training_update(self, train_id: str, data: dict):
        if train_id in self.subscribers:
            message = json.dumps({
                "type": "training_update",
                "train_id": train_id,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            for connection in self.subscribers[train_id][:]:  # Copy list to avoid modification during iteration
                try:
                    await connection.send_text(message)
                except:
                    # Remove disconnected connections
                    self.subscribers[train_id].remove(connection)

manager = ConnectionManager()

# Simple in-memory storage for testing
active_trainings = {}

class TrainingRequest(BaseModel):
    base_model: str
    dataset_ids: list
    params: dict

class TestRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.get("/")
async def root():
    return {"message": "Talent Factory Test Server"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                train_id = message.get("train_id")
                if train_id:
                    await manager.subscribe_to_training(websocket, train_id)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/train/start")
async def start_training(request: TrainingRequest):
    """Start a simple test training"""
    train_id = f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Initialize training as started
    active_trainings[train_id] = {
        "status": "running",
        "progress": 0,
        "message": "Training started",
        "result": None
    }
    
    # Start the simulation in the background using asyncio.create_task
    async def simulate_training():
        for progress in [10, 25, 50, 75, 90, 100]:
            await asyncio.sleep(0.5)  # Simulate training time
            active_trainings[train_id]["progress"] = progress
            active_trainings[train_id]["message"] = f"Training progress: {progress}%"
            
            # Broadcast update via WebSocket
            await manager.broadcast_training_update(train_id, {
                "status": "running" if progress < 100 else "completed",
                "progress": progress,
                "message": active_trainings[train_id]["message"]
            })
        
        # Mark as completed
        active_trainings[train_id]["status"] = "completed"
        active_trainings[train_id]["result"] = {
            "model_path": f"/test/models/{train_id}",
            "train_id": train_id
        }
        
        # Final update
        await manager.broadcast_training_update(train_id, {
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully"
        })
    
    # Start the simulation in the background
    asyncio.create_task(simulate_training())
    
    return {
        "train_id": train_id,
        "status": "started",
        "message": "Training started successfully"
    }

@app.get("/train/status/{train_id}")
async def get_training_status(train_id: str):
    """Get training status"""
    if train_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    return active_trainings[train_id]

@app.post("/train/test/{train_id}")
async def test_trained_model(train_id: str, request: TestRequest):
    """Test a trained model with a custom prompt"""
    if train_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    training = active_trainings[train_id]
    if training["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training is not completed yet")
    
    # Simulate a response
    simulated_response = f"""Model response to: '{request.prompt}'

This is a simulated response from the trained model. The model has been fine-tuned and is now responding to your custom prompt.

Prompt: {request.prompt}
Max tokens: {request.max_tokens}
Training ID: {train_id}
Timestamp: {datetime.now().isoformat()}

The model would normally generate a more sophisticated response based on the training data, but this is a test simulation."""
    
    return {
        "response": simulated_response,
        "prompt": request.prompt,
        "model_path": training["result"]["model_path"],
        "train_id": train_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/train/active")
async def get_active_trainings():
    """Get active trainings"""
    return {
        "active_trainings": list(active_trainings.keys()),
        "count": len(active_trainings)
    }

@app.get("/datasets/available")
async def get_available_datasets():
    """Get available datasets"""
    return {
        "datasets": [
            {
                "id": "test-dataset-1",
                "name": "Test Dataset 1",
                "description": "A test dataset for demonstration",
                "size": 100,
                "format": "json"
            },
            {
                "id": "test-dataset-2", 
                "name": "Test Dataset 2",
                "description": "Another test dataset",
                "size": 200,
                "format": "json"
            }
        ]
    }

@app.get("/models/list")
async def get_models_list():
    """Get available models"""
    return {
        "models": [
            {
                "id": "test-model-1",
                "name": "Test Model 1",
                "description": "A test model for demonstration",
                "type": "text-generation",
                "size": "7B"
            },
            {
                "id": "test-model-2",
                "name": "Test Model 2", 
                "description": "Another test model",
                "type": "text-generation",
                "size": "13B"
            }
        ]
    }

@app.get("/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    return {
        "stats": {
            "total_talents": 5,
            "active_trainings": len(active_trainings),
            "completed_trainings": 10,
            "total_datasets": 3
        },
        "recent_activity": [
            {
                "id": "activity-1",
                "type": "training_completed",
                "message": "Training completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

@app.get("/mcp/talents")
async def get_mcp_talents():
    """Get MCP talents"""
    return {
        "talents": [
            {
                "id": "talent-1",
                "name": "Test Talent 1",
                "description": "A test talent",
                "status": "active"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
