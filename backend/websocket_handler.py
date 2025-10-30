"""
WebSocket Handler for Real-time Training Status
"""

import json
import asyncio
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.training_subscribers: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        
        # Remove from training subscribers
        for train_id, subscribers in self.training_subscribers.items():
            subscribers.discard(websocket)
            if not subscribers:
                del self.training_subscribers[train_id]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def subscribe_to_training(self, websocket: WebSocket, train_id: str):
        """Subscribe to training updates for a specific training run"""
        if train_id not in self.training_subscribers:
            self.training_subscribers[train_id] = set()
        
        self.training_subscribers[train_id].add(websocket)
        logger.info(f"Subscribed to training {train_id}. Total subscribers: {len(self.training_subscribers[train_id])}")
    
    async def unsubscribe_from_training(self, websocket: WebSocket, train_id: str):
        """Unsubscribe from training updates"""
        if train_id in self.training_subscribers:
            self.training_subscribers[train_id].discard(websocket)
            if not self.training_subscribers[train_id]:
                del self.training_subscribers[train_id]
            logger.info(f"Unsubscribed from training {train_id}")
    
    async def broadcast_training_update(self, train_id: str, data: Dict[str, Any]):
        """Broadcast training update to all subscribers"""
        if train_id not in self.training_subscribers:
            return
        
        message = {
            "type": "training_update",
            "train_id": train_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all subscribers
        disconnected = set()
        for websocket in self.training_subscribers[train_id]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send training update: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.training_subscribers[train_id].discard(websocket)
        
        logger.info(f"Broadcasted training update for {train_id} to {len(self.training_subscribers[train_id])} subscribers")
    
    async def broadcast_system_update(self, data: Dict[str, Any]):
        """Broadcast system-wide updates to all connections"""
        message = {
            "type": "system_update",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all active connections
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send system update: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.active_connections.discard(websocket)
        
        logger.info(f"Broadcasted system update to {len(self.active_connections)} connections")
    
    async def broadcast_talent_event(self, event_type: str, talent_data: Dict[str, Any]):
        """Broadcast talent-related events"""
        message = {
            "type": "talent_event",
            "event": event_type,
            "data": talent_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all active connections
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send talent event: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.active_connections.discard(websocket)
        
        logger.info(f"Broadcasted talent event {event_type} to {len(self.active_connections)} connections")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe_training":
                train_id = message.get("train_id")
                if train_id:
                    await websocket_manager.subscribe_to_training(websocket, train_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "train_id": train_id,
                        "status": "subscribed"
                    }))
            
            elif message_type == "unsubscribe_training":
                train_id = message.get("train_id")
                if train_id:
                    await websocket_manager.unsubscribe_from_training(websocket, train_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "train_id": train_id,
                        "status": "unsubscribed"
                    }))
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

# Helper functions for broadcasting updates
async def broadcast_training_progress(train_id: str, progress: int, status: str, details: Dict[str, Any] = None):
    """Broadcast training progress update"""
    data = {
        "progress": progress,
        "status": status,
        "details": details or {}
    }
    await websocket_manager.broadcast_training_update(train_id, data)

async def broadcast_training_completion(train_id: str, success: bool, result: Dict[str, Any] = None):
    """Broadcast training completion"""
    data = {
        "progress": 100,
        "status": "completed" if success else "failed",
        "success": success,
        "result": result or {}
    }
    await websocket_manager.broadcast_training_update(train_id, data)

async def broadcast_talent_published(talent_data: Dict[str, Any]):
    """Broadcast talent publication event"""
    await websocket_manager.broadcast_talent_event("talent_published", talent_data)

async def broadcast_talent_updated(talent_data: Dict[str, Any]):
    """Broadcast talent update event"""
    await websocket_manager.broadcast_talent_event("talent_updated", talent_data)

async def broadcast_system_status(status_data: Dict[str, Any]):
    """Broadcast system status update"""
    await websocket_manager.broadcast_system_update(status_data)
