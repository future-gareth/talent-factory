"""
Authentication module for Talent Factory
"""

import os
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class User(BaseModel):
    username: str
    is_active: bool
    last_login: Optional[str] = None

class AuthConfig(BaseModel):
    enabled: bool = False
    username: str = "admin"
    password_hash: str = ""
    token_secret: str = ""
    token_expiry_hours: int = 24

class AuthService:
    """Authentication service for Talent Factory"""
    
    def __init__(self, config_path: str = "/opt/talent-factory/auth.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize default config if needed
        if not self.config.token_secret:
            self.config.token_secret = secrets.token_urlsafe(32)
            self.save_config()
        
        logger.info(f"AuthService initialized. Auth enabled: {self.config.enabled}")
    
    def load_config(self) -> AuthConfig:
        """Load authentication configuration"""
        try:
            if os.path.exists(self.config_path):
                import json
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return AuthConfig(**data)
            else:
                return AuthConfig()
        except Exception as e:
            logger.warning(f"Failed to load auth config: {e}")
            return AuthConfig()
    
    def save_config(self):
        """Save authentication configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            import json
            with open(self.config_path, 'w') as f:
                json.dump(self.config.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save auth config: {e}")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == password_hash
    
    def create_token(self, username: str) -> str:
        """Create JWT token"""
        payload = {
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=self.config.token_expiry_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.config.token_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config.token_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        if not self.config.enabled:
            return True  # Auth disabled, allow all
        
        if username != self.config.username:
            return False
        
        if not self.config.password_hash:
            return False
        
        return self.verify_password(password, self.config.password_hash)
    
    def set_password(self, username: str, password: str):
        """Set password for user"""
        if username != self.config.username:
            raise ValueError("Invalid username")
        
        self.config.password_hash = self.hash_password(password)
        self.save_config()
        logger.info(f"Password set for user: {username}")
    
    def enable_auth(self, username: str, password: str):
        """Enable authentication with username and password"""
        self.config.enabled = True
        self.config.username = username
        self.config.password_hash = self.hash_password(password)
        self.save_config()
        logger.info(f"Authentication enabled for user: {username}")
    
    def disable_auth(self):
        """Disable authentication"""
        self.config.enabled = False
        self.config.password_hash = ""
        self.save_config()
        logger.info("Authentication disabled")
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user information"""
        if not self.config.enabled:
            return User(username="anonymous", is_active=True)
        
        if username != self.config.username:
            return None
        
        return User(
            username=username,
            is_active=True,
            last_login=datetime.now().isoformat()
        )

# Global auth service instance
auth_service = None

def init_auth_service(config_path: str = "/opt/talent-factory/auth.json"):
    """Initialize authentication service"""
    global auth_service
    auth_service = AuthService(config_path)

def get_auth_service() -> AuthService:
    """Get authentication service instance"""
    if not auth_service:
        raise HTTPException(status_code=500, detail="Authentication service not initialized")
    return auth_service

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get current authenticated user"""
    auth_svc = get_auth_service()
    
    # If auth is disabled, return anonymous user
    if not auth_svc.config.enabled:
        return User(username="anonymous", is_active=True)
    
    # If no credentials provided, return None
    if not credentials:
        return None
    
    # Verify token
    payload = auth_svc.verify_token(credentials.credentials)
    if not payload:
        return None
    
    # Get user
    user = auth_svc.get_user(payload["username"])
    return user

async def require_auth(user: Optional[User] = Depends(get_current_user)) -> User:
    """Require authentication"""
    auth_svc = get_auth_service()
    
    # If auth is disabled, allow access
    if not auth_svc.config.enabled:
        return User(username="anonymous", is_active=True)
    
    # If no user or user is not active, deny access
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Authentication endpoints
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """Login endpoint"""
    auth_svc = get_auth_service()
    
    if not auth_svc.config.enabled:
        raise HTTPException(status_code=400, detail="Authentication is disabled")
    
    if not auth_svc.authenticate(login_data.username, login_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth_svc.create_token(login_data.username)
    
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=auth_svc.config.token_expiry_hours * 3600
    )

@auth_router.get("/me", response_model=User)
async def get_current_user_info(user: User = Depends(require_auth)):
    """Get current user information"""
    return user

@auth_router.post("/enable")
async def enable_authentication(username: str, password: str):
    """Enable authentication (admin only)"""
    auth_svc = get_auth_service()
    auth_svc.enable_auth(username, password)
    return {"message": "Authentication enabled"}

@auth_router.post("/disable")
async def disable_authentication():
    """Disable authentication (admin only)"""
    auth_svc = get_auth_service()
    auth_svc.disable_auth()
    return {"message": "Authentication disabled"}

@auth_router.post("/set-password")
async def set_password(username: str, password: str, user: User = Depends(require_auth)):
    """Set password for user"""
    auth_svc = get_auth_service()
    auth_svc.set_password(username, password)
    return {"message": "Password updated"}

@auth_router.get("/status")
async def get_auth_status():
    """Get authentication status"""
    auth_svc = get_auth_service()
    return {
        "enabled": auth_svc.config.enabled,
        "username": auth_svc.config.username if auth_svc.config.enabled else None
    }
