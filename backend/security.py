"""
Talent Factory Security Module
Implements PII detection, local-first security, and audit logging
"""

import re
import json
import logging
import hashlib
import ipaddress
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIDetector:
    """PII Detection and Masking"""
    
    def __init__(self):
        # PII patterns
        self.patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'mac_address': r'\b[0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}\b',
            'driver_license': r'\b[A-Z]{1,2}\d{6,8}\b',
            'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
            'bank_account': r'\b\d{8,17}\b',
            'routing_number': r'\b\d{9}\b'
        }
        
        # Replacement patterns
        self.replacements = {
            'ssn': '[SSN_MASKED]',
            'credit_card': '[CARD_MASKED]',
            'email': '[EMAIL_MASKED]',
            'phone': '[PHONE_MASKED]',
            'ip_address': '[IP_MASKED]',
            'mac_address': '[MAC_MASKED]',
            'driver_license': '[DL_MASKED]',
            'passport': '[PASSPORT_MASKED]',
            'bank_account': '[ACCOUNT_MASKED]',
            'routing_number': '[ROUTING_MASKED]'
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        detected = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[pii_type] = list(set(matches))  # Remove duplicates
        
        return detected
    
    def has_pii(self, text: str) -> bool:
        """Check if text contains PII"""
        detected = self.detect_pii(text)
        return len(detected) > 0
    
    def mask_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Mask PII in text and return masked text with count of masked items"""
        masked_text = text
        masked_count = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                masked_text = re.sub(pattern, self.replacements[pii_type], masked_text, flags=re.IGNORECASE)
                masked_count[pii_type] = len(matches)
        
        return masked_text, masked_count
    
    def validate_dataset_safety(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset for PII and safety"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect PII
            detected_pii = self.detect_pii(content)
            has_pii = len(detected_pii) > 0
            
            # Mask PII
            masked_content, masked_count = self.mask_pii(content)
            
            # Calculate safety score
            safety_score = self._calculate_safety_score(content, detected_pii)
            
            return {
                "has_pii": has_pii,
                "detected_pii": detected_pii,
                "masked_count": masked_count,
                "safety_score": safety_score,
                "masked_content": masked_content,
                "original_size": len(content),
                "masked_size": len(masked_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate dataset safety: {e}")
            return {
                "has_pii": True,  # Assume unsafe if validation fails
                "detected_pii": {},
                "masked_count": {},
                "safety_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_safety_score(self, content: str, detected_pii: Dict[str, List[str]]) -> float:
        """Calculate safety score based on PII detection"""
        if not detected_pii:
            return 1.0
        
        # Weight different PII types
        weights = {
            'ssn': 0.3,
            'credit_card': 0.25,
            'email': 0.1,
            'phone': 0.1,
            'ip_address': 0.05,
            'mac_address': 0.05,
            'driver_license': 0.1,
            'passport': 0.1,
            'bank_account': 0.2,
            'routing_number': 0.15
        }
        
        total_penalty = 0.0
        for pii_type, matches in detected_pii.items():
            weight = weights.get(pii_type, 0.1)
            penalty = weight * len(matches)
            total_penalty += penalty
        
        # Normalize penalty based on content length
        content_length = len(content)
        normalized_penalty = min(total_penalty / (content_length / 1000), 1.0)
        
        return max(0.0, 1.0 - normalized_penalty)

class LocalSecurityManager:
    """Local-First Security Manager"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.config_file = base_dir / "security_config.json"
        self.load_config()
    
    def load_config(self):
        """Load security configuration"""
        default_config = {
            "local_only": True,
            "allowed_networks": ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"],
            "block_external": True,
            "require_auth": False,
            "audit_enabled": True,
            "pii_detection": True,
            "data_retention_days": 30
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save security configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save security config: {e}")
    
    def is_allowed_network(self, client_ip: str) -> bool:
        """Check if client IP is in allowed networks"""
        if not self.config.get("local_only", True):
            return True
        
        try:
            client_addr = ipaddress.ip_address(client_ip)
            allowed_networks = self.config.get("allowed_networks", [])
            
            for network in allowed_networks:
                if client_addr in ipaddress.ip_network(network):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check network access for {client_ip}: {e}")
            return False
    
    def validate_request(self, request: Request) -> bool:
        """Validate incoming request"""
        client_ip = request.client.host
        
        # Check network access
        if not self.is_allowed_network(client_ip):
            logger.warning(f"Blocked request from unauthorized network: {client_ip}")
            return False
        
        # Check for external access attempts
        if self.config.get("block_external", True):
            if self._is_external_request(request):
                logger.warning(f"Blocked external request from: {client_ip}")
                return False
        
        return True
    
    def _is_external_request(self, request: Request) -> bool:
        """Check if request is from external source"""
        # Check referer header
        referer = request.headers.get("referer", "")
        if referer and not any(domain in referer for domain in ["localhost", "127.0.0.1", "talentfactory.local"]):
            return True
        
        # Check user agent for suspicious patterns
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_patterns = ["bot", "crawler", "spider", "scraper"]
        if any(pattern in user_agent for pattern in suspicious_patterns):
            return True
        
        return False

class AuditLogger:
    """Audit Logging System"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logs_dir = base_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize audit database
        self.audit_db = base_dir / "audit.db"
        self.init_audit_db()
    
    def init_audit_db(self):
        """Initialize audit database"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    user_id TEXT,
                    client_ip TEXT,
                    resource TEXT,
                    details TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def log_action(
        self, 
        action: str, 
        details: Dict[str, Any] = None,
        user_id: str = None,
        client_ip: str = None,
        resource: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Log an action for audit trail"""
        timestamp = datetime.now().isoformat()
        
        # Log to file
        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "user_id": user_id,
            "client_ip": client_ip,
            "resource": resource,
            "details": details or {},
            "success": success,
            "error_message": error_message
        }
        
        log_file = self.logs_dir / f"audit_{datetime.now().strftime('%Y-%m-%d')}.log"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        # Log to database
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_logs (timestamp, action, user_id, client_ip, resource, details, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, action, user_id, client_ip, resource,
                json.dumps(details) if details else None, success, error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to write audit database: {e}")
    
    def get_audit_logs(
        self, 
        start_date: str = None, 
        end_date: str = None,
        action: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit logs with filters"""
        try:
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            logs = []
            for row in results:
                logs.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "action": row[2],
                    "user_id": row[3],
                    "client_ip": row[4],
                    "resource": row[5],
                    "details": json.loads(row[6]) if row[6] else {},
                    "success": row[7],
                    "error_message": row[8]
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    def cleanup_old_logs(self, retention_days: int = 30):
        """Clean up old audit logs"""
        try:
            cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            # Clean database
            conn = sqlite3.connect(self.audit_db)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM audit_logs WHERE timestamp < ?', (cutoff_iso,))
            conn.commit()
            conn.close()
            
            # Clean log files
            for log_file in self.logs_dir.glob("audit_*.log"):
                if log_file.stat().st_mtime < cutoff_date:
                    log_file.unlink()
            
            logger.info(f"Cleaned up audit logs older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup audit logs: {e}")

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security Middleware"""
    
    def __init__(self, app, security_manager: LocalSecurityManager, audit_logger: AuditLogger):
        super().__init__(app)
        self.security_manager = security_manager
        self.audit_logger = audit_logger
    
    async def dispatch(self, request: Request, call_next):
        # Validate request
        if not self.security_manager.validate_request(request):
            self.audit_logger.log_action(
                "request_blocked",
                {"path": request.url.path, "method": request.method},
                client_ip=request.client.host,
                success=False,
                error_message="Unauthorized network access"
            )
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Log request
        start_time = datetime.now()
        
        try:
            response = await call_next(request)
            
            # Log successful request
            duration = (datetime.now() - start_time).total_seconds()
            self.audit_logger.log_action(
                "request_processed",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_seconds": duration
                },
                client_ip=request.client.host,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            duration = (datetime.now() - start_time).total_seconds()
            self.audit_logger.log_action(
                "request_failed",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "duration_seconds": duration
                },
                client_ip=request.client.host,
                success=False,
                error_message=str(e)
            )
            raise

# Initialize security components
pii_detector = PIIDetector()
security_manager = None
audit_logger = None

def init_security(base_dir: Path):
    """Initialize security components"""
    global security_manager, audit_logger
    
    security_manager = LocalSecurityManager(base_dir)
    audit_logger = AuditLogger(base_dir)
    
    logger.info("Security components initialized")

def get_security_manager() -> LocalSecurityManager:
    """Get security manager instance"""
    if not security_manager:
        raise RuntimeError("Security manager not initialized")
    return security_manager

def get_audit_logger() -> AuditLogger:
    """Get audit logger instance"""
    if not audit_logger:
        raise RuntimeError("Audit logger not initialized")
    return audit_logger

def get_pii_detector() -> PIIDetector:
    """Get PII detector instance"""
    return pii_detector
