"""
Configuration loader for avatar training.
Handles YAML config files and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema


class AvatarConfigLoader:
    """Loads and validates avatar training configurations."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.configs_dir = self.base_dir / "configs" / "avatar"
        
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration against the expected schema."""
        result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = [
            "id", "kind", "sdx_version", "base_model", 
            "train_data_dir", "resolution", "lora_rank",
            "max_train_steps", "learning_rate", "batch_size"
        ]
        
        for field in required_fields:
            if field not in config:
                result["errors"].append(f"Missing required field: {field}")
        
        # Validate field types and values
        if "id" in config:
            if not isinstance(config["id"], str):
                result["errors"].append("id must be a string")
            elif not config["id"].startswith("avatar."):
                result["errors"].append("id must start with 'avatar.'")
        
        if "kind" in config:
            if config["kind"] != "lora":
                result["errors"].append("kind must be 'lora'")
        
        if "sdx_version" in config:
            if config["sdx_version"] not in ["1.5", "xl"]:
                result["errors"].append("sdx_version must be '1.5' or 'xl'")
        
        if "resolution" in config:
            if not isinstance(config["resolution"], int) or config["resolution"] < 256 or config["resolution"] > 1024:
                result["errors"].append("resolution must be an integer between 256 and 1024")
        
        if "lora_rank" in config:
            if not isinstance(config["lora_rank"], int) or config["lora_rank"] < 1 or config["lora_rank"] > 128:
                result["errors"].append("lora_rank must be an integer between 1 and 128")
        
        if "max_train_steps" in config:
            if not isinstance(config["max_train_steps"], int) or config["max_train_steps"] < 100:
                result["errors"].append("max_train_steps must be an integer >= 100")
        
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if isinstance(lr, str):
                try:
                    lr = float(lr)
                except ValueError:
                    result["errors"].append("learning_rate must be a number")
            if not isinstance(lr, (int, float)) or lr <= 0:
                result["errors"].append("learning_rate must be a positive number")
        
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] < 1:
                result["errors"].append("batch_size must be a positive integer")
        
        # Check if train_data_dir exists
        if "train_data_dir" in config:
            train_dir = Path(config["train_data_dir"])
            if not train_dir.exists():
                result["warnings"].append(f"Training data directory does not exist: {train_dir}")
            elif not train_dir.is_dir():
                result["warnings"].append(f"Training data path is not a directory: {train_dir}")
        
        # Validate token field for identity talents
        if "id" in config and config["id"].startswith("avatar.identity"):
            if "token" not in config:
                result["errors"].append("Identity talents must have a 'token' field")
            elif not isinstance(config["token"], str):
                result["errors"].append("token must be a string")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def list_configs(self) -> list[Path]:
        """List all available configuration files."""
        if not self.configs_dir.exists():
            return []
        
        return list(self.configs_dir.glob("*.yaml")) + list(self.configs_dir.glob("*.yml"))
    
    def get_config_info(self, config_path: Path) -> Dict[str, Any]:
        """Get information about a configuration file."""
        try:
            config = self.load_config(config_path)
            validation = self.validate_config(config)
            
            return {
                "path": str(config_path),
                "name": config_path.stem,
                "config": config,
                "valid": validation["valid"],
                "errors": validation["errors"],
                "warnings": validation["warnings"]
            }
        except Exception as e:
            return {
                "path": str(config_path),
                "name": config_path.stem,
                "error": str(e),
                "valid": False
            }
