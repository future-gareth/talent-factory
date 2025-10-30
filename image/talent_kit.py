"""
Talent Kit management for Persona Foundry integration.
Handles the creation, validation, and management of avatar talent kits.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import jsonschema
from datetime import datetime


class TalentKit:
    """Manages avatar talent kits for Persona Foundry."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.talents_dir = self.base_dir / "talents" / "avatar"
        self.data_dir = self.base_dir / "data" / "avatar"
        self.configs_dir = self.base_dir / "configs" / "avatar"
        self.schemas_dir = self.base_dir / "schemas"
        
        # Ensure directories exist
        self.talents_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
    def get_schema(self) -> Dict[str, Any]:
        """Load the avatar talent schema."""
        schema_path = self.schemas_dir / "avatar_talent.json"
        with open(schema_path, 'r') as f:
            return json.load(f)
    
    def create_kit_structure(self, talent_id: str) -> Path:
        """Create the folder structure for a talent kit."""
        kit_dir = self.talents_dir / talent_id
        kit_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (kit_dir / "examples").mkdir(exist_ok=True)
        (kit_dir / "controls").mkdir(exist_ok=True)
        
        return kit_dir
    
    def validate_kit(self, talent_id: str) -> Dict[str, Any]:
        """Validate a talent kit against the schema."""
        kit_dir = self.talents_dir / talent_id
        manifest_path = kit_dir / "talent.json"
        weights_path = kit_dir / "weights.safetensors"
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "kit_path": str(kit_dir)
        }
        
        # Check if kit directory exists
        if not kit_dir.exists():
            result["errors"].append(f"Kit directory not found: {kit_dir}")
            return result
        
        # Check if manifest exists
        if not manifest_path.exists():
            result["errors"].append(f"Manifest not found: {manifest_path}")
            return result
        
        # Check if weights exist
        if not weights_path.exists():
            result["errors"].append(f"Weights not found: {weights_path}")
            return result
        
        # Load and validate manifest
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON in manifest: {e}")
            return result
        
        # Validate against schema
        try:
            schema = self.get_schema()
            jsonschema.validate(manifest, schema)
        except jsonschema.ValidationError as e:
            result["errors"].append(f"Schema validation failed: {e.message}")
            return result
        except Exception as e:
            result["errors"].append(f"Schema validation error: {e}")
            return result
        
        # Check file sizes
        try:
            weights_size = weights_path.stat().st_size
            manifest_size_mb = manifest.get("size_mb", 0)
            actual_size_mb = weights_size / (1024 * 1024)
            
            if abs(actual_size_mb - manifest_size_mb) > 0.1:
                result["warnings"].append(
                    f"Size mismatch: manifest says {manifest_size_mb}MB, "
                    f"actual is {actual_size_mb:.2f}MB"
                )
        except Exception as e:
            result["warnings"].append(f"Could not check file size: {e}")
        
        # Check examples directory
        examples_dir = kit_dir / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
            if len(example_files) == 0:
                result["warnings"].append("No example images found")
            elif len(example_files) < 3:
                result["warnings"].append(f"Only {len(example_files)} example images found (recommended: 3-6)")
        
        result["valid"] = len(result["errors"]) == 0
        result["manifest"] = manifest
        return result
    
    def list_kits(self) -> List[Dict[str, Any]]:
        """List all talent kits and their validation status."""
        kits = []
        
        if not self.talents_dir.exists():
            return kits
        
        for kit_dir in self.talents_dir.iterdir():
            if kit_dir.is_dir():
                talent_id = kit_dir.name
                validation = self.validate_kit(talent_id)
                kits.append({
                    "id": talent_id,
                    "path": str(kit_dir),
                    "valid": validation["valid"],
                    "errors": validation["errors"],
                    "warnings": validation["warnings"],
                    "manifest": validation.get("manifest")
                })
        
        return kits
    
    def get_kit_path(self, talent_id: str) -> Path:
        """Get the path to a talent kit."""
        return self.talents_dir / talent_id
    
    def ensure_data_directories(self, talent_type: str, talent_name: str) -> Path:
        """Ensure data directories exist for a talent."""
        if talent_type == "identity":
            data_path = self.data_dir / "identity" / talent_name / "images"
        elif talent_type == "style":
            data_path = self.data_dir / "style" / talent_name / "images"
        else:
            raise ValueError(f"Invalid talent type: {talent_type}")
        
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path
    
    def create_manifest_template(self, talent_id: str, **kwargs) -> Dict[str, Any]:
        """Create a manifest template for a talent."""
        now = datetime.now().isoformat()
        
        # Convert string numbers to proper types
        def convert_number(value, default):
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return value
        
        template = {
            "id": talent_id,
            "category": "Presence/Avatar",
            "kind": "lora",
            "sdx_version": kwargs.get("sdx_version", "1.5"),
            "base_model": kwargs.get("base_model", "runwayml/stable-diffusion-v1-5"),
            "lora_rank": kwargs.get("lora_rank", 16),
            "default_weight": kwargs.get("default_weight", 1.0),
            "negatives": kwargs.get("negatives", "photo, text, watermark"),
            "size_mb": kwargs.get("size_mb", 0.0),
            "created_at": now,
            "training_config": {
                "resolution": kwargs.get("resolution", 640),
                "max_train_steps": kwargs.get("max_train_steps", 3000),
                "learning_rate": convert_number(kwargs.get("learning_rate", 1e-4), 1e-4),
                "batch_size": kwargs.get("batch_size", 1)
            },
            "metadata": {
                "description": kwargs.get("description", ""),
                "tags": kwargs.get("tags", []),
                "author": kwargs.get("author", ""),
                "license": kwargs.get("license", "Private")
            }
        }
        
        # Add optional fields
        if "token" in kwargs:
            template["token"] = kwargs["token"]
        
        if "intended_mix" in kwargs:
            template["intended_mix"] = kwargs["intended_mix"]
        
        return template
    
    def save_manifest(self, talent_id: str, manifest: Dict[str, Any]) -> Path:
        """Save a manifest to a talent kit."""
        kit_dir = self.get_kit_path(talent_id)
        manifest_path = kit_dir / "talent.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
