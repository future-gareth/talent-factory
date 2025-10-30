"""
Thermal management for Talent Factory.

Provides quiet/low-heat defaults for Apple Silicon and other platforms to
prevent excessive fan noise and heat generation during training.
"""

import os
import platform
import subprocess
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class HeatProfile(Enum):
    """Heat profile options."""
    QUIET = "quiet"
    NORMAL = "normal"


class ThermalManager:
    """Manages thermal settings for training."""
    
    def __init__(self):
        self.is_macos = platform.system() == "Darwin"
        self.is_apple_silicon = self._detect_apple_silicon()
        self.current_profile = HeatProfile.NORMAL
        self.original_env = {}
        
        logger.info(f"Thermal manager initialized. macOS: {self.is_macos}, Apple Silicon: {self.is_apple_silicon}")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            if not self.is_macos:
                return False
            
            arch = platform.machine()
            if arch != "arm64":
                return False
            
            # Check for Apple Silicon chip
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                hardware = data.get("SPHardwareDataType", [{}])[0]
                chip_type = hardware.get("chip_type", "")
                
                apple_silicon_chips = ["Apple M1", "Apple M2", "Apple M3", "Apple M4"]
                return any(chip in chip_type for chip in apple_silicon_chips)
            
            return True  # Fallback for arm64 macOS
        except Exception as e:
            logger.warning(f"Error detecting Apple Silicon: {e}")
            return False
    
    def apply_thermal_profile(self, profile: HeatProfile = HeatProfile.QUIET) -> Dict[str, Any]:
        """
        Apply thermal management settings.
        
        Args:
            profile: Heat profile to apply
            
        Returns:
            Dictionary of applied settings
        """
        self.current_profile = profile
        applied_settings = {}
        
        logger.info(f"Applying thermal profile: {profile.value}")
        
        if profile == HeatProfile.QUIET:
            applied_settings = self._apply_quiet_profile()
        elif profile == HeatProfile.NORMAL:
            applied_settings = self._apply_normal_profile()
        
        logger.info(f"Applied thermal settings: {applied_settings}")
        return applied_settings
    
    def _apply_quiet_profile(self) -> Dict[str, Any]:
        """Apply quiet/low-heat profile."""
        settings = {}
        
        # Set OpenMP threads for CPU-bound operations
        if "OMP_NUM_THREADS" not in os.environ:
            self.original_env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
            os.environ["OMP_NUM_THREADS"] = "4"
            settings["omp_threads"] = 4
        
        # Apple Silicon specific settings
        if self.is_apple_silicon:
            # Set MPS watermarks for memory management
            if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
                self.original_env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.6"
                settings["mps_high_watermark"] = 0.6
            
            if "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
                self.original_env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.4"
                settings["mps_low_watermark"] = 0.4
        
        # macOS specific settings
        if self.is_macos:
            # Reduce system load
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                self.original_env["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
                settings["cuda_alloc_conf"] = "max_split_size_mb:128"
        
        settings["profile"] = "quiet"
        settings["platform"] = "apple_silicon" if self.is_apple_silicon else "other"
        
        return settings
    
    def _apply_normal_profile(self) -> Dict[str, Any]:
        """Apply normal profile (restore defaults)."""
        settings = {}
        
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
        
        # Clear original env tracking
        self.original_env.clear()
        
        settings["profile"] = "normal"
        settings["platform"] = "apple_silicon" if self.is_apple_silicon else "other"
        
        return settings
    
    def get_training_config(self, profile: HeatProfile = HeatProfile.QUIET) -> Dict[str, Any]:
        """
        Get training configuration optimized for thermal profile.
        
        Args:
            profile: Heat profile to optimize for
            
        Returns:
            Training configuration dictionary
        """
        if profile == HeatProfile.QUIET:
            return {
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_steps": 1000,  # Time-boxed for testing
                "warmup_steps": 50,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "max_grad_norm": 1.0,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "adam_epsilon": 1e-8,
                "profile": "quiet"
            }
        else:
            return {
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "max_steps": 5000,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500,
                "max_grad_norm": 1.0,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "adam_epsilon": 1e-8,
                "profile": "normal"
            }
    
    def restore_original_settings(self):
        """Restore original environment settings."""
        self._apply_normal_profile()
        logger.info("Restored original thermal settings")
    
    def get_thermal_info(self) -> Dict[str, Any]:
        """Get thermal management information."""
        return {
            "is_macos": self.is_macos,
            "is_apple_silicon": self.is_apple_silicon,
            "current_profile": self.current_profile.value,
            "applied_settings": {
                "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                "mps_high_watermark": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                "mps_low_watermark": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
                "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            },
            "recommendations": {
                "apple_silicon": "Use quiet profile for low heat and fan noise",
                "other_platforms": "Use normal profile for maximum performance"
            }
        }


# Global thermal manager instance
_thermal_manager = None


def get_thermal_manager() -> ThermalManager:
    """Get the global thermal manager instance."""
    global _thermal_manager
    if _thermal_manager is None:
        _thermal_manager = ThermalManager()
    return _thermal_manager


def apply_quiet_profile() -> Dict[str, Any]:
    """Apply quiet thermal profile."""
    manager = get_thermal_manager()
    return manager.apply_thermal_profile(HeatProfile.QUIET)


def apply_normal_profile() -> Dict[str, Any]:
    """Apply normal thermal profile."""
    manager = get_thermal_manager()
    return manager.apply_thermal_profile(HeatProfile.NORMAL)


def get_training_config(profile: HeatProfile = HeatProfile.QUIET) -> Dict[str, Any]:
    """Get training configuration for thermal profile."""
    manager = get_thermal_manager()
    return manager.get_training_config(profile)


if __name__ == "__main__":
    # Test the thermal manager
    manager = ThermalManager()
    
    print("Thermal Manager Test")
    print("=" * 40)
    
    info = manager.get_thermal_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nApplying quiet profile...")
    quiet_settings = manager.apply_thermal_profile(HeatProfile.QUIET)
    print(f"Applied settings: {quiet_settings}")
    
    print("\nTraining config (quiet):")
    config = manager.get_training_config(HeatProfile.QUIET)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nRestoring original settings...")
    manager.restore_original_settings()
