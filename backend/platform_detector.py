"""
Platform detection for Talent Factory backend selection.

Detects Apple Silicon vs other systems and recommends the optimal training backend.
"""

import platform
import subprocess
import sys
from typing import Dict, Any, Optional
from pathlib import Path


def detect_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon (M1/M2/M3/etc).
    
    Returns:
        bool: True if running on Apple Silicon, False otherwise
    """
    try:
        # Check if we're on macOS
        if platform.system() != "Darwin":
            return False
        
        # Check architecture
        arch = platform.machine()
        if arch != "arm64":
            return False
        
        # Additional check using system_profiler
        try:
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
                
                # Check for Apple Silicon chip types
                apple_silicon_chips = ["Apple M1", "Apple M2", "Apple M3", "Apple M4"]
                return any(chip in chip_type for chip in apple_silicon_chips)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        # Fallback: if we're on arm64 macOS, assume Apple Silicon
        return True
        
    except Exception as e:
        print(f"Warning: Error detecting Apple Silicon: {e}")
        return False


def detect_cuda_availability() -> bool:
    """
    Detect if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_mlx_availability() -> bool:
    """
    Detect if MLX is available.
    
    Returns:
        bool: True if MLX is available, False otherwise
    """
    try:
        import mlx
        import mlx.core as mx
        # Test basic MLX functionality
        mx.array([1, 2, 3])
        return True
    except ImportError:
        return False
    except Exception:
        return False


def ensure_mlx_availability() -> bool:
    """
    Ensure MLX is available, installing if needed.
    
    Returns:
        bool: True if MLX is available after installation attempt
    """
    if detect_mlx_availability():
        return True
    
    # Try to install MLX dependencies
    try:
        from dependency_manager import ensure_dependencies
        result = ensure_dependencies(auto_install=False)
        
        if result.get("overall_success", False):
            return detect_mlx_availability()
        
        return False
    except Exception as e:
        logger.warning(f"Failed to ensure MLX availability: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict containing system information
    """
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "is_apple_silicon": detect_apple_silicon(),
        "cuda_available": detect_cuda_availability(),
        "mlx_available": detect_mlx_availability(),
    }
    
    # Add macOS-specific info
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sw_vers"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'ProductVersion:' in line:
                        info["macos_version"] = line.split(':')[1].strip()
                    elif 'ProductName:' in line:
                        info["macos_name"] = line.split(':')[1].strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
    
    return info


def pick_training_backend(force_backend: Optional[str] = None, auto_install: bool = True) -> str:
    """
    Pick the optimal training backend based on platform detection.
    
    Args:
        force_backend: Override auto-detection with specific backend
        auto_install: If True, automatically install missing dependencies
        
    Returns:
        str: Backend name ('mlx', 'cuda', or 'cpu')
    """
    if force_backend:
        return force_backend
    
    # Check for Apple Silicon first
    if detect_apple_silicon():
        if detect_mlx_availability():
            return "mlx"
        elif auto_install:
            # Try to install MLX dependencies
            print("Apple Silicon detected. Installing MLX dependencies for optimal performance...")
            if ensure_mlx_availability():
                print("✅ MLX installed successfully!")
                return "mlx"
            else:
                print("⚠️  MLX installation failed. Falling back to CPU.")
                return "cpu"
        else:
            print("Warning: Apple Silicon detected but MLX not available. Falling back to CPU.")
            return "cpu"
    
    # Check for CUDA on other platforms
    if detect_cuda_availability():
        return "cuda"
    
    # Default to CPU
    return "cpu"


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about available backends and recommendations.
    
    Returns:
        Dict containing backend information
    """
    system_info = get_system_info()
    recommended_backend = pick_training_backend()
    
    return {
        "system": system_info,
        "recommended_backend": recommended_backend,
        "available_backends": {
            "mlx": system_info["mlx_available"],
            "cuda": system_info["cuda_available"],
            "cpu": True,  # CPU is always available
        },
        "backend_recommendations": {
            "apple_silicon": "mlx" if system_info["mlx_available"] else "cpu",
            "other_platforms": "cuda" if system_info["cuda_available"] else "cpu",
        }
    }


def print_backend_banner():
    """Print a banner showing detected platform and backend selection."""
    info = get_backend_info()
    system = info["system"]
    backend = info["recommended_backend"]
    
    print("=" * 60)
    print("TALENT FACTORY - PLATFORM DETECTION")
    print("=" * 60)
    print(f"Platform: {system['platform']} {system['architecture']}")
    
    if system["is_apple_silicon"]:
        print("✅ Apple Silicon detected")
        if system["mlx_available"]:
            print("✅ MLX available - Using native Apple Silicon backend")
        else:
            print("⚠️  MLX not available - Install with: pip install mlx mlx-lm")
    else:
        print("ℹ️  Non-Apple Silicon platform")
        if system["cuda_available"]:
            print("✅ CUDA available - Using GPU acceleration")
        else:
            print("ℹ️  CUDA not available - Using CPU backend")
    
    print(f"Selected Backend: {backend.upper()}")
    print("=" * 60)


if __name__ == "__main__":
    # Test the platform detection
    print_backend_banner()
    
    info = get_backend_info()
    print("\nDetailed System Info:")
    for key, value in info["system"].items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommended Backend: {info['recommended_backend']}")
    print(f"Available Backends: {info['available_backends']}")
