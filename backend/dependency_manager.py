"""
Dependency manager for Talent Factory.

Automatically installs required dependencies based on platform detection,
with user permission prompts when needed.
"""

import subprocess
import sys
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages automatic dependency installation."""
    
    def __init__(self):
        self.is_apple_silicon = self._detect_apple_silicon()
        self.installed_deps = set()
        
        logger.info(f"Dependency manager initialized. Apple Silicon: {self.is_apple_silicon}")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            import platform
            if platform.system() != "Darwin":
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
    
    def _check_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed."""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except ImportError:
            return False
    
    def _install_package(self, package_name: str, pip_name: Optional[str] = None) -> bool:
        """
        Install a package using pip.
        
        Args:
            package_name: Python package name to check
            pip_name: pip package name (if different from package_name)
            
        Returns:
            bool: True if installation successful
        """
        pip_package = pip_name or package_name
        
        try:
            logger.info(f"Installing {pip_package}...")
            
            # Use the same Python executable
            python_exe = sys.executable
            
            # Install the package
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", pip_package],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully installed {pip_package}")
                self.installed_deps.add(package_name)
                return True
            else:
                logger.error(f"Failed to install {pip_package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {pip_package}")
            return False
        except Exception as e:
            logger.error(f"Error installing {pip_package}: {e}")
            return False
    
    def _prompt_user_permission(self, packages: List[str]) -> bool:
        """
        Prompt user for permission to install packages.
        
        Args:
            packages: List of packages to install
            
        Returns:
            bool: True if user grants permission
        """
        print("\n" + "=" * 60)
        print("TALENT FACTORY - DEPENDENCY INSTALLATION")
        print("=" * 60)
        print("Apple Silicon detected! To enable optimal performance,")
        print("Talent Factory needs to install additional dependencies.")
        print()
        print("Packages to install:")
        for package in packages:
            print(f"  • {package}")
        print()
        print("This will improve training performance and reduce heat generation.")
        print("=" * 60)
        
        while True:
            response = input("Install these dependencies? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def ensure_apple_silicon_deps(self, auto_install: bool = False) -> Dict[str, Any]:
        """
        Ensure Apple Silicon dependencies are installed.
        
        Args:
            auto_install: If True, install without prompting
            
        Returns:
            Dict with installation results
        """
        if not self.is_apple_silicon:
            return {
                "required": False,
                "reason": "Not running on Apple Silicon"
            }
        
        # Check required packages
        required_packages = {
            "mlx": "mlx",
            "mlx.core": "mlx-lm"
        }
        
        missing_packages = []
        for package, pip_name in required_packages.items():
            if not self._check_package_installed(package):
                missing_packages.append(pip_name)
        
        if not missing_packages:
            return {
                "required": True,
                "installed": True,
                "packages": list(required_packages.keys())
            }
        
        # Ask for permission if not auto-installing
        if not auto_install:
            if not self._prompt_user_permission(missing_packages):
                return {
                    "required": True,
                    "installed": False,
                    "reason": "User declined installation",
                    "missing_packages": missing_packages
                }
        
        # Install missing packages
        installation_results = {}
        all_successful = True
        
        for package, pip_name in required_packages.items():
            if package not in self.installed_deps and not self._check_package_installed(package):
                success = self._install_package(package, pip_name)
                installation_results[pip_name] = success
                if not success:
                    all_successful = False
        
        return {
            "required": True,
            "installed": all_successful,
            "packages": list(required_packages.keys()),
            "installation_results": installation_results,
            "missing_packages": missing_packages if not all_successful else []
        }
    
    def ensure_general_deps(self) -> Dict[str, Any]:
        """
        Ensure general dependencies are installed.
        
        Returns:
            Dict with installation results
        """
        # Check for general ML dependencies
        general_packages = {
            "torch": "torch",
            "transformers": "transformers",
            "datasets": "datasets",
            "peft": "peft",
            "scikit-learn": "scikit-learn",
            "pandas": "pandas",
            "numpy": "numpy"
        }
        
        missing_packages = []
        for package, pip_name in general_packages.items():
            if not self._check_package_installed(package):
                missing_packages.append(pip_name)
        
        if not missing_packages:
            return {
                "required": True,
                "installed": True,
                "packages": list(general_packages.keys())
            }
        
        # Auto-install general dependencies
        installation_results = {}
        all_successful = True
        
        for package, pip_name in general_packages.items():
            if package not in self.installed_deps and not self._check_package_installed(package):
                success = self._install_package(package, pip_name)
                installation_results[pip_name] = success
                if not success:
                    all_successful = False
        
        return {
            "required": True,
            "installed": all_successful,
            "packages": list(general_packages.keys()),
            "installation_results": installation_results,
            "missing_packages": missing_packages if not all_successful else []
        }
    
    def get_installation_status(self) -> Dict[str, Any]:
        """Get current installation status."""
        status = {
            "is_apple_silicon": self.is_apple_silicon,
            "installed_deps": list(self.installed_deps),
            "mlx_available": self._check_package_installed("mlx"),
            "torch_available": self._check_package_installed("torch"),
            "transformers_available": self._check_package_installed("transformers"),
        }
        
        if self.is_apple_silicon:
            status["apple_silicon_deps"] = self.ensure_apple_silicon_deps(auto_install=True)
        
        status["general_deps"] = self.ensure_general_deps()
        
        return status


# Global dependency manager instance
_dependency_manager = None


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


def ensure_dependencies(auto_install: bool = False) -> Dict[str, Any]:
    """
    Ensure all required dependencies are installed.
    
    Args:
        auto_install: If True, install without prompting
        
    Returns:
        Dict with installation results
    """
    manager = get_dependency_manager()
    
    # Ensure general dependencies first
    general_result = manager.ensure_general_deps()
    
    # Ensure Apple Silicon dependencies if needed
    apple_result = manager.ensure_apple_silicon_deps(auto_install=auto_install)
    
    return {
        "general": general_result,
        "apple_silicon": apple_result,
        "overall_success": general_result.get("installed", False) and apple_result.get("installed", True)
    }


if __name__ == "__main__":
    # Test the dependency manager
    manager = DependencyManager()
    
    print("Dependency Manager Test")
    print("=" * 40)
    
    status = manager.get_installation_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\nEnsuring dependencies...")
    result = ensure_dependencies(auto_install=False)
    print(f"Installation result: {result}")
