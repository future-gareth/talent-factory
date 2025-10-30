"""
Training router for Talent Factory backend selection.

Routes training requests to the appropriate backend (MLX, CUDA, CPU) based on
platform detection and user preferences.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from pathlib import Path

from platform_detector import pick_training_backend, get_backend_info, print_backend_banner

logger = logging.getLogger(__name__)


class TrainingRouter:
    """Routes training requests to the appropriate backend."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.backend_info = get_backend_info()
        self.recommended_backend = self.backend_info["recommended_backend"]
        
        # Backend modules (lazy loading)
        self._mlx_engine = None
        self._cuda_engine = None
        self._cpu_engine = None
        
        logger.info(f"Training router initialized. Recommended backend: {self.recommended_backend}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        return self.backend_info
    
    def print_banner(self):
        """Print platform detection banner."""
        print_backend_banner()
    
    def _get_mlx_engine(self):
        """Get MLX training engine (lazy loading)."""
        if self._mlx_engine is None:
            try:
                from mlx_engine import MLXTrainingEngine
                self._mlx_engine = MLXTrainingEngine(self.base_dir)
                logger.info("MLX training engine loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import MLX engine: {e}")
                raise RuntimeError("MLX backend not available. Install with: pip install mlx mlx-lm")
        return self._mlx_engine
    
    def _get_cuda_engine(self):
        """Get CUDA training engine (lazy loading)."""
        if self._cuda_engine is None:
            try:
                from training_engine import TrainingEngine
                self._cuda_engine = TrainingEngine(self.base_dir)
                logger.info("CUDA training engine loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import CUDA engine: {e}")
                raise RuntimeError("CUDA backend not available")
        return self._cuda_engine
    
    def _get_cpu_engine(self):
        """Get CPU training engine (lazy loading)."""
        if self._cpu_engine is None:
            try:
                from training_engine import TrainingEngine
                self._cpu_engine = TrainingEngine(self.base_dir)
                logger.info("CPU training engine loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import CPU engine: {e}")
                raise RuntimeError("CPU backend not available")
        return self._cpu_engine
    
    def get_engine(self, backend: Optional[str] = None):
        """
        Get the appropriate training engine.
        
        Args:
            backend: Override backend selection
            
        Returns:
            Training engine instance
        """
        selected_backend = backend or self.recommended_backend
        
        logger.info(f"Getting training engine for backend: {selected_backend}")
        
        if selected_backend == "mlx":
            return self._get_mlx_engine()
        elif selected_backend == "cuda":
            return self._get_cuda_engine()
        elif selected_backend == "cpu":
            return self._get_cpu_engine()
        else:
            raise ValueError(f"Unknown backend: {selected_backend}")
    
    async def train_model(
        self,
        model_id: str,
        dataset_path: str,
        output_dir: str,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        backend: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model using the appropriate backend.
        
        Args:
            model_id: Model identifier
            dataset_path: Path to training dataset
            output_dir: Output directory for trained model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            backend: Override backend selection
            progress_callback: Progress callback function
            **kwargs: Additional arguments
            
        Returns:
            Training result dictionary
        """
        selected_backend = backend or self.recommended_backend
        
        logger.info(f"Starting training with backend: {selected_backend}")
        logger.info(f"Model: {model_id}, Dataset: {dataset_path}, Output: {output_dir}")
        
        # Get the appropriate engine
        engine = self.get_engine(selected_backend)
        
        # Call the engine's train_model method
        result = await engine.train_model(
            model_id=model_id,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_callback=progress_callback,
            **kwargs
        )
        
        # Add backend information to result
        result["backend_used"] = selected_backend
        result["backend_info"] = self.backend_info
        
        logger.info(f"Training completed with backend: {selected_backend}")
        return result
    
    def get_compatible_models(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Get list of models compatible with the specified backend.
        
        Args:
            backend: Override backend selection
            
        Returns:
            Dictionary of compatible models
        """
        selected_backend = backend or self.recommended_backend
        
        if selected_backend == "mlx":
            # MLX-compatible models
            return {
                "models": [
                    {
                        "id": "mlx-mistral-7b",
                        "name": "Mistral 7B (MLX)",
                        "description": "Mistral 7B optimized for Apple Silicon",
                        "size": "7B",
                        "backend": "mlx"
                    },
                    {
                        "id": "mlx-llama-7b",
                        "name": "Llama 7B (MLX)",
                        "description": "Llama 7B optimized for Apple Silicon",
                        "size": "7B",
                        "backend": "mlx"
                    }
                ]
            }
        else:
            # CUDA/CPU compatible models (existing)
            return {
                "models": [
                    {
                        "id": "microsoft/DialoGPT-medium",
                        "name": "DialoGPT Medium",
                        "description": "Conversational AI model",
                        "size": "345M",
                        "backend": "cuda/cpu"
                    },
                    {
                        "id": "microsoft/DialoGPT-large",
                        "name": "DialoGPT Large",
                        "description": "Large conversational AI model",
                        "size": "774M",
                        "backend": "cuda/cpu"
                    },
                    {
                        "id": "distilbert-base-uncased",
                        "name": "DistilBERT",
                        "description": "Distilled BERT model",
                        "size": "66M",
                        "backend": "cuda/cpu"
                    },
                    {
                        "id": "gpt2",
                        "name": "GPT-2",
                        "description": "GPT-2 base model",
                        "size": "117M",
                        "backend": "cuda/cpu"
                    }
                ]
            }


# Global router instance
_router_instance = None


def get_training_router(base_dir: Path) -> TrainingRouter:
    """Get the global training router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = TrainingRouter(base_dir)
    return _router_instance


if __name__ == "__main__":
    # Test the training router
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        router = TrainingRouter(Path(temp_dir))
        router.print_banner()
        
        print("\nBackend Info:")
        info = router.get_backend_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\nCompatible Models:")
        models = router.get_compatible_models()
        for model in models["models"]:
            print(f"  {model['id']}: {model['name']} ({model['backend']})")
