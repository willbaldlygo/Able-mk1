"""Model management service for Able with dual AI provider support."""
import ollama
import anthropic
import asyncio
import json
import time
import psutil
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from datetime import datetime
import logging

from config import config

logger = logging.getLogger(__name__)

class ModelService:
    """Manages AI model switching and monitoring for both Anthropic and Ollama providers."""
    
    def __init__(self):
        self.ollama_client = None
        self.anthropic_client = None
        self.current_provider = config.default_provider
        self.current_model = self._get_default_model()
        self._performance_metrics = {}
        self.initialization_status = {
            "ollama_connected": False,
            "anthropic_configured": bool(config.anthropic_api_key),
            "last_check": datetime.now()
        }
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all available AI providers."""
        # Initialize Ollama if enabled
        if config.ollama_enabled:
            self._initialize_ollama()
        
        # Initialize Anthropic if configured
        if config.anthropic_api_key:
            self._initialize_anthropic()
    
    def _initialize_ollama(self) -> None:
        """Initialize Ollama client and check connectivity."""
        try:
            # Initialize with longer timeout for large models like gpt-oss:20b
            self.ollama_client = ollama.Client(
                host=config.ollama_host,
                timeout=60.0  # 60 second timeout for large models
            )
            # Test connection with a simple list call
            response = self.ollama_client.list()
            model_count = len(response.models) if hasattr(response, 'models') else len(response.get('models', []))
            self.initialization_status["ollama_connected"] = True
            logger.info(f"Ollama connected successfully with {model_count} models")
        except Exception as e:
            self.initialization_status["ollama_connected"] = False
            logger.warning(f"Ollama initialization failed: {str(e)}")
            self.ollama_client = None
    
    def _initialize_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.warning(f"Anthropic initialization failed: {str(e)}")
            self.anthropic_client = None
    
    def _get_default_model(self) -> str:
        """Get default model based on current provider."""
        if self.current_provider == "ollama":
            return config.ollama_model
        return config.claude_model
        
    def test_connection(self) -> bool:
        """Test Ollama server connection."""
        try:
            if self.ollama_client:
                response = self.ollama_client.list()
                return True
            return False
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            return False
    
    def test_anthropic_connection(self) -> bool:
        """Test Anthropic API connection."""
        try:
            if not self.anthropic_client:
                return False
            response = self.anthropic_client.messages.create(
                model=config.claude_model,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": "Hello"
                }]
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic connection failed: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get list of available models from all providers."""
        models = {
            "anthropic": [
                {
                    "name": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "provider": "anthropic",
                    "available": bool(config.anthropic_api_key),
                    "type": "remote"
                },
                {
                    "name": "claude-3-haiku-20240307", 
                    "display_name": "Claude 3 Haiku",
                    "provider": "anthropic",
                    "available": bool(config.anthropic_api_key),
                    "type": "remote"
                }
            ],
            "ollama": []
        }
        
        # Get Ollama models if available
        if self.ollama_client and self.initialization_status["ollama_connected"]:
            try:
                ollama_response = self.ollama_client.list()
                # Handle both dict and object response formats
                if hasattr(ollama_response, 'models'):
                    ollama_models = ollama_response.models
                elif isinstance(ollama_response, dict):
                    ollama_models = ollama_response.get('models', [])
                else:
                    ollama_models = []
                
                for model in ollama_models:
                    # Handle both dict and object formats
                    model_name = model.model if hasattr(model, 'model') else model.get('name', 'Unknown')
                    model_size = model.size if hasattr(model, 'size') else model.get('size', 0)
                    model_modified = model.modified_at if hasattr(model, 'modified_at') else model.get('modified_at', '')
                    
                    models["ollama"].append({
                        "name": model_name,
                        "display_name": model_name.replace(':', ' '),
                        "provider": "ollama", 
                        "available": True,
                        "type": "local",
                        "size": model_size,
                        "modified": str(model_modified)
                    })
            except Exception as e:
                logger.error(f"Error fetching Ollama models: {str(e)}")
        
        return models
    
    def check_model_availability(self, model_name: str, provider: str = "ollama") -> bool:
        """Check if a specific model is available for the given provider."""
        models = self.get_available_models()
        if provider in models:
            return any(model['name'] == model_name and model['available'] for model in models[provider])
        return False
    
    def start_performance_tracking(self, request_id: str) -> None:
        """Start tracking performance for a request."""
        self._performance_metrics[request_id] = {
            "provider": self.current_provider,
            "model": self.current_model,
            "start_time": time.time(),
            "status": "running"
        }
    
    def end_performance_tracking(self, request_id: str, success: bool = True, token_count: int = 0) -> None:
        """End performance tracking and record metrics."""
        if request_id in self._performance_metrics:
            stats = self._performance_metrics[request_id]
            stats.update({
                "end_time": time.time(),
                "duration": time.time() - stats["start_time"],
                "success": success,
                "token_count": token_count,
                "status": "completed" if success else "failed"
            })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary across all tracked requests."""
        completed_requests = [r for r in self._performance_metrics.values() if r.get("status") == "completed"]
        
        if not completed_requests:
            return {"no_data": True}
        
        # Calculate averages by provider
        provider_stats = {}
        for req in completed_requests:
            provider = req["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "requests": 0,
                    "total_duration": 0,
                    "total_tokens": 0,
                    "successes": 0
                }
            
            stats = provider_stats[provider]
            stats["requests"] += 1
            stats["total_duration"] += req["duration"]
            stats["total_tokens"] += req.get("token_count", 0)
            if req["success"]:
                stats["successes"] += 1
        
        # Calculate averages
        for provider, stats in provider_stats.items():
            if stats["requests"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["requests"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["requests"]
                stats["success_rate"] = stats["successes"] / stats["requests"]
        
        return {
            "total_requests": len(completed_requests),
            "by_provider": provider_stats,
            "current_provider": self.current_provider
        }
    
    def download_model(self, model_name: str) -> Dict:
        """Download/pull a model from Ollama registry."""
        try:
            logger.info(f"Starting download of model: {model_name}")
            self.ollama_client.pull(model_name)
            return {
                'success': True,
                'message': f'Model {model_name} downloaded successfully',
                'model_name': model_name
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to download model {model_name}: {str(e)}',
                'model_name': model_name
            }
    
    def delete_model(self, model_name: str) -> Dict:
        """Delete a model from local storage."""
        try:
            self.ollama_client.delete(model_name)
            return {
                'success': True,
                'message': f'Model {model_name} deleted successfully',
                'model_name': model_name
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to delete model {model_name}: {str(e)}',
                'model_name': model_name
            }
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        try:
            if not self.ollama_client:
                return None
            response = self.ollama_client.show(model_name)
            return {
                'name': model_name,
                'modelfile': response.get('modelfile', ''),
                'parameters': response.get('parameters', ''),
                'template': response.get('template', ''),
                'details': response.get('details', {}),
                'model_info': response.get('model_info', {})
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    async def switch_model(self, provider: str, model_name: str) -> Dict:
        """Switch active model and provider."""
        if provider not in ["anthropic", "ollama"]:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Verify model availability
        available_models = self.get_available_models()
        provider_models = [m['name'] for m in available_models[provider]]
        
        if model_name not in provider_models:
            raise ValueError(f"Model {model_name} not available for provider {provider}")
        
        # Check provider readiness
        if provider == "anthropic" and not config.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        if provider == "ollama" and not self.initialization_status["ollama_connected"]:
            # Try to reconnect
            self._initialize_ollama()
            if not self.initialization_status["ollama_connected"]:
                raise ValueError("Ollama not available")
        
        # Perform switch
        old_provider = self.current_provider
        old_model = self.current_model
        
        self.current_provider = provider
        self.current_model = model_name
        
        # Log switch
        logger.info(f"Model switched from {old_provider}:{old_model} to {provider}:{model_name}")
        
        return {
            "success": True,
            "previous": {"provider": old_provider, "model": old_model},
            "current": {"provider": provider, "model": model_name},
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_model_status(self) -> Dict:
        """Get current model status and health information."""
        return {
            "current": {
                "provider": self.current_provider,
                "model": self.current_model,
                "active": True
            },
            "providers": {
                "anthropic": {
                    "available": bool(config.anthropic_api_key),
                    "configured": bool(config.anthropic_api_key),
                    "connection_test": self.test_anthropic_connection()
                },
                "ollama": {
                    "available": self.initialization_status["ollama_connected"],
                    "configured": config.ollama_enabled,
                    "host": config.ollama_host,
                    "connection_test": self.test_connection()
                }
            },
            "performance": self.get_performance_summary(),
            "last_check": self.initialization_status["last_check"].isoformat()
        }
    
    async def health_check(self) -> Dict:
        """Comprehensive health check for all AI providers."""
        health_status = {
            "overall": "healthy",
            "providers": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check Anthropic
        if config.anthropic_api_key:
            try:
                anthropic_healthy = self.test_anthropic_connection()
                health_status["providers"]["anthropic"] = {
                    "status": "healthy" if anthropic_healthy else "unhealthy",
                    "configured": True,
                    "available": anthropic_healthy
                }
            except Exception as e:
                health_status["providers"]["anthropic"] = {
                    "status": "error",
                    "configured": True,
                    "available": False,
                    "error": str(e)
                }
        else:
            health_status["providers"]["anthropic"] = {
                "status": "not_configured",
                "configured": False,
                "available": False
            }
        
        # Check Ollama
        ollama_status = "not_configured"
        if config.ollama_enabled:
            if self.ollama_client and self.initialization_status["ollama_connected"]:
                try:
                    # Test with a simple list call
                    models = self.ollama_client.list()
                    ollama_status = "healthy"
                    health_status["providers"]["ollama"] = {
                        "status": "healthy",
                        "configured": True,
                        "available": True,
                        "models_count": len(models.get('models', []))
                    }
                except Exception as e:
                    ollama_status = "unhealthy"
                    health_status["providers"]["ollama"] = {
                        "status": "unhealthy",
                        "configured": True,
                        "available": False,
                        "error": str(e)
                    }
            else:
                ollama_status = "disconnected"
                health_status["providers"]["ollama"] = {
                    "status": "disconnected",
                    "configured": True,
                    "available": False
                }
        else:
            health_status["providers"]["ollama"] = {
                "status": "not_configured",
                "configured": False,
                "available": False
            }
        
        # Set overall status
        provider_statuses = [p["status"] for p in health_status["providers"].values()]
        if any(s == "healthy" for s in provider_statuses):
            health_status["overall"] = "healthy"
        elif any(s in ["unhealthy", "error", "disconnected"] for s in provider_statuses):
            health_status["overall"] = "degraded"
        else:
            health_status["overall"] = "unhealthy"
        
        # Update last check time
        self.initialization_status["last_check"] = datetime.now()
        
        return health_status
    
    def generate_response(self, model_name: str, prompt: str, stream: bool = False) -> Dict:
        """Generate response using specified Ollama model."""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            response = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                stream=stream,
                options={
                    'temperature': config.temperature,
                    'num_ctx': config.max_tokens * 2,  # Context window
                    'num_predict': config.max_tokens
                }
            )
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Record performance metrics
            self._record_performance(
                model_name, 
                end_time - start_time,
                memory_after - memory_before
            )
            
            if stream:
                return {'stream': response, 'success': True}
            else:
                return {
                    'response': response['response'],
                    'success': True,
                    'model': model_name,
                    'created_at': response.get('created_at', ''),
                    'done': response.get('done', True),
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_count': response.get('eval_count', 0),
                    'eval_duration': response.get('eval_duration', 0)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': model_name
            }
    
    async def generate_stream_response(self, model_name: str, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response using specified Ollama model."""
        try:
            if not self.ollama_client:
                yield "Error: Ollama client not initialized"
                return
                
            stream = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': config.temperature,
                    'num_ctx': config.max_tokens * 2,
                    'num_predict': config.max_tokens
                }
            )
            
            for chunk in stream:
                if chunk.get('response'):
                    yield chunk['response']
                if chunk.get('done', False):
                    break
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _record_performance(self, model_name: str, duration: float, memory_usage: float):
        """Record performance metrics for a model."""
        if model_name not in self._performance_metrics:
            self._performance_metrics[model_name] = {
                'total_requests': 0,
                'total_duration': 0,
                'total_memory': 0,
                'last_used': None
            }
        
        metrics = self._performance_metrics[model_name]
        metrics['total_requests'] += 1
        metrics['total_duration'] += duration
        metrics['total_memory'] += memory_usage
        metrics['last_used'] = datetime.now().isoformat()
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all models."""
        metrics = {}
        for model_name, data in self._performance_metrics.items():
            if data['total_requests'] > 0:
                metrics[model_name] = {
                    'total_requests': data['total_requests'],
                    'avg_response_time': data['total_duration'] / data['total_requests'],
                    'avg_memory_usage': data['total_memory'] / data['total_requests'],
                    'last_used': data['last_used']
                }
        return metrics
    
    def get_system_status(self) -> Dict:
        """Get system status and resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total / 1024 / 1024 / 1024,  # GB
                    'available': memory.available / 1024 / 1024 / 1024,  # GB
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total / 1024 / 1024 / 1024,  # GB
                    'free': disk.free / 1024 / 1024 / 1024,  # GB
                    'percent': (disk.used / disk.total) * 100
                },
                'ollama_connection': self.test_connection()
            }
        except Exception as e:
            return {
                'error': f'Failed to get system status: {str(e)}'
            }


# Global instance
model_service = ModelService()