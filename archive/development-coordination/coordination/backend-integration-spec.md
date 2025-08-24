# Backend Integration Specification - Ollama

## Immediate Tasks for Backend Agent

### 1. Add Ollama Dependencies
```bash
# Add to requirements.txt
ollama>=0.1.0
```

### 2. AI Service Enhancement Pattern
```python
# ai_service.py enhancement approach
class AIService:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.ollama_client = ollama.Client()  # New Ollama client
        self.current_provider = config.ai_provider  # "anthropic" or "ollama"
        self.current_model = config.current_model
    
    async def generate_response_unified(self, question: str, sources: List[SourceInfo]) -> ChatResponse:
        """Unified response interface supporting both providers."""
        if self.current_provider == "ollama":
            return await self._generate_ollama_response(question, sources)
        else:
            return await self._generate_anthropic_response(question, sources)
```

### 3. Model Management Service
```python
# New file: backend/services/model_service.py
class ModelService:
    def __init__(self):
        self.ollama_client = ollama.Client()
    
    async def get_available_models(self):
        """Get list of available local and remote models."""
        pass
    
    async def switch_model(self, provider: str, model_name: str):
        """Switch active model and provider."""
        pass
```

### 4. API Endpoints to Add
- `GET /models/available` - List available models
- `POST /models/switch` - Switch active model
- `GET /models/status` - Current model status
- Enhanced `/chat` endpoint with model parameter

## Integration Points
- Config updates needed in `backend/config.py`
- Frontend API calls in `frontend/src/services/api.js`
- Model performance monitoring hooks
- Error handling for local model failures