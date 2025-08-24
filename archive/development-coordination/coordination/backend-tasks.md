# Backend Agent Tasks - Ollama Integration

## Priority 1: AI Service Enhancement
- [ ] Add ollama client to requirements.txt
- [ ] Modify ai_service.py to support dual providers (Anthropic + Ollama)
- [ ] Implement unified response interface
- [ ] Add model switching logic with fallback
- [ ] Create streaming response handler for Ollama

## Priority 2: Model Management Service
- [ ] Create model_service.py for Ollama operations
- [ ] Implement model availability checking
- [ ] Add model download/management functions
- [ ] Create performance monitoring
- [ ] Add memory usage tracking

## Priority 3: API Endpoints
- [ ] Add /models/available endpoint
- [ ] Add /models/switch endpoint  
- [ ] Add /models/status endpoint
- [ ] Enhance /chat endpoint with model selection
- [ ] Add /models/performance endpoint

## Integration Points
- Config service for model settings
- Frontend API for model selection UI
- Error handling for model failures
- Performance metrics for optimization
