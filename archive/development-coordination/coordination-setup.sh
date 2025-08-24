#!/bin/bash
# Avi Multi-Agent Setup Script

echo "🚀 Setting up Avi Multi-Agent Development Environment..."

# Create coordination directory structure
mkdir -p coordination/{agent-reports,integration-specs,shared-artifacts}

# Create task specification files
cat > coordination/backend-tasks.md << 'EOF'
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
EOF

cat > coordination/frontend-tasks.md << 'EOF'
# Frontend Agent Tasks - Model Selection UI

## Priority 1: Model Selection Interface
- [ ] Create ModelSelector.js component
- [ ] Add model status indicators (online/offline/loading)
- [ ] Implement model switching confirmation dialog
- [ ] Add performance metrics display
- [ ] Create model preference persistence

## Priority 2: Enhanced Chat Interface
- [ ] Add real-time model switching capability
- [ ] Implement response time indicators
- [ ] Add local vs cloud model visual indicators
- [ ] Create error state handling for model failures
- [ ] Add model performance feedback

## Priority 3: Settings Panel Enhancement
- [ ] Add model preference section
- [ ] Create performance tuning options
- [ ] Add fallback configuration interface
- [ ] Implement debug mode toggle
- [ ] Add model comparison features

## Integration Points
- API service updates for model management
- State management for model selection
- Error boundary for model failures
- Performance monitoring hooks
EOF

cat > coordination/devops-tasks.md << 'EOF'
# DevOps Agent Tasks - Configuration & Testing

## Priority 1: Environment Management
- [ ] Add Ollama installation verification
- [ ] Create model path configuration
- [ ] Set up performance monitoring
- [ ] Add resource usage tracking
- [ ] Create environment health checks

## Priority 2: Testing Framework
- [ ] Unit tests for ai_service model switching
- [ ] Integration tests for dual-provider setup
- [ ] Performance benchmarking suite
- [ ] Error scenario testing
- [ ] Model availability testing

## Priority 3: Documentation & Deployment
- [ ] Update setup instructions for Ollama
- [ ] Create troubleshooting guide
- [ ] Add performance tuning documentation
- [ ] Update API documentation
- [ ] Create deployment scripts

## Integration Points
- Backend configuration interface
- Frontend environment detection
- Performance monitoring dashboard
- Error reporting system
EOF

cat > coordination/ai-optimization-tasks.md << 'EOF'
# AI Optimization Agent Tasks - Intelligence Enhancement

## Priority 1: Model Performance Analysis
- [ ] Create response quality comparison framework
- [ ] Implement speed vs accuracy analysis
- [ ] Add memory optimization strategies
- [ ] Create batch processing optimization
- [ ] Develop model selection algorithms

## Priority 2: Search Enhancement
- [ ] Implement intelligent query routing (local vs cloud)
- [ ] Create result quality metrics
- [ ] Optimize embedding strategies for local models
- [ ] Enhance relevance scoring
- [ ] Add context window optimization

## Priority 3: Advanced Features
- [ ] GraphRAG compatibility with local models
- [ ] Multi-hop reasoning optimization
- [ ] Knowledge graph local processing
- [ ] Adaptive context management
- [ ] Response caching strategies

## Integration Points
- Backend model routing logic
- Frontend performance display
- Configuration tuning parameters
- Error handling for AI failures
EOF

# Create agent initialization commands
cat > coordination/start-agents.sh << 'EOF'
#!/bin/bash
echo "🤖 Initializing Avi Agent Panel..."

# Architecture Orchestrator (current session continues)
echo "Architecture Orchestrator: Active in current session"

# Backend Specialist
echo "Starting Backend Specialist..."
claude-code --session "avi-backend" "I am the Backend Specialist for the Avi project. My role is to implement server-side features, AI service integration, and database operations. I will read coordination/backend-tasks.md for my current priorities."

# Frontend Specialist  
echo "Starting Frontend Specialist..."
claude-code --session "avi-frontend" "I am the Frontend Specialist for the Avi project. My role is to create React components, manage UI state, and implement user experiences. I will read coordination/frontend-tasks.md for my current priorities."

# DevOps & Configuration
echo "Starting DevOps Specialist..."
claude-code --session "avi-devops" "I am the DevOps & Configuration specialist for the Avi project. My role is to manage environment setup, testing, and deployment. I will read coordination/devops-tasks.md for my current priorities."

# AI & Search Optimizer
echo "Starting AI Optimizer..."
claude-code --session "avi-ai" "I am the AI & Search Optimizer for the Avi project. My role is to enhance machine learning performance and search capabilities. I will read coordination/ai-optimization-tasks.md for my current priorities."

echo "✅ All agents initialized! Check individual sessions for task progress."
EOF

# Create quick command reference
cat > coordination/agent-commands.md << 'EOF'
# Quick Agent Command Reference

## Backend Agent Commands
```bash
# Start Ollama integration
claude-code --session "avi-backend" --file backend/services/ai_service.py "Begin Ollama integration following backend-tasks.md Priority 1"

# Create model service
claude-code --session "avi-backend" --create backend/services/model_service.py "Create Ollama model management service"

# Update requirements
claude-code --session "avi-backend" --file requirements.txt "Add ollama client dependency"
```

## Frontend Agent Commands  
```bash
# Create model selector
claude-code --session "avi-frontend" --create frontend/src/components/ModelSelector.js "Create model selection component following frontend-tasks.md"

# Update API client
claude-code --session "avi-frontend" --file frontend/src/services/api.js "Add model management API calls"

# Enhance chat interface
claude-code --session "avi-frontend" --file frontend/src/components/ChatInterface.js "Add model switching capability"
```

## DevOps Agent Commands
```bash
# Update configuration
claude-code --session "avi-devops" --file backend/config.py "Add Ollama configuration management"

# Create tests
claude-code --session "avi-devops" --create tests/test_model_switching.py "Create model switching test suite"

# Update documentation
claude-code --session "avi-devops" --file README.md "Add Ollama setup and usage instructions"
```

## AI Optimizer Commands
```bash
# Performance analysis
claude-code --session "avi-ai" --create backend/services/performance_monitor.py "Create model performance monitoring"

# Query routing
claude-code --session "avi-ai" --file backend/services/ai_service.py "Implement intelligent model routing logic"

# Optimization
claude-code --session "avi-ai" --project "Analyze and optimize search performance for local models"
```

## Coordination Commands
```bash
# Daily sync
claude-code --session "avi-backend" "Report current task status to coordination/agent-reports/backend-daily.md"
claude-code --session "avi-frontend" "Report current task status to coordination/agent-reports/frontend-daily.md"
claude-code --session "avi-devops" "Report current task status to coordination/agent-reports/devops-daily.md"
claude-code --session "avi-ai" "Report current task status to coordination/agent-reports/ai-daily.md"

# Integration review
claude-code --project "Review all agent progress and identify integration issues"
```
EOF

# Make scripts executable
chmod +x coordination/start-agents.sh

echo "✅ Agent coordination system ready!"
echo "📁 Task files created in coordination/"
echo "🚀 Run 'coordination/start-agents.sh' to initialize all agents"
echo "📖 See 'coordination/agent-commands.md' for quick command reference"