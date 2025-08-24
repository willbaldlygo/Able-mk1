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
