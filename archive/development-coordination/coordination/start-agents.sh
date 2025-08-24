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
