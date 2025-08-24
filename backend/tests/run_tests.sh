#!/bin/bash
# Test runner script for Able backend tests

echo "🧪 Running Able Backend Integration Tests..."

# Check if we're in the correct directory
if [ ! -f "config.py" ]; then
    echo "❌ Please run from backend/ directory"
    exit 1
fi

# Install test dependencies if needed
echo "Installing test dependencies..."
pip install -r tests/requirements-test.txt

# Run the integration test directly (works without pytest)
echo "Running manual integration test..."
cd tests
python test_ollama_integration.py

# Run with pytest if available
echo ""
echo "Running with pytest..."
cd ..
python -m pytest tests/test_ollama_integration.py -v -s

echo "✅ Tests completed!"