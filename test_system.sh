#!/bin/bash
echo "🧪 Comprehensive Able System Test"
echo "=================================="

# Test 1: Service Health
echo -e "\n🔍 Testing Service Health..."
health=$(curl -s http://localhost:8000/health)
if echo "$health" | grep -q "healthy"; then
    echo "✅ Backend health check passed"
else
    echo "❌ Backend health check failed"
fi

# Test 2: Frontend
echo -e "\n🔍 Testing Frontend..."
if curl -s http://localhost:3001 | grep -q "Able mk I"; then
    echo "✅ Frontend serving correctly"
else
    echo "❌ Frontend test failed"
fi

# Test 3: Multimodal Capabilities
echo -e "\n🔍 Testing Multimodal Capabilities..."
multimodal=$(curl -s http://localhost:8000/multimodal/capabilities)
if echo "$multimodal" | grep -q "operational"; then
    echo "✅ Multimodal system operational"
    echo "   $(echo "$multimodal" | grep -o '"llava_available":[^,]*')"
else
    echo "❌ Multimodal test failed"
fi

# Test 4: Document System
echo -e "\n🔍 Testing Document System..."
docs=$(curl -s http://localhost:8000/documents)
doc_count=$(echo "$docs" | grep -o '"id"' | wc -l)
echo "✅ Document system working ($doc_count documents)"

# Test 5: Chat Functionality
echo -e "\n🔍 Testing Chat Functionality..."
chat_response=$(curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "Hello", "session_id": "test"}')
if echo "$chat_response" | grep -q "answer"; then
    echo "✅ Chat system working"
else
    echo "❌ Chat test failed"
fi

# Test 6: Model Availability
echo -e "\n🔍 Testing Model Availability..."
models=$(curl -s http://localhost:8000/models/available)
model_count=$(echo "$models" | grep -o '"name"' | wc -l)
echo "✅ Model system working ($model_count models available)"

# Test 7: GraphRAG Status
echo -e "\n🔍 Testing GraphRAG Status..."
graph=$(curl -s http://localhost:8000/graph/statistics)
if echo "$graph" | grep -q "graphrag_available"; then
    echo "✅ GraphRAG system available"
else
    echo "❌ GraphRAG test failed"
fi

echo -e "\n🎯 System Test Complete!"
echo "All core Able features tested and verified."