#!/usr/bin/env python3
"""
Simple multimodal functionality test for Able
"""
import requests
import json

def test_multimodal_capabilities():
    """Test multimodal capabilities endpoint"""
    try:
        response = requests.get("http://localhost:8000/multimodal/capabilities")
        if response.status_code == 200:
            data = response.json()
            print("✅ Multimodal capabilities:", data.get("status"))
            print(f"   LLava available: {data.get('llava_available')}")
            return True
        else:
            print("❌ Multimodal capabilities test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing multimodal: {e}")
        return False

def test_chat_functionality():
    """Test basic chat functionality"""
    try:
        payload = {
            "question": "What is machine learning?",
            "session_id": "test-session"
        }
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            answer_length = len(data.get("answer", ""))
            sources_count = len(data.get("sources", []))
            print(f"✅ Chat functionality working (response: {answer_length} chars, sources: {sources_count})")
            return True
        else:
            print("❌ Chat functionality test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing chat: {e}")
        return False

def test_document_listing():
    """Test document listing"""
    try:
        response = requests.get("http://localhost:8000/documents")
        if response.status_code == 200:
            docs = response.json()
            print(f"✅ Document listing working ({len(docs)} documents)")
            return True
        else:
            print("❌ Document listing test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing documents: {e}")
        return False

def main():
    print("🧪 Testing Able Multimodal System")
    print("=" * 50)
    
    tests = [
        ("Multimodal Capabilities", test_multimodal_capabilities),
        ("Chat Functionality", test_chat_functionality),
        ("Document Listing", test_document_listing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Able system is fully functional.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()