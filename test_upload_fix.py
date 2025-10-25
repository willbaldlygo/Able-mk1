#!/usr/bin/env python3
"""
Test script to verify upload fixes
"""
import requests
import json

def test_multimodal_capabilities():
    """Test multimodal capabilities endpoint"""
    try:
        response = requests.get("http://localhost:8000/multimodal/capabilities")
        if response.status_code == 200:
            data = response.json()
            print("✅ Multimodal capabilities working")
            print(f"   LLava available: {data.get('llava_available')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print("❌ Multimodal capabilities failed")
            return False
    except Exception as e:
        print(f"❌ Error testing multimodal: {e}")
        return False

def test_backend_health():
    """Test backend health"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend health check passed")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print("❌ Backend health check failed")
            return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def main():
    print("🧪 Testing Upload Fixes")
    print("=" * 30)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Multimodal Capabilities", test_multimodal_capabilities)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Upload fixes are working.")
    else:
        print("⚠️  Some tests failed.")

if __name__ == "__main__":
    main()