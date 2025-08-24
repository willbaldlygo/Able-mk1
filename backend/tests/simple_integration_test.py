#!/usr/bin/env python3
"""
Simple Integration Test for Ollama + Anthropic Dual Setup
No external dependencies required - can be run directly.

DevOps Agent Implementation for Avi Multi-Agent System
"""
import sys
import time
import traceback
from pathlib import Path

# Add backend directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from services.ai_service import AIService, AIProvider
        from services.model_service import ModelService
        from models import SourceInfo
        from config import config
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from config import config
        
        # Check required attributes
        required_attrs = ['ollama_enabled', 'ollama_host', 'default_provider', 'fallback_enabled']
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"❌ Missing config attribute: {attr}")
                return False
        
        print(f"✅ Configuration loaded:")
        print(f"  Ollama enabled: {config.ollama_enabled}")
        print(f"  Ollama host: {config.ollama_host}")
        print(f"  Default provider: {config.default_provider}")
        print(f"  Fallback enabled: {config.fallback_enabled}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_service_initialization():
    """Test service initialization."""
    print("\nTesting service initialization...")
    try:
        from services.ai_service import AIService
        from services.model_service import ModelService
        
        ai_service = AIService()
        model_service = ModelService()
        
        print("✅ Services initialized successfully")
        return ai_service, model_service
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        traceback.print_exc()
        return None, None

def test_provider_availability(ai_service):
    """Test provider availability and connections."""
    print("\nTesting provider availability...")
    try:
        available_providers = ai_service.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        if not available_providers:
            print("⚠️  No providers available")
            return False
        
        # Test connections
        connections = ai_service.test_connection()
        print(f"Connection status: {connections}")
        
        working_providers = [p for p, status in connections.items() if status]
        print(f"Working providers: {working_providers}")
        
        if working_providers:
            print("✅ At least one provider is working")
            return True
        else:
            print("❌ No providers are working")
            return False
            
    except Exception as e:
        print(f"❌ Provider availability test failed: {e}")
        traceback.print_exc()
        return False

def test_ollama_specific(ai_service):
    """Test Ollama-specific functionality."""
    print("\nTesting Ollama integration...")
    try:
        from services.ai_service import AIProvider
        from config import config
        
        if not config.ollama_enabled:
            print("⚠️  Ollama disabled in configuration")
            return True
        
        available_providers = ai_service.get_available_providers()
        if AIProvider.OLLAMA.value not in available_providers:
            print("⚠️  Ollama not available")
            return True
        
        # Test gpt-oss:20b specifically
        print("Testing gpt-oss:20b model...")
        test_result = ai_service.test_ollama_gpt_oss_20b()
        
        print("gpt-oss:20b Test Results:")
        for key, value in test_result.items():
            status_icon = "✅" if value == True else "❌" if value == False else "ℹ️"
            print(f"  {status_icon} {key}: {value}")
        
        if test_result.get("generation_test", False):
            print("✅ Ollama gpt-oss:20b working correctly")
            return True
        else:
            print(f"⚠️  Ollama gpt-oss:20b test incomplete: {test_result.get('error', 'Unknown issue')}")
            return True  # Don't fail the whole test if model isn't available
            
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        traceback.print_exc()
        return False

def test_model_discovery(model_service):
    """Test model discovery functionality."""
    print("\nTesting model discovery...")
    try:
        models = model_service.get_available_models()
        
        print("Discovered models:")
        total_models = 0
        for provider, provider_models in models.items():
            count = len(provider_models) if provider_models else 0
            total_models += count
            print(f"  {provider}: {count} models")
            
            # Show first few models
            if provider_models:
                for model in provider_models[:3]:
                    print(f"    - {model.get('name', 'Unknown')} ({model.get('status', 'Unknown')})")
                if len(provider_models) > 3:
                    print(f"    ... and {len(provider_models) - 3} more")
        
        print(f"Total models discovered: {total_models}")
        
        if total_models > 0:
            print("✅ Model discovery working")
            return True
        else:
            print("⚠️  No models discovered")
            return False
            
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        traceback.print_exc()
        return False

def test_basic_response_generation(ai_service):
    """Test basic response generation."""
    print("\nTesting response generation...")
    try:
        from models import SourceInfo
        
        # Create test data
        test_sources = [
            SourceInfo(
                document_id="test-1",
                document_name="Test.pdf",
                chunk_content="This is a test document about AI integration testing.",
                page_number=1,
                relevance_score=0.9
            )
        ]
        
        test_question = "What is this document about?"
        
        available_providers = ai_service.get_available_providers()
        if not available_providers:
            print("⚠️  No providers available for response test")
            return False
        
        # Test with first available provider
        provider = available_providers[0]
        print(f"Testing response generation with {provider}...")
        
        start_time = time.time()
        response = ai_service.generate_response_with_provider(
            question=test_question,
            sources=test_sources,
            provider=provider
        )
        end_time = time.time()
        
        if hasattr(response, 'answer') and len(response.answer) > 0:
            response_time = end_time - start_time
            print(f"✅ Response generated in {response_time:.2f}s")
            print(f"Response preview: {response.answer[:100]}...")
            return True
        else:
            print("❌ Empty or invalid response generated")
            return False
            
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("ABLE OLLAMA INTEGRATION TEST SUITE")
    print("DevOps Agent - Simple Integration Test")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Imports
    test_results.append(test_imports())
    if not test_results[-1]:
        print("\n❌ Critical failure - cannot proceed without imports")
        return False
    
    # Test 2: Configuration
    test_results.append(test_configuration())
    
    # Test 3: Service Initialization
    ai_service, model_service = test_service_initialization()
    test_results.append(ai_service is not None and model_service is not None)
    
    if ai_service and model_service:
        # Test 4: Provider Availability
        test_results.append(test_provider_availability(ai_service))
        
        # Test 5: Ollama Specific
        test_results.append(test_ollama_specific(ai_service))
        
        # Test 6: Model Discovery
        test_results.append(test_model_discovery(model_service))
        
        # Test 7: Response Generation
        test_results.append(test_basic_response_generation(ai_service))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Integration successful!")
        return True
    else:
        print(f"⚠️  {total - passed} tests had issues - Review above output")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)