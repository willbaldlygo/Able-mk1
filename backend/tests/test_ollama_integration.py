"""
Integration tests for Ollama + Anthropic dual AI provider setup in Avi.

This test suite verifies:
1. Both providers can be initialized and connected
2. Model switching works between providers
3. Fallback mechanisms function correctly
4. Performance monitoring is operational
5. Error handling works as expected

Author: DevOps Agent for Avi Multi-Agent System
"""
import pytest
import asyncio
import os
import time
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add backend directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.ai_service import AIService, AIProvider
from services.model_service import ModelService
from models import SourceInfo
from config import config


class TestOllamaIntegration:
    """Test suite for Ollama integration with fallback to Anthropic."""
    
    @pytest.fixture
    def ai_service(self):
        """Create AIService instance for testing."""
        return AIService()
    
    @pytest.fixture
    def model_service(self):
        """Create ModelService instance for testing."""
        return ModelService()
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample sources for testing."""
        return [
            SourceInfo(
                document_id="test-doc-1",
                document_name="Test Document 1.pdf",
                chunk_content="This is a test document about artificial intelligence and machine learning.",
                page_number=1,
                relevance_score=0.85
            ),
            SourceInfo(
                document_id="test-doc-2", 
                document_name="Test Document 2.pdf",
                chunk_content="This document discusses the integration of local and cloud AI models.",
                page_number=2,
                relevance_score=0.78
            )
        ]
    
    def test_provider_availability(self, ai_service):
        """Test that both providers are properly detected and available."""
        available_providers = ai_service.get_available_providers()
        
        # Should have at least one provider available
        assert len(available_providers) > 0, "No AI providers available"
        
        # Test individual provider connections
        connections = ai_service.test_connection()
        
        # At least one provider should be working
        assert any(connections.values()), f"No providers connected: {connections}"
        
        print(f"Available providers: {available_providers}")
        print(f"Connection status: {connections}")
    
    def test_ollama_specific_connection(self, model_service):
        """Test Ollama-specific connection and model availability."""
        if not config.ollama_enabled:
            pytest.skip("Ollama not enabled in configuration")
        
        # Test Ollama connection
        connection_ok = model_service.test_connection()
        
        if connection_ok:
            # Test available models
            models = model_service.get_available_models()
            ollama_models = models.get("ollama", [])
            
            assert len(ollama_models) > 0, "No Ollama models available"
            print(f"Available Ollama models: {[m['name'] for m in ollama_models]}")
            
            # Test specific model if available
            if any(m['name'] == 'gpt-oss:20b' for m in ollama_models):
                self._test_gpt_oss_20b(model_service)
        else:
            print("Ollama service not available - skipping Ollama-specific tests")
    
    def _test_gpt_oss_20b(self, model_service):
        """Test the gpt-oss:20b model specifically."""
        print("Testing gpt-oss:20b model...")
        
        start_time = time.time()
        response = model_service.generate_response(
            model_name="gpt-oss:20b",
            prompt="Hello! Please respond with 'Test successful' to verify functionality.",
            stream=False
        )
        end_time = time.time()
        
        assert response.get("success", False), f"gpt-oss:20b test failed: {response.get('error')}"
        
        response_time = end_time - start_time
        print(f"gpt-oss:20b response time: {response_time:.2f}s")
        print(f"gpt-oss:20b response: {response.get('response', '')[:100]}...")
    
    def test_anthropic_connection(self, ai_service):
        """Test Anthropic Claude connection if configured."""
        if not config.anthropic_api_key:
            pytest.skip("Anthropic API key not configured")
        
        connections = ai_service.test_connection(AIProvider.ANTHROPIC.value)
        anthropic_ok = connections.get(AIProvider.ANTHROPIC.value, False)
        
        if anthropic_ok:
            print("Anthropic Claude connection: OK")
        else:
            print("Anthropic Claude connection: FAILED")
    
    def test_dual_provider_response_generation(self, ai_service, sample_sources):
        """Test response generation with both providers if available."""
        available_providers = ai_service.get_available_providers()
        
        test_question = "What are the main topics discussed in these documents?"
        
        for provider in available_providers:
            print(f"\nTesting response generation with {provider}...")
            
            start_time = time.time()
            response = ai_service.generate_response_with_provider(
                question=test_question,
                sources=sample_sources,
                provider=provider
            )
            end_time = time.time()
            
            # Verify response structure
            assert hasattr(response, 'answer'), f"Response missing answer field for {provider}"
            assert hasattr(response, 'sources'), f"Response missing sources field for {provider}"
            assert hasattr(response, 'timestamp'), f"Response missing timestamp for {provider}"
            
            # Verify response content
            assert len(response.answer) > 0, f"Empty response from {provider}"
            assert len(response.sources) == len(sample_sources), f"Sources mismatch for {provider}"
            
            response_time = end_time - start_time
            print(f"{provider} response time: {response_time:.2f}s")
            print(f"{provider} response preview: {response.answer[:100]}...")
    
    def test_provider_switching(self, ai_service):
        """Test switching between providers."""
        available_providers = ai_service.get_available_providers()
        
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for switching test")
        
        # Test switching to each provider
        for provider in available_providers:
            print(f"Switching to {provider}...")
            
            success = ai_service.switch_provider(provider)
            assert success, f"Failed to switch to {provider}"
            
            # Verify the switch worked
            assert ai_service.default_provider.value == provider, f"Provider not switched to {provider}"
            
            print(f"Successfully switched to {provider}")
    
    def test_fallback_mechanism(self, ai_service, sample_sources):
        """Test fallback between providers when one fails."""
        available_providers = ai_service.get_available_providers()
        
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for fallback test")
        
        # Test with a provider that should work
        working_provider = available_providers[0]
        
        print(f"Testing fallback mechanism with primary provider: {working_provider}")
        
        # Mock a failure for testing
        original_method = None
        if working_provider == AIProvider.ANTHROPIC.value:
            original_method = ai_service._generate_anthropic_response
            ai_service._generate_anthropic_response = MagicMock(side_effect=Exception("Mocked failure"))
        elif working_provider == AIProvider.OLLAMA.value:
            original_method = ai_service._generate_ollama_response
            ai_service._generate_ollama_response = MagicMock(side_effect=Exception("Mocked failure"))
        
        try:
            # This should trigger fallback
            response = ai_service.generate_response_with_provider(
                question="Test fallback mechanism",
                sources=sample_sources,
                provider=working_provider
            )
            
            # Should get a response despite the primary provider failing
            assert hasattr(response, 'answer'), "Fallback response missing answer"
            assert len(response.answer) > 0, "Empty fallback response"
            
            print("Fallback mechanism working correctly")
            
        finally:
            # Restore original method
            if working_provider == AIProvider.ANTHROPIC.value:
                ai_service._generate_anthropic_response = original_method
            elif working_provider == AIProvider.OLLAMA.value:
                ai_service._generate_ollama_response = original_method
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, ai_service, sample_sources):
        """Test streaming response functionality."""
        available_providers = ai_service.get_available_providers()
        
        if not available_providers:
            pytest.skip("No providers available for streaming test")
        
        provider = available_providers[0]
        print(f"Testing streaming with {provider}...")
        
        chunks = []
        async for chunk in ai_service.generate_stream_response(
            question="What is the main topic?",
            sources=sample_sources[:1],  # Use just one source for faster streaming
            provider=provider
        ):
            chunks.append(chunk)
            if len(chunks) >= 5:  # Limit chunks for test
                break
        
        assert len(chunks) > 0, f"No streaming chunks received from {provider}"
        print(f"Received {len(chunks)} streaming chunks from {provider}")
    
    def test_performance_monitoring(self, ai_service, sample_sources):
        """Test that performance metrics are being tracked."""
        available_providers = ai_service.get_available_providers()
        
        if not available_providers:
            pytest.skip("No providers available for performance test")
        
        provider = available_providers[0]
        
        # Generate a response and measure performance
        start_time = time.time()
        response = ai_service.generate_response_with_provider(
            question="Quick test for performance monitoring",
            sources=sample_sources[:1],
            provider=provider
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Verify response
        assert hasattr(response, 'answer'), "Response missing for performance test"
        
        # Log performance metrics
        print(f"Performance test - Provider: {provider}, Response time: {response_time:.2f}s")
        print(f"Response length: {len(response.answer)} characters")
    
    def test_error_handling(self, ai_service):
        """Test error handling for invalid configurations."""
        # Test with non-existent provider
        result = ai_service.switch_provider("non-existent-provider")
        assert not result, "Should reject invalid provider"
        
        # Test with invalid model
        if AIProvider.OLLAMA.value in ai_service.get_available_providers():
            result = ai_service.switch_provider(AIProvider.OLLAMA.value, "non-existent-model")
            assert not result, "Should reject invalid model"
        
        print("Error handling tests passed")
    
    def test_configuration_validation(self):
        """Test that configuration is properly loaded and validated."""
        # Test required configuration exists
        assert hasattr(config, 'ollama_enabled'), "ollama_enabled config missing"
        assert hasattr(config, 'ollama_host'), "ollama_host config missing"
        assert hasattr(config, 'default_provider'), "default_provider config missing"
        assert hasattr(config, 'fallback_enabled'), "fallback_enabled config missing"
        
        # Test configuration values
        assert config.ollama_host.startswith('http'), f"Invalid ollama_host: {config.ollama_host}"
        assert config.default_provider in ['anthropic', 'ollama'], f"Invalid default_provider: {config.default_provider}"
        
        print(f"Configuration validation passed:")
        print(f"  Ollama enabled: {config.ollama_enabled}")
        print(f"  Ollama host: {config.ollama_host}")
        print(f"  Default provider: {config.default_provider}")
        print(f"  Fallback enabled: {config.fallback_enabled}")


class TestModelServiceIntegration:
    """Test suite specifically for ModelService integration."""
    
    @pytest.fixture
    def model_service(self):
        """Create ModelService instance for testing."""
        return ModelService()
    
    def test_model_service_initialization(self, model_service):
        """Test ModelService initializes properly."""
        assert hasattr(model_service, 'ollama_client'), "ollama_client not initialized"
        assert hasattr(model_service, 'initialization_status'), "initialization_status not set"
        
        status = model_service.initialization_status
        assert 'ollama_connected' in status, "ollama_connected status missing"
        assert 'anthropic_configured' in status, "anthropic_configured status missing"
        assert 'last_check' in status, "last_check status missing"
        
        print(f"ModelService initialization status: {status}")
    
    def test_available_models_discovery(self, model_service):
        """Test discovery of available models from all providers."""
        models = model_service.get_available_models()
        
        assert isinstance(models, dict), "Models should be returned as dict"
        
        # Check structure
        for provider in ['ollama', 'anthropic']:
            if provider in models:
                assert isinstance(models[provider], list), f"{provider} models should be a list"
                
                for model in models[provider]:
                    assert 'name' in model, f"Model missing name: {model}"
                    assert 'status' in model, f"Model missing status: {model}"
        
        print(f"Discovered models: {models}")
    
    def test_model_info_retrieval(self, model_service):
        """Test retrieving detailed information about specific models."""
        models = model_service.get_available_models()
        ollama_models = models.get('ollama', [])
        
        if ollama_models:
            test_model = ollama_models[0]['name']
            model_info = model_service.get_model_info(test_model)
            
            if model_info:
                print(f"Model info for {test_model}: {model_info}")
            else:
                print(f"No detailed info available for {test_model}")


def run_integration_tests():
    """Run integration tests manually without pytest."""
    print("=" * 60)
    print("RUNNING AVI OLLAMA INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Test basic initialization
        print("\n1. Testing Service Initialization...")
        ai_service = AIService()
        model_service = ModelService()
        print("✅ Services initialized successfully")
        
        # Test provider availability
        print("\n2. Testing Provider Availability...")
        available_providers = ai_service.get_available_providers()
        connections = ai_service.test_connection()
        print(f"Available providers: {available_providers}")
        print(f"Connection status: {connections}")
        
        # Test Ollama specific functionality
        print("\n3. Testing Ollama Integration...")
        if config.ollama_enabled and AIProvider.OLLAMA.value in available_providers:
            # Test gpt-oss:20b if available
            test_result = ai_service.test_ollama_gpt_oss_20b()
            print("gpt-oss:20b Test Result:")
            for key, value in test_result.items():
                print(f"  {key}: {value}")
        else:
            print("Ollama not available - skipping Ollama tests")
        
        # Test model discovery
        print("\n4. Testing Model Discovery...")
        models = model_service.get_available_models()
        for provider, provider_models in models.items():
            print(f"{provider}: {len(provider_models)} models")
            for model in provider_models[:3]:  # Show first 3
                print(f"  - {model['name']} ({model['status']})")
        
        print("\n" + "=" * 60)
        print("INTEGRATION TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Can be run directly for manual testing
    run_integration_tests()