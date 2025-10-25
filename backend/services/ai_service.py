"""AI service for Able with GraphRAG integration, dual provider support, and multimodal capabilities."""
import anthropic
import asyncio
import base64
import requests
from datetime import datetime
from typing import List, Optional, Union, AsyncGenerator, Dict
from enum import Enum

from config import config
from models import (
    SourceInfo,
    ChatResponse,
    ChatRequest,
    EnhancedSourceInfo,
    EnhancedChatResponse,
    EntityInfo,
    RelationshipInfo
)
from services.model_service import model_service
from .response_formatter import ResponseFormatter


class AIProvider(Enum):
    """Available AI providers."""
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class AIService:
    """Multi-provider AI service with GraphRAG support (Anthropic + Ollama)."""
    
    def __init__(self):
        # Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.claude_model = config.claude_model
        
        # Ollama integration
        self.model_service = model_service
        self.ollama_model = config.ollama_model
        
        # Provider settings
        self.default_provider = AIProvider(config.default_provider)
        self.fallback_enabled = config.fallback_enabled
        
        # Response formatter
        self.formatter = ResponseFormatter()
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers."""
        providers = []
        
        # Check Anthropic
        if config.anthropic_api_key:
            providers.append(AIProvider.ANTHROPIC.value)
            
        # Check Ollama
        if config.ollama_enabled and self.model_service.test_connection():
            providers.append(AIProvider.OLLAMA.value)
            
        return providers
    
    def switch_provider(self, provider: str, model: Optional[str] = None) -> bool:
        """Switch the default AI provider."""
        try:
            new_provider = AIProvider(provider)
            
            # Validate provider availability
            if provider not in self.get_available_providers():
                return False
                
            self.default_provider = new_provider
            
            # Update model if specified
            if model:
                if provider == AIProvider.ANTHROPIC.value:
                    self.claude_model = model
                elif provider == AIProvider.OLLAMA.value:
                    # Verify model exists
                    if self.model_service.check_model_availability(model):
                        self.ollama_model = model
                    else:
                        return False
            
            return True
        except ValueError:
            return False
    
    def generate_response_with_provider(
        self, 
        question: str, 
        sources: List[SourceInfo], 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        disable_formatting: bool = False
    ) -> ChatResponse:
        """Generate response using specified provider with fallback."""
        target_provider = AIProvider(provider) if provider else self.default_provider
        
        # Try primary provider
        try:
            if target_provider == AIProvider.ANTHROPIC:
                return self._generate_anthropic_response(question, sources, model, disable_formatting)
            elif target_provider == AIProvider.OLLAMA:
                return self._generate_ollama_response(question, sources, model, disable_formatting)
        except Exception as e:
            print(f"Primary provider ({target_provider.value}) failed: {e}")
            
            # Try fallback if enabled
            if self.fallback_enabled:
                fallback_provider = (
                    AIProvider.OLLAMA if target_provider == AIProvider.ANTHROPIC 
                    else AIProvider.ANTHROPIC
                )
                
                if fallback_provider.value in self.get_available_providers():
                    try:
                        print(f"Attempting fallback to {fallback_provider.value}")
                        if fallback_provider == AIProvider.ANTHROPIC:
                            return self._generate_anthropic_response(question, sources, model, disable_formatting)
                        elif fallback_provider == AIProvider.OLLAMA:
                            return self._generate_ollama_response(question, sources, model, disable_formatting)
                    except Exception as fallback_error:
                        print(f"Fallback provider failed: {fallback_error}")
            
            # Return error response
            return ChatResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                timestamp=datetime.now()
            )
    
    def generate_response(self, question: str, sources: List[SourceInfo]) -> ChatResponse:
        """Generate response using default provider with fallback."""
        return self.generate_response_with_provider(question, sources)
    
    def _generate_anthropic_response(
        self, 
        question: str, 
        sources: List[SourceInfo], 
        model: Optional[str] = None,
        disable_formatting: bool = False
    ) -> ChatResponse:
        """Generate response using Anthropic Claude."""
        # Prepare context
        context = self._prepare_context(sources)
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(question, context)
        
        # Use specified model or default
        model_to_use = model or self.claude_model
        
        # Call Claude
        response = self.anthropic_client.messages.create(
            model=model_to_use,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Detect formatting requirements and format response
        raw_answer = response.content[0].text
        if disable_formatting:
            formatted_answer = raw_answer
        else:
            format_requirements = self.formatter.detect_format_requirements(question)
            formatted_answer = self.formatter.format_response(raw_answer, format_requirements)
        
        return ChatResponse(
            answer=formatted_answer,
            sources=sources,
            timestamp=datetime.now()
        )
    
    def _generate_ollama_response(
        self, 
        question: str, 
        sources: List[SourceInfo], 
        model: Optional[str] = None,
        disable_formatting: bool = False
    ) -> ChatResponse:
        """Generate response using Ollama."""
        # Prepare context
        context = self._prepare_context(sources)
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(question, context)
        
        # Use specified model or default
        model_to_use = model or self.ollama_model
        
        # Call Ollama
        response = self.model_service.generate_response(model_to_use, prompt, stream=False)
        
        if not response['success']:
            raise Exception(response.get('error', 'Ollama generation failed'))
        
        # Detect formatting requirements and format response
        raw_answer = response['response']
        if disable_formatting:
            formatted_answer = raw_answer
        else:
            format_requirements = self.formatter.detect_format_requirements(question)
            formatted_answer = self.formatter.format_response(raw_answer, format_requirements)
        
        return ChatResponse(
            answer=formatted_answer,
            sources=sources,
            timestamp=datetime.now()
        )
    
    def test_connection(self, provider: Optional[str] = None) -> Dict[str, bool]:
        """Test AI provider connections."""
        results = {}
        providers_to_test = [provider] if provider else [p.value for p in AIProvider]
        
        for prov in providers_to_test:
            if prov == AIProvider.ANTHROPIC.value:
                try:
                    response = self.anthropic_client.messages.create(
                        model=self.claude_model,
                        max_tokens=10,
                        messages=[{
                            "role": "user",
                            "content": "Hello"
                        }]
                    )
                    results[prov] = True
                except Exception:
                    results[prov] = False
                    
            elif prov == AIProvider.OLLAMA.value:
                results[prov] = self.model_service.test_connection()
        
        return results
    
    def test_ollama_gpt_oss_20b(self) -> Dict[str, Union[bool, str, Dict]]:
        """Test Ollama connection specifically with gpt-oss:20b model."""
        test_result = {
            "model": "gpt-oss:20b",
            "connection_available": False,
            "model_available": False,
            "generation_test": False,
            "response_time": None,
            "error": None,
            "test_response": None
        }
        
        try:
            # First check if Ollama service is running
            if not self.model_service.test_connection():
                test_result["error"] = "Ollama service not available"
                return test_result
            
            test_result["connection_available"] = True
            
            # Check if gpt-oss:20b model is available
            available_models = self.model_service.get_available_models()
            ollama_models = [model["name"] for model in available_models.get("ollama", [])]
            
            if "gpt-oss:20b" not in ollama_models:
                test_result["error"] = f"Model gpt-oss:20b not found. Available models: {ollama_models}"
                return test_result
            
            test_result["model_available"] = True
            
            # Test generation with the model
            import time
            start_time = time.time()
            
            response = self.model_service.generate_response(
                model_name="gpt-oss:20b",
                prompt="Hello! Please respond with 'Connection successful' to test the model.",
                stream=False
            )
            
            end_time = time.time()
            test_result["response_time"] = round(end_time - start_time, 2)
            
            if response.get("success", False):
                test_result["generation_test"] = True
                test_result["test_response"] = response.get("response", "")
            else:
                test_result["error"] = response.get("error", "Unknown generation error")
                
        except Exception as e:
            test_result["error"] = f"Test failed with exception: {str(e)}"
        
        return test_result
    
    def test_ollama_model_detailed(self, model_name: str) -> Dict[str, Union[bool, str, Dict, float]]:
        """Perform detailed test of any Ollama model with performance metrics."""
        test_result = {
            "model": model_name,
            "connection_available": False,
            "model_available": False,
            "model_info": None,
            "generation_test": False,
            "response_time": None,
            "memory_usage": None,
            "error": None,
            "test_response": None
        }
        
        try:
            # Check Ollama connection
            if not self.model_service.test_connection():
                test_result["error"] = "Ollama service not available"
                return test_result
            
            test_result["connection_available"] = True
            
            # Check model availability
            available_models = self.model_service.get_available_models()
            ollama_models = available_models.get("ollama", [])
            model_exists = any(model["name"] == model_name for model in ollama_models)
            
            if not model_exists:
                available_names = [model["name"] for model in ollama_models]
                test_result["error"] = f"Model {model_name} not found. Available: {available_names[:5]}"
                return test_result
            
            test_result["model_available"] = True
            
            # Get model info
            model_info = self.model_service.get_model_info(model_name)
            if model_info:
                test_result["model_info"] = {
                    "parameters": model_info.get("parameters", "N/A"),
                    "size": model_info.get("details", {}).get("parameter_size", "N/A"),
                    "family": model_info.get("details", {}).get("family", "N/A")
                }
            
            # Performance test
            import time
            import psutil
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            test_prompt = "Please respond with exactly 'Test successful' to verify model functionality."
            response = self.model_service.generate_response(
                model_name=model_name,
                prompt=test_prompt,
                stream=False
            )
            
            end_time = time.time()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            test_result["response_time"] = round(end_time - start_time, 2)
            test_result["memory_usage"] = round(memory_after - memory_before, 2)
            
            if response.get("success", False):
                test_result["generation_test"] = True
                test_result["test_response"] = response.get("response", "").strip()
            else:
                test_result["error"] = response.get("error", "Generation failed")
                
        except Exception as e:
            test_result["error"] = f"Detailed test failed: {str(e)}"
        
        return test_result
    
    async def generate_stream_response(
        self, 
        question: str, 
        sources: List[SourceInfo],
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using specified provider."""
        target_provider = AIProvider(provider) if provider else self.default_provider
        
        # Prepare context and prompt
        context = self._prepare_context(sources)
        prompt = self._create_enhanced_prompt(question, context)
        
        try:
            if target_provider == AIProvider.ANTHROPIC:
                model_to_use = model or self.claude_model
                # Anthropic streaming
                async with self.anthropic_client.messages.stream(
                    model=model_to_use,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                ) as stream:
                    async for chunk in stream.text_stream:
                        yield chunk
                        
            elif target_provider == AIProvider.OLLAMA:
                model_to_use = model or self.ollama_model
                # Ollama streaming
                async for chunk in self.model_service.generate_stream_response(model_to_use, prompt):
                    yield chunk
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _prepare_context(self, sources: List[SourceInfo]) -> str:
        """Prepare enhanced context from sources."""
        if not sources:
            return "No relevant documents found."
        
        # Group sources by document
        document_groups = {}
        for source in sources:
            doc_name = source.document_name
            if doc_name not in document_groups:
                document_groups[doc_name] = []
            document_groups[doc_name].append(source)
        
        # Format context with document grouping
        context_parts = []
        excerpt_counter = 1
        
        for doc_name, doc_sources in document_groups.items():
            context_parts.append(f"\n=== From Document: {doc_name} ===")
            
            for source in doc_sources:
                context_parts.append(f"""
Excerpt {excerpt_counter} (relevance: {source.relevance_score:.2f}):
{source.chunk_content}
---""")
                excerpt_counter += 1
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create enhanced prompt with better instructions."""
        return f"""You are a helpful research assistant analyzing PDF documents. Answer the user's question based on the provided document excerpts.

IMPORTANT INSTRUCTIONS:
1. When the user refers to "documents" or "docs", they mean the complete PDF files, not individual excerpts
2. Multiple excerpts may come from the same document - group your analysis by document when relevant
3. Base your answer primarily on the information provided in the excerpts
4. Be specific about which documents support your statements
5. If the excerpts don't contain enough information to fully answer the question, explain specifically what information is missing and provide whatever partial answer you can from the available sources
6. Keep your response concise but comprehensive
7. If you need to make inferences, clearly distinguish them from facts stated in the excerpts
8. Consider the relevance scores - higher scores indicate more relevant content

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {question}

Please provide a helpful and accurate response based on the document excerpts provided."""
    
    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        """Generate response using traditional flow for backwards compatibility."""
        from services.vector_service import VectorService
        vector_service = VectorService()
        
        # Get sources using vector search
        sources = vector_service.search_documents(
            query=request.question,
            document_ids=request.document_ids
        )
        
        return self.generate_response_with_sources(request.question, sources)
    
    def generate_response_with_sources(self, question: str, sources: List[SourceInfo]) -> ChatResponse:
        """Generate response with provided sources (backwards compatibility)."""
        return self.generate_response_with_provider(question, sources)
    
    def generate_enhanced_response(
        self, 
        question: str, 
        sources: List[EnhancedSourceInfo], 
        search_type: str = "vector",
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> EnhancedChatResponse:
        """Generate enhanced response with GraphRAG context using specified provider."""
        target_provider = AIProvider(provider) if provider else self.default_provider
        
        try:
            # Prepare enhanced context with entities and relationships
            context = self._prepare_enhanced_context(sources, search_type)
            
            # Create GraphRAG-aware prompt
            prompt = self._create_graphrag_prompt(question, context, search_type)
            
            if target_provider == AIProvider.ANTHROPIC:
                raw_answer = self._generate_anthropic_enhanced_response(prompt, model)
            elif target_provider == AIProvider.OLLAMA:
                raw_answer = self._generate_ollama_enhanced_response(prompt, model)
            else:
                raise ValueError(f"Unsupported provider: {target_provider}")
            
            # Apply formatting to enhanced responses
            format_requirements = self.formatter.detect_format_requirements(question)
            formatted_answer = self.formatter.format_response(raw_answer, format_requirements)
            
            return EnhancedChatResponse(
                answer=formatted_answer,
                sources=sources,
                timestamp=datetime.now(),
                search_type=search_type
            )
            
        except Exception as e:
            # Try fallback if enabled
            if self.fallback_enabled and provider:
                fallback_provider = (
                    AIProvider.OLLAMA if target_provider == AIProvider.ANTHROPIC 
                    else AIProvider.ANTHROPIC
                )
                
                if fallback_provider.value in self.get_available_providers():
                    try:
                        return self.generate_enhanced_response(
                            question, sources, search_type, fallback_provider.value, model
                        )
                    except Exception:
                        pass  # Continue to error response
            
            return EnhancedChatResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                timestamp=datetime.now(),
                search_type="error"
            )
    
    def _generate_anthropic_enhanced_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate enhanced response using Anthropic."""
        model_to_use = model or self.claude_model
        response = self.anthropic_client.messages.create(
            model=model_to_use,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content[0].text
    
    def _generate_ollama_enhanced_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate enhanced response using Ollama."""
        model_to_use = model or self.ollama_model
        response = self.model_service.generate_response(model_to_use, prompt, stream=False)
        
        if not response['success']:
            raise Exception(response.get('error', 'Ollama generation failed'))
            
        return response['response']
    
    def _prepare_enhanced_context(self, sources: List[EnhancedSourceInfo], search_type: str) -> str:
        """Prepare enhanced context with entity and relationship information."""
        if not sources:
            return "No relevant documents found."
        
        context_parts = []
        excerpt_counter = 1
        
        # Group sources by document
        document_groups = {}
        for source in sources:
            doc_name = source.document_name
            if doc_name not in document_groups:
                document_groups[doc_name] = []
            document_groups[doc_name].append(source)
        
        for doc_name, doc_sources in document_groups.items():
            context_parts.append(f"\n=== From Document: {doc_name} ===")
            
            # Collect all entities and relationships from this document
            all_entities = []
            all_relationships = []
            
            for source in doc_sources:
                context_parts.append(f"""
Excerpt {excerpt_counter} (relevance: {source.relevance_score:.2f}):
{source.chunk_content}""")
                
                # Add entity information if available
                if source.entities:
                    entity_names = [entity.name for entity in source.entities]
                    context_parts.append(f"Key entities: {', '.join(entity_names)}")
                    all_entities.extend(source.entities)
                
                # Add relationship information if available
                if source.relationships:
                    rel_descriptions = [f"{rel.source_entity} -> {rel.target_entity}" for rel in source.relationships]
                    context_parts.append(f"Key relationships: {', '.join(rel_descriptions[:3])}")  # Limit to avoid clutter
                    all_relationships.extend(source.relationships)
                
                context_parts.append("---")
                excerpt_counter += 1
            
            # Add document-level entity and relationship summary
            if all_entities:
                unique_entities = {entity.name: entity for entity in all_entities}
                entity_types = {}
                for entity in unique_entities.values():
                    entity_type = entity.entity_type
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity.name)
                
                context_parts.append(f"\nDocument entities by type:")
                for entity_type, names in entity_types.items():
                    context_parts.append(f"  {entity_type}: {', '.join(names[:5])}")  # Limit to 5 per type
            
            if all_relationships:
                context_parts.append(f"\nKey relationships in {doc_name}:")
                unique_rels = {(rel.source_entity, rel.target_entity): rel for rel in all_relationships}
                for rel in list(unique_rels.values())[:5]:  # Limit to 5 relationships
                    context_parts.append(f"  {rel.source_entity} -> {rel.target_entity} ({rel.relationship_type})")
        
        return "\n".join(context_parts)
    
    def _create_graphrag_prompt(self, question: str, context: str, search_type: str) -> str:
        """Create GraphRAG-aware prompt with entity and relationship understanding."""
        search_context = {
            "global": "This answer was generated using global knowledge synthesis across all documents, focusing on broad themes and patterns.",
            "local": "This answer was generated using local entity relationship analysis, focusing on specific connections and detailed information.",
            "vector": "This answer was generated using semantic similarity search with entity enhancement."
        }
        
        return f"""You are Able, an advanced research assistant that combines semantic search with knowledge graph analysis. You can understand both document content and the relationships between entities (people, organizations, concepts, etc.) mentioned in the documents.

SEARCH METHOD USED: {search_type.upper()}
{search_context.get(search_type, "")}

IMPORTANT INSTRUCTIONS:
1. When the user refers to "documents" or "docs", they mean complete PDF files, not individual excerpts
2. Pay attention to entity information and relationships provided - they help you understand connections across documents
3. Use entity relationships to provide richer, more connected answers
4. When entities appear in multiple documents, highlight these cross-document connections
5. Base your answer on the provided excerpts and entity information
6. Be specific about which documents support your statements
7. If discussing relationships between entities, cite the documents where these relationships are mentioned
8. Keep responses comprehensive but focused
9. When appropriate, highlight key entities and their roles in your answer

ENHANCED DOCUMENT CONTEXT WITH ENTITIES:
{context}

USER QUESTION: {question}

Please provide a comprehensive answer that leverages both the document content and the entity relationship information provided."""

    # Multimodal/Vision capabilities

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 for llava processing."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def check_llava_availability(self) -> bool:
        """Check if llava model is available in ollama."""
        try:
            response = requests.get(f"{config.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if "llava" in model.get("name", "").lower():
                        return True
            return False
        except Exception as e:
            print(f"Could not check llava availability: {e}")
            return False

    def generate_multimodal_response(
        self,
        question: str,
        sources: List[SourceInfo],
        image_paths: List[str] = None,
        visual_context: Optional[str] = None
    ) -> ChatResponse:
        """
        Generate response using multimodal capabilities (text + images).

        Args:
            question: User's question
            sources: Document sources
            image_paths: List of image file paths to analyze
            visual_context: Pre-generated visual context descriptions

        Returns:
            ChatResponse with multimodal analysis
        """
        # Check if llava is available for image analysis
        llava_available = self.check_llava_availability()

        # Prepare text context
        text_context = self._prepare_context(sources)

        # Prepare visual context
        image_descriptions = []
        if image_paths and llava_available:
            for image_path in image_paths[:3]:  # Limit to 3 images to avoid token limits
                description = self._analyze_image_with_llava(image_path, question)
                if description:
                    image_descriptions.append(f"Image Analysis: {description}")

        # Use provided visual context if no new analysis was done
        if not image_descriptions and visual_context:
            image_descriptions.append(f"Visual Context: {visual_context}")

        # Create multimodal prompt
        multimodal_prompt = self._create_multimodal_prompt(
            question,
            text_context,
            image_descriptions
        )

        # Generate response based on provider
        try:
            if self.default_provider == AIProvider.ANTHROPIC:
                response = self._generate_anthropic_multimodal_response(multimodal_prompt)
            else:
                response = self._generate_ollama_multimodal_response(multimodal_prompt)

            return ChatResponse(
                answer=response,
                sources=sources,
                timestamp=datetime.now()
            )

        except Exception as e:
            return ChatResponse(
                answer=f"I encountered an error while processing your multimodal question: {str(e)}",
                sources=sources,
                timestamp=datetime.now()
            )

    def _analyze_image_with_llava(self, image_path: str, context_question: str = None) -> Optional[str]:
        """Analyze a single image using llava model."""
        try:
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return None

            # Create context-aware prompt
            if context_question:
                prompt = f"""Analyze this image in the context of this question: "{context_question}"

Describe what you see with focus on:
1. Text content (OCR if any)
2. Charts, graphs, diagrams, or tables
3. Visual elements relevant to the question
4. Key information that would help answer the question
5. Overall context and setting

Be detailed but concise."""
            else:
                prompt = """Describe this image in detail, focusing on:
1. Main subjects and content
2. Any text visible (OCR)
3. Charts, diagrams, tables, or data visualizations
4. Important visual elements
5. Context that would be useful for document understanding"""

            payload = {
                "model": "llava:latest",  # Use available llava model
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent analysis
                    "top_k": 10,
                    "top_p": 0.9
                }
            }

            response = requests.post(
                f"{config.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"llava analysis failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"Error analyzing image with llava: {e}")
            return None

    def _create_multimodal_prompt(
        self,
        question: str,
        text_context: str,
        image_descriptions: List[str]
    ) -> str:
        """Create prompt that combines text and visual information."""
        visual_section = ""
        if image_descriptions:
            visual_section = f"""

VISUAL INFORMATION:
{chr(10).join(image_descriptions)}
"""

        return f"""You are Able, an advanced multimodal research assistant that can analyze both text documents and visual content (images, charts, diagrams, etc.).

INSTRUCTIONS:
1. Use both the text content and visual information to answer the question
2. When referencing visual elements, be specific about what you observed
3. Integrate visual insights with textual information for a comprehensive answer
4. If visual elements contradict or supplement text information, highlight this
5. Be specific about which sources (text or visual) support your statements

DOCUMENT TEXT CONTEXT:
{text_context}
{visual_section}

USER QUESTION: {question}

Please provide a comprehensive answer that integrates both textual and visual information."""

    def _generate_anthropic_multimodal_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate multimodal response using Anthropic Claude."""
        model_to_use = model or self.claude_model

        response = self.anthropic_client.messages.create(
            model=model_to_use,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response.content[0].text

    def _generate_ollama_multimodal_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate multimodal response using Ollama."""
        model_to_use = model or self.ollama_model

        response = self.model_service.generate_response(model_to_use, prompt, stream=False)

        if not response['success']:
            raise Exception(response.get('error', 'Ollama generation failed'))

        return response['response']

    def test_multimodal_capabilities(self) -> Dict[str, Union[bool, str]]:
        """Test multimodal capabilities including llava availability."""
        result = {
            "llava_available": False,
            "llava_model": None,
            "test_successful": False,
            "error": None
        }

        try:
            # Check if llava is available
            result["llava_available"] = self.check_llava_availability()

            if result["llava_available"]:
                # Try to identify the specific llava model
                response = requests.get(f"{config.ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    for model in models:
                        model_name = model.get("name", "")
                        if "llava" in model_name.lower():
                            result["llava_model"] = model_name
                            break

                # Simple test if we found a model
                if result["llava_model"]:
                    result["test_successful"] = True
            else:
                result["error"] = "No llava model found in ollama"

        except Exception as e:
            result["error"] = str(e)

        return result