"""Hybrid search service combining GraphRAG and vector search for Able."""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from config import config
from models import (
    EnhancedChatRequest, 
    EnhancedChatResponse, 
    EnhancedSourceInfo,
    GraphSearchResult,
    EntityInfo,
    RelationshipInfo
)
from services.graphrag_service import GraphRAGService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)

class HybridSearchService:
    """Intelligent search system combining GraphRAG global/local search with vector search."""
    
    def __init__(self):
        self.graphrag_service = GraphRAGService()
        self.vector_service = VectorService()
        
        # Search strategy thresholds
        self.global_search_threshold = 0.7  # Confidence threshold for global search
        self.local_search_threshold = 0.6   # Confidence threshold for local search
        self.fallback_to_vector = True      # Always fall back to vector if GraphRAG fails
    
    async def intelligent_search(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """Perform intelligent search using the best strategy for the query."""
        start_time = time.time()
        
        # Determine search strategy
        if request.search_type == "auto":
            search_type = self._analyze_query_strategy(request.question)
        else:
            search_type = request.search_type or "vector"
        
        logger.info(f"Using search strategy: {search_type} for query: {request.question[:100]}...")
        
        # Execute search based on strategy
        try:
            if search_type == "global" and self.graphrag_service.is_available():
                result = await self._global_search(request)
            elif search_type == "local" and self.graphrag_service.is_available():
                result = await self._local_search(request)
            else:
                result = await self._vector_search(request)
            
            # If GraphRAG search failed and fallback is enabled, try vector search
            if not result.answer and self.fallback_to_vector and search_type != "vector":
                logger.info("GraphRAG search failed, falling back to vector search")
                result = await self._vector_search(request)
            
            # Log search performance
            total_time = time.time() - start_time
            logger.info(f"Search completed in {total_time:.2f}s using {result.search_type} strategy")
            
            return result
        
        except Exception as e:
            logger.error(f"Search error: {e}")
            # Fallback to vector search on any error
            if search_type != "vector":
                logger.info("Falling back to vector search due to error")
                return await self._vector_search(request)
            else:
                # Return error response
                return EnhancedChatResponse(
                    answer="I encountered an error while searching. Please try rephrasing your question.",
                    sources=[],
                    timestamp=datetime.now(),
                    search_type="error"
                )
    
    def _analyze_query_strategy(self, query: str) -> str:
        """Analyze query to determine the best search strategy."""
        query_lower = query.lower()
        
        # Global search indicators (broad, research-oriented questions)
        global_indicators = [
            'overview', 'summary', 'general', 'broad', 'overall', 'comprehensive',
            'what are the main', 'key themes', 'major trends', 'across all',
            'in general', 'trends', 'patterns', 'compare', 'contrast',
            'summarize', 'research shows', 'studies indicate', 'literature',
            'field of', 'domain of', 'area of study'
        ]
        
        # Local search indicators (specific entities, relationships)
        local_indicators = [
            'specific', 'particular', 'relationship between', 'how does', 
            'what is the connection', 'entity', 'person', 'organization',
            'specific case', 'detailed', 'precise', 'who is', 'what is',
            'where is', 'when did', 'how is', 'why does', 'connection between',
            'related to', 'associated with', 'linked to'
        ]
        
        # Vector search indicators (simple factual queries)
        vector_indicators = [
            'define', 'definition', 'meaning of', 'what does', 'simple',
            'quick question', 'brief', 'short answer', 'exactly',
            'specifically mentions', 'quote', 'citation', 'reference'
        ]
        
        # Count indicators
        global_score = sum(1 for indicator in global_indicators if indicator in query_lower)
        local_score = sum(1 for indicator in local_indicators if indicator in query_lower)
        vector_score = sum(1 for indicator in vector_indicators if indicator in query_lower)
        
        # Additional heuristics
        if len(query.split()) > 15:  # Long queries tend to be research-oriented
            global_score += 1
        
        if '?' in query and len(query.split()) < 8:  # Short questions
            vector_score += 1
        
        # Determine strategy
        max_score = max(global_score, local_score, vector_score)
        
        if max_score == 0:
            return "vector"  # Default fallback
        elif global_score == max_score:
            return "global"
        elif local_score == max_score:
            return "local"
        else:
            return "vector"
    
    async def _global_search(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """Perform global search using community summaries."""
        try:
            # Execute GraphRAG global search
            graph_result = await self.graphrag_service.query_global_context(
                query=request.question,
                community_level=2
            )
            
            if graph_result["success"]:
                # Create graph search result
                graph_insights = GraphSearchResult(
                    answer=graph_result["answer"],
                    search_type="global",
                    context_data=graph_result.get("context_data", {}),
                    completion_time=graph_result.get("completion_time"),
                    llm_calls=graph_result.get("llm_calls"),
                    prompt_tokens=graph_result.get("prompt_tokens")
                )
                
                # Get supporting vector sources for attribution
                vector_sources = await self._get_supporting_sources(request, graph_result["answer"])
                
                return EnhancedChatResponse(
                    answer=graph_result["answer"],
                    sources=vector_sources,
                    timestamp=datetime.now(),
                    search_type="global",
                    graph_insights=graph_insights
                )
            
            else:
                raise Exception(f"Global search failed: {graph_result.get('message', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Global search error: {e}")
            raise
    
    async def _local_search(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """Perform local search using entity relationships."""
        try:
            # Execute GraphRAG local search
            graph_result = await self.graphrag_service.query_local_context(
                query=request.question,
                conversation_history=[]  # Could be extended to include chat history
            )
            
            if graph_result["success"]:
                # Create graph search result
                graph_insights = GraphSearchResult(
                    answer=graph_result["answer"],
                    search_type="local",
                    context_data=graph_result.get("context_data", {}),
                    completion_time=graph_result.get("completion_time"),
                    llm_calls=graph_result.get("llm_calls"),
                    prompt_tokens=graph_result.get("prompt_tokens")
                )
                
                # Get supporting vector sources for attribution
                vector_sources = await self._get_supporting_sources(request, graph_result["answer"])
                
                # Enhance sources with entity/relationship information
                enhanced_sources = await self._enhance_sources_with_entities(vector_sources, request)
                
                return EnhancedChatResponse(
                    answer=graph_result["answer"],
                    sources=enhanced_sources,
                    timestamp=datetime.now(),
                    search_type="local",
                    graph_insights=graph_insights
                )
            
            else:
                raise Exception(f"Local search failed: {graph_result.get('message', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Local search error: {e}")
            raise
    
    async def _vector_search(self, request: EnhancedChatRequest) -> EnhancedChatResponse:
        """Perform traditional vector search."""
        try:
            # Convert to original chat request format
            from models import ChatRequest
            chat_request = ChatRequest(
                question=request.question,
                document_ids=request.document_ids
            )
            
            # Execute vector search
            vector_result = self.vector_service.strategic_search(
                query=request.question,
                document_ids=request.document_ids
            )
            
            # Convert sources to enhanced format
            enhanced_sources = []
            for source in vector_result:
                enhanced_source = EnhancedSourceInfo(
                    document_id=source.document_id,
                    document_name=source.document_name,
                    chunk_content=source.chunk_content,
                    relevance_score=source.relevance_score
                )
                enhanced_sources.append(enhanced_source)
            
            # Enhance sources with entity information if available
            if request.include_entities:
                enhanced_sources = await self._enhance_sources_with_entities(enhanced_sources, request)
            
            # Generate answer using AI service
            from services.ai_service import AIService
            ai_service = AIService()
            ai_response = await ai_service.generate_response(chat_request)
            
            return EnhancedChatResponse(
                answer=ai_response.answer,
                sources=enhanced_sources,
                timestamp=datetime.now(),
                search_type="vector"
            )
        
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            raise
    
    async def _get_supporting_sources(self, request: EnhancedChatRequest, answer: str) -> List[EnhancedSourceInfo]:
        """Get supporting vector sources for GraphRAG answers."""
        try:
            # Use vector search to find relevant sources
            vector_result = self.vector_service.strategic_search(
                query=request.question,
                document_ids=request.document_ids,
                max_results=5  # Fewer sources for GraphRAG answers
            )
            
            # Convert to enhanced format
            enhanced_sources = []
            for source in vector_result:
                enhanced_source = EnhancedSourceInfo(
                    document_id=source.document_id,
                    document_name=source.document_name,
                    chunk_content=source.chunk_content,
                    relevance_score=source.relevance_score
                )
                enhanced_sources.append(enhanced_source)
            
            return enhanced_sources
        
        except Exception as e:
            logger.error(f"Error getting supporting sources: {e}")
            return []
    
    async def _enhance_sources_with_entities(self, sources: List[EnhancedSourceInfo], request: EnhancedChatRequest) -> List[EnhancedSourceInfo]:
        """Enhance sources with entity and relationship information."""
        if not self.graphrag_service.is_available():
            return sources
        
        try:
            for source in sources:
                # Get document entities
                doc_entities = self.graphrag_service.get_document_entities(source.document_id)
                
                # Convert to EntityInfo objects and filter relevant ones
                source.entities = []
                for entity_data in doc_entities:
                    # Simple relevance check - entity mentioned in source content
                    if entity_data.get('name', '').lower() in source.chunk_content.lower():
                        entity = EntityInfo(
                            id=entity_data.get('id', ''),
                            name=entity_data.get('name', ''),
                            entity_type=entity_data.get('type', 'UNKNOWN'),
                            description=entity_data.get('description'),
                            source_document_id=source.document_id,
                            source_document_name=source.document_name,
                            attributes=entity_data
                        )
                        source.entities.append(entity)
                
                # Get relationships for entities in this source
                if request.include_relationships:
                    source.relationships = []
                    for entity in source.entities:
                        entity_rels = self.graphrag_service.get_entity_relationships(entity.name)
                        for rel_data in entity_rels:
                            relationship = RelationshipInfo(
                                id=f"{entity.name}_{rel_data.get('target', '')}",
                                source_entity=entity.name,
                                target_entity=rel_data.get('target', ''),
                                relationship_type=rel_data.get('relationship', 'related to'),
                                weight=rel_data.get('weight', 1.0),
                                source_document_id=source.document_id,
                                attributes=rel_data.get('metadata', {})
                            )
                            source.relationships.append(relationship)
            
            return sources
        
        except Exception as e:
            logger.error(f"Error enhancing sources with entities: {e}")
            return sources
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search performance and availability."""
        return {
            "graphrag_available": self.graphrag_service.is_available(),
            "vector_search_available": True,
            "graph_statistics": self.graphrag_service.get_graph_statistics(),
            "search_strategies": ["global", "local", "vector", "auto"],
            "fallback_enabled": self.fallback_to_vector
        }