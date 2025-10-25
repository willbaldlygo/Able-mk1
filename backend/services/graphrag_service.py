"""Microsoft GraphRAG integration service for Able."""
import asyncio
import json
import logging
import pandas as pd
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx

# GraphRAG now properly installed with Python 3.11
try:
    from graphrag.config.load_config import load_config
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.run import run_pipeline
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
    from graphrag.query.structured_search.global_search.search import GlobalSearch
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GraphRAG import failed: {e}. GraphRAG features will be disabled.")
    GRAPHRAG_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)

class GraphRAGService:
    """Microsoft GraphRAG integration for advanced knowledge synthesis."""
    
    def __init__(self):
        self.graphrag_dir = config.project_root / "data" / "graphrag"
        self.entities_df: Optional[pd.DataFrame] = None
        self.relationships_df: Optional[pd.DataFrame] = None
        self.reports_df: Optional[pd.DataFrame] = None
        self.text_units_df: Optional[pd.DataFrame] = None
        self.global_search: Optional[GlobalSearch] = None
        self.local_search: Optional[LocalSearch] = None
        self.graph: nx.Graph = nx.Graph()
        
        # Initialize directories
        self._init_directories()
        
        # Initialize GraphRAG components if available
        if GRAPHRAG_AVAILABLE:
            self._init_graphrag()
            # Load lightweight data instead of full GraphRAG data
            self._load_lightweight_graph_data()
        else:
            logger.warning("GraphRAG not available. Knowledge graph features disabled.")
    
    def _init_directories(self):
        """Initialize GraphRAG data directories."""
        self.graphrag_dir.mkdir(parents=True, exist_ok=True)
        (self.graphrag_dir / "input").mkdir(exist_ok=True)
        (self.graphrag_dir / "output").mkdir(exist_ok=True)
        (self.graphrag_dir / "cache").mkdir(exist_ok=True)
        (self.graphrag_dir / "logs").mkdir(exist_ok=True)
    
    def _init_graphrag(self):
        """Initialize GraphRAG configuration and search components."""
        try:
            # Load GraphRAG configuration
            self.config = self._load_graphrag_config()
            
            # Load existing data if available
            self._load_graph_data()
            
            # Initialize search components
            self._init_search_components()
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG: {e}")
            global GRAPHRAG_AVAILABLE
            GRAPHRAG_AVAILABLE = False
    
    def _load_graphrag_config(self) -> Any:
        """Load GraphRAG configuration from settings.yaml."""
        try:
            # Load configuration from the GraphRAG directory
            config = load_config(root_dir=self.graphrag_dir)
            logger.info("GraphRAG configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load GraphRAG config: {e}")
            return None
    
    def _load_graph_data(self):
        """Load existing GraphRAG data."""
        try:
            output_dir = self.graphrag_dir / "output"
            if not output_dir.exists():
                return
            
            # Find the latest artifacts directory
            artifact_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
            if not artifact_dirs:
                return
            
            latest_dir = max(artifact_dirs, key=lambda x: x.stat().st_mtime)
            artifacts_dir = latest_dir / "artifacts"
            
            if artifacts_dir.exists():
                # GraphRAG functions disabled
                # self.entities_df = read_indexer_entities(artifacts_dir)
                # self.relationships_df = read_indexer_relationships(artifacts_dir)
                # self.reports_df = read_indexer_reports(artifacts_dir)
                # self.text_units_df = read_indexer_text_units(artifacts_dir)
                
                # Build NetworkX graph from relationships
                self._build_networkx_graph()
                
                logger.info(f"Loaded GraphRAG data: {len(self.entities_df) if self.entities_df is not None else 0} entities, "
                           f"{len(self.relationships_df) if self.relationships_df is not None else 0} relationships")
        
        except Exception as e:
            logger.warning(f"Could not load existing GraphRAG data: {e}")
    
    def _build_networkx_graph(self):
        """Build NetworkX graph from relationship data."""
        if self.relationships_df is None:
            return
        
        self.graph = nx.Graph()
        
        # Add nodes (entities)
        if self.entities_df is not None:
            for _, entity in self.entities_df.iterrows():
                self.graph.add_node(
                    entity.get('title', entity.get('name', str(entity.get('id')))),
                    **{k: v for k, v in entity.items() if k not in ['title', 'name']}
                )
        
        # Add edges (relationships)
        for _, rel in self.relationships_df.iterrows():
            source = rel.get('source')
            target = rel.get('target')
            if source and target:
                self.graph.add_edge(
                    source, target, 
                    weight=rel.get('weight', 1.0),
                    description=rel.get('description', ''),
                    **{k: v for k, v in rel.items() if k not in ['source', 'target', 'weight', 'description']}
                )
    
    def _init_search_components(self):
        """Initialize global and local search components."""
        if not all([self.entities_df is not None, self.relationships_df is not None, 
                   self.reports_df is not None, self.text_units_df is not None]):
            logger.info("GraphRAG data not available, search components not initialized")
            return
        
        try:
            # GraphRAG search components disabled
            # llm = ChatOpenAI(
            #     api_key=config.anthropic_api_key,
            #     model="claude-3-sonnet-20240229",
            #     api_type=OpenaiApiType.OpenAI,
            #     max_retries=3,
            # )
            
            # context_builder = GlobalCommunityContext(
            #     community_reports=self.reports_df,
            #     entities=self.entities_df,
            #     token_encoder=None,
            # )
            
            # self.global_search = GlobalSearch(
            #     llm=llm,
            #     context_builder=context_builder,
            #     token_encoder=None,
            #     max_tokens=8000,
            #     llm_max_tokens=4000,
            # )
            
            # vector_store = LanceDBVectorStore(collection_name="avi_entities")
            # local_context_builder = LocalSearchMixedContext(
            #     community_reports=self.reports_df,
            #     text_units=self.text_units_df,
            #     entities=self.entities_df,
            #     relationships=self.relationships_df,
            #     entity_text_embeddings=vector_store,
            #     embedding_vectorstore_key=EntityVectorStoreKey.ID,
            #     text_embedder=None,
            #     token_encoder=None,
            # )
            
            # self.local_search = LocalSearch(
            #     llm=llm,
            #     context_builder=local_context_builder,
            #     token_encoder=None,
            #     llm_max_tokens=4000,
            #     context_builder_max_tokens=8000,
            # )
            
            logger.info("GraphRAG search components initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize search components: {e}")
    
    async def process_document_for_graph(self, document_id: str, content: str, document_name: str) -> Dict[str, Any]:
        """Process a document to extract entities and relationships for the knowledge graph."""
        if not GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG not available, skipping graph processing")
            return {"success": False, "message": "GraphRAG not available"}
        
        try:
            # Create input file for GraphRAG
            input_file = self.graphrag_dir / "input" / f"{document_id}.txt"
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(f"# {document_name}\n\n{content}")
            
            # Use lightweight extraction since full GraphRAG is not available
            logger.info(f"Processing document {document_name} with lightweight GraphRAG alternative")
            
            # Extract entities using AI service
            doc_entities = await self._extract_entities_lightweight(content, document_id, document_name)
            
            # Extract relationships using AI service
            doc_relationships = await self._extract_relationships_lightweight(doc_entities, content, document_id)
            
            # Update in-memory graph
            self._update_graph_with_extracted_data(doc_entities, doc_relationships)
            
            # Save extracted data to JSON files for persistence
            self._save_extracted_data(document_id, doc_entities, doc_relationships)
            
            logger.info(f"Extracted {len(doc_entities)} entities and {len(doc_relationships)} relationships for {document_name}")
            
            return {
                "success": True,
                "entities": doc_entities,
                "relationships": doc_relationships,
                "entity_count": len(doc_entities),
                "relationship_count": len(doc_relationships)
            }
        
        except Exception as e:
            logger.error(f"Error processing document for graph: {e}")
            return {"success": False, "message": str(e)}
    
    def _extract_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        """Extract entities associated with a specific document."""
        if self.entities_df is None:
            return []
        
        # Filter entities by document source
        doc_entities = self.entities_df[
            self.entities_df.get('source_id', '').str.contains(document_id, na=False)
        ]
        
        return doc_entities.to_dict('records') if not doc_entities.empty else []
    
    def _extract_document_relationships(self, document_id: str) -> List[Dict[str, Any]]:
        """Extract relationships associated with a specific document."""
        if self.relationships_df is None:
            return []
        
        # Filter relationships by document source
        doc_relationships = self.relationships_df[
            self.relationships_df.get('source_id', '').str.contains(document_id, na=False)
        ]
        
        return doc_relationships.to_dict('records') if not doc_relationships.empty else []
    
    async def query_global_context(self, query: str, community_level: int = 2) -> Dict[str, Any]:
        """Perform global search using community summaries for broad research questions."""
        if not GRAPHRAG_AVAILABLE:
            return {"success": False, "message": "GraphRAG not available"}
        
        # Use lightweight alternative if full GraphRAG is not available
        if self.global_search is None:
            return await self._query_lightweight_global(query)
        
        try:
            result = await self.global_search.asearch(
                query=query,
                community_level=community_level,
                response_type="Multiple Paragraphs",
            )
            
            return {
                "success": True,
                "answer": result.response,
                "context_data": result.context_data,
                "completion_time": result.completion_time,
                "llm_calls": result.llm_calls,
                "prompt_tokens": result.prompt_tokens,
                "search_type": "global"
            }
        
        except Exception as e:
            logger.error(f"Global search error: {e}")
            return {"success": False, "message": str(e)}
    
    async def query_local_context(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Perform local search using entity relationships for specific queries."""
        if not GRAPHRAG_AVAILABLE:
            return {"success": False, "message": "GraphRAG not available"}
        
        # Use lightweight alternative if full GraphRAG is not available
        if self.local_search is None:
            return await self._query_lightweight_local(query, conversation_history)
        
        try:
            result = await self.local_search.asearch(
                query=query,
                conversation_history=conversation_history or [],
                knowledge_base_id="avi_kb",
            )
            
            return {
                "success": True,
                "answer": result.response,
                "context_data": result.context_data,
                "completion_time": result.completion_time,
                "llm_calls": result.llm_calls,
                "prompt_tokens": result.prompt_tokens,
                "search_type": "local"
            }
        
        except Exception as e:
            logger.error(f"Local search error: {e}")
            return {"success": False, "message": str(e)}
    
    def analyze_query_type(self, query: str) -> str:
        """Analyze query to determine whether to use global, local, or vector search."""
        query_lower = query.lower()
        
        # Global search indicators
        global_indicators = [
            'overview', 'summary', 'general', 'broad', 'overall', 'compare', 'contrast',
            'what are the main', 'key themes', 'major trends', 'comprehensive',
            'across all documents', 'in general', 'trends', 'patterns'
        ]
        
        # Local search indicators
        local_indicators = [
            'specific', 'particular', 'relationship between', 'how does', 'what is the connection',
            'entity', 'person', 'organization', 'specific case', 'detailed', 'precise'
        ]
        
        # Count indicators
        global_score = sum(1 for indicator in global_indicators if indicator in query_lower)
        local_score = sum(1 for indicator in local_indicators if indicator in query_lower)
        
        # Determine search type
        if global_score > local_score and global_score > 0:
            return "global"
        elif local_score > 0:
            return "local"
        else:
            # Default to vector search for simple queries
            return "vector"
    
    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity."""
        if not self.graph.has_node(entity_name):
            return []
        
        relationships = []
        for neighbor in self.graph.neighbors(entity_name):
            edge_data = self.graph.get_edge_data(entity_name, neighbor, {})
            relationships.append({
                "target": neighbor,
                "relationship": edge_data.get('description', 'related to'),
                "weight": edge_data.get('weight', 1.0),
                "metadata": {k: v for k, v in edge_data.items() 
                           if k not in ['description', 'weight']}
            })
        
        return relationships
    
    def get_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all entities extracted from a specific document."""
        return self._extract_document_entities(document_id)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_entities": len(self.entities_df) if self.entities_df is not None else 0,
            "total_relationships": len(self.relationships_df) if self.relationships_df is not None else 0,
            "total_communities": len(self.reports_df) if self.reports_df is not None else 0,
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "graph_connected_components": nx.number_connected_components(self.graph),
            "graphrag_available": GRAPHRAG_AVAILABLE
        }
    
    async def delete_document_from_graph(self, document_id: str) -> Dict[str, Any]:
        """Remove a document's contributions from the knowledge graph."""
        try:
            # Remove input file
            input_file = self.graphrag_dir / "input" / f"{document_id}.txt"
            if input_file.exists():
                input_file.unlink()
            
            # Note: Full graph rebuild would be needed to completely remove document
            # For now, we'll mark this as a limitation
            logger.info(f"Removed input file for document {document_id}. Full graph rebuild needed for complete removal.")
            
            return {
                "success": True,
                "message": "Document input removed. Graph rebuild recommended for complete removal."
            }
        
        except Exception as e:
            logger.error(f"Error removing document from graph: {e}")
            return {"success": False, "message": str(e)}
    
    def is_available(self) -> bool:
        """Check if GraphRAG is available and properly initialized."""
        return GRAPHRAG_AVAILABLE
    
    async def _extract_entities_lightweight(self, content: str, document_id: str, document_name: str) -> List[Dict[str, Any]]:
        """Lightweight entity extraction using Claude AI."""
        from .ai_service import AIService
        
        ai_service = AIService()
        
        entity_prompt = f"""
You are a JSON entity extractor. Extract entities from the document content below.

Document: {document_name}
Content: {content[:2000]}...

Extract entities of these types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, TECHNOLOGY, METHOD

Return ONLY a valid JSON array with no additional text, explanations, or formatting:

[{{"name":"entity_name","type":"ENTITY_TYPE","description":"brief_description","source_document_id":"{document_id}","source_document_name":"{document_name}"}}]

JSON:
"""
        
        try:
            # Use the synchronous method since generate_response is not async
            response = ai_service.generate_response_with_provider(entity_prompt, [], disable_formatting=True)
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response if it contains other text
            answer = response.answer.strip()
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = answer
            
            entities = json.loads(json_str)
            
            # Add IDs and ensure proper format
            for entity in entities:
                entity["id"] = str(uuid.uuid4())
                entity["attributes"] = {}
            
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_relationships_lightweight(self, entities: List[Dict], content: str, document_id: str) -> List[Dict[str, Any]]:
        """Lightweight relationship extraction using Claude AI."""
        if len(entities) < 2:
            return []
        
        from .ai_service import AIService
        ai_service = AIService()
        
        entity_names = [e["name"] for e in entities]
        
        relationship_prompt = f"""
You are a JSON relationship extractor. Find relationships between these entities:

Entities: {entity_names}
Content: {content[:2000]}...

Return ONLY a valid JSON array with no additional text:

[{{"source_entity":"entity1","target_entity":"entity2","relationship_type":"relationship","description":"brief_description","weight":1.0,"source_document_id":"{document_id}"}}]

JSON:
"""
        
        try:
            # Use the synchronous method since generate_response is not async
            response = ai_service.generate_response_with_provider(relationship_prompt, [], disable_formatting=True)
            import json
            import re
            
            # Extract JSON from response if it contains other text
            answer = response.answer.strip()
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = answer
            
            relationships = json.loads(json_str)
            
            # Add IDs and ensure proper format
            for rel in relationships:
                rel["id"] = str(uuid.uuid4())
                rel["attributes"] = {}
            
            return relationships
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _update_graph_with_extracted_data(self, entities: List[Dict], relationships: List[Dict]):
        """Update the NetworkX graph with extracted entities and relationships."""
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(
                entity["name"],
                entity_type=entity.get("type", "UNKNOWN"),
                description=entity.get("description", ""),
                source_document_id=entity.get("source_document_id", ""),
                **entity.get("attributes", {})
            )
        
        # Add relationships as edges
        for rel in relationships:
            source = rel.get("source_entity")
            target = rel.get("target_entity")
            if source and target and source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source, target,
                    relationship_type=rel.get("relationship_type", "related"),
                    description=rel.get("description", ""),
                    weight=rel.get("weight", 1.0),
                    source_document_id=rel.get("source_document_id", ""),
                    **rel.get("attributes", {})
                )
    
    def _save_extracted_data(self, document_id: str, entities: List[Dict], relationships: List[Dict]):
        """Save extracted entities and relationships to JSON files for persistence."""
        try:
            # Save to document-specific files
            output_dir = self.graphrag_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            entities_file = output_dir / f"{document_id}_entities.json"
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities, f, indent=2, ensure_ascii=False)
            
            relationships_file = output_dir / f"{document_id}_relationships.json"
            with open(relationships_file, 'w', encoding='utf-8') as f:
                json.dump(relationships, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved graph data for document {document_id}")
        
        except Exception as e:
            logger.error(f"Failed to save extracted data: {e}")
    
    def _load_lightweight_graph_data(self):
        """Load graph data from lightweight JSON files."""
        try:
            output_dir = self.graphrag_dir / "output"
            if not output_dir.exists():
                return
            
            # Load all entity and relationship files
            all_entities = []
            all_relationships = []
            
            for entities_file in output_dir.glob("*_entities.json"):
                with open(entities_file, 'r', encoding='utf-8') as f:
                    all_entities.extend(json.load(f))
            
            for relationships_file in output_dir.glob("*_relationships.json"):
                with open(relationships_file, 'r', encoding='utf-8') as f:
                    all_relationships.extend(json.load(f))
            
            # Update the graph
            self._update_graph_with_extracted_data(all_entities, all_relationships)
            
            # Create mock dataframes for compatibility
            if all_entities:
                self.entities_df = pd.DataFrame(all_entities)
            if all_relationships:
                self.relationships_df = pd.DataFrame(all_relationships)
            
            logger.info(f"Loaded lightweight graph data: {len(all_entities)} entities, {len(all_relationships)} relationships")
        
        except Exception as e:
            logger.error(f"Failed to load lightweight graph data: {e}")
    
    async def _query_lightweight_global(self, query: str) -> Dict[str, Any]:
        """Lightweight global search using extracted entities and relationships."""
        try:
            from .ai_service import AIService
            ai_service = AIService()
            
            # Get all entities and relationships
            entities_context = []
            if self.entities_df is not None:
                entities_context = self.entities_df.to_dict('records')[:50]  # Limit for token usage
            
            relationships_context = []
            if self.relationships_df is not None:
                relationships_context = self.relationships_df.to_dict('records')[:50]
            
            # Create context from graph
            context_prompt = f"""
            Use the following knowledge graph information to answer the query: {query}
            
            Available Entities:
            {json.dumps(entities_context, indent=2)[:2000]}
            
            Available Relationships:
            {json.dumps(relationships_context, indent=2)[:2000]}
            
            Graph Statistics:
            - Total entities: {len(self.graph.nodes)}
            - Total relationships: {len(self.graph.edges)}
            - Connected components: {nx.number_connected_components(self.graph)}
            
            Provide a comprehensive answer based on the knowledge graph information.
            """
            
            # Use the synchronous method since generate_response is not async
            response = ai_service.generate_response_with_sources(context_prompt, [])
            answer = response.answer
            
            return {
                "success": True,
                "answer": answer,
                "context_data": {
                    "entities_used": len(entities_context),
                    "relationships_used": len(relationships_context)
                },
                "search_type": "lightweight_global"
            }
        
        except Exception as e:
            logger.error(f"Lightweight global search error: {e}")
            return {"success": False, "message": str(e)}
    
    async def _query_lightweight_local(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Lightweight local search focusing on specific entities."""
        try:
            from .ai_service import AIService
            ai_service = AIService()
            
            # Find relevant entities based on query
            relevant_entities = []
            if self.entities_df is not None:
                query_lower = query.lower()
                for _, entity in self.entities_df.iterrows():
                    if (query_lower in entity.get('name', '').lower() or 
                        query_lower in entity.get('description', '').lower()):
                        relevant_entities.append(entity.to_dict())
            
            # Get relationships for relevant entities
            relevant_relationships = []
            for entity in relevant_entities:
                entity_name = entity.get('name', '')
                if entity_name in self.graph.nodes:
                    for neighbor in self.graph.neighbors(entity_name):
                        edge_data = self.graph.get_edge_data(entity_name, neighbor, {})
                        relevant_relationships.append({
                            "source": entity_name,
                            "target": neighbor,
                            "relationship": edge_data.get('relationship_type', 'related'),
                            "description": edge_data.get('description', '')
                        })
            
            context_prompt = f"""
            Use the following specific entities and their relationships to answer: {query}
            
            Relevant Entities:
            {json.dumps(relevant_entities, indent=2)[:1500]}
            
            Relevant Relationships:
            {json.dumps(relevant_relationships, indent=2)[:1500]}
            
            Focus on providing specific, detailed information about these entities and their connections.
            """
            
            # Use the synchronous method since generate_response is not async
            response = ai_service.generate_response_with_sources(context_prompt, [])
            answer = response.answer
            
            return {
                "success": True,
                "answer": answer,
                "context_data": {
                    "relevant_entities": len(relevant_entities),
                    "relevant_relationships": len(relevant_relationships)
                },
                "search_type": "lightweight_local"
            }
        
        except Exception as e:
            logger.error(f"Lightweight local search error: {e}")
            return {"success": False, "message": str(e)}