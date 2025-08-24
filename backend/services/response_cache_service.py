"""Intelligent Response Caching Service for AI optimization."""
import hashlib
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import pickle
import gzip
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

from config import config
from models import ChatRequest, ChatResponse, EnhancedChatRequest, EnhancedChatResponse

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""
    EXACT_MATCH = "exact_match"          # Exact query match
    SEMANTIC_MATCH = "semantic_match"    # Semantic similarity match
    TEMPLATE_MATCH = "template_match"    # Template-based match
    HYBRID = "hybrid"                    # Combination of strategies


@dataclass
class CacheEntry:
    """Cache entry structure."""
    cache_key: str
    original_query: str
    query_hash: str
    response_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    expiry_time: Optional[datetime]
    cache_hit_score: float
    provider: str
    search_type: str
    document_context: Optional[List[str]] = None


@dataclass
class CachePerformanceMetrics:
    """Cache performance tracking."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_response_time_cached: float
    avg_response_time_uncached: float
    time_saved: float
    storage_size_mb: float
    cleanup_operations: int


class ResponseCacheService:
    """Intelligent caching system for AI responses with multiple strategies."""
    
    def __init__(self):
        self.cache_dir = config.project_root / "data" / "response_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache subdirectories for different strategies
        self.exact_cache_dir = self.cache_dir / "exact"
        self.semantic_cache_dir = self.cache_dir / "semantic"
        self.template_cache_dir = self.cache_dir / "template"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for cache_subdir in [self.exact_cache_dir, self.semantic_cache_dir, 
                           self.template_cache_dir, self.metadata_dir]:
            cache_subdir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = self._load_cache_config()
        
        # Semantic similarity model for cache matching
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # In-memory cache for frequently accessed entries
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_size = 100  # Max entries in memory
        
        # Performance tracking
        self.performance_metrics = CachePerformanceMetrics(
            total_requests=0,
            cache_hits=0,
            cache_misses=0,
            hit_rate=0.0,
            avg_response_time_cached=0.0,
            avg_response_time_uncached=0.0,
            time_saved=0.0,
            storage_size_mb=0.0,
            cleanup_operations=0
        )
        
        # Query templates for template matching
        self.query_templates = self._initialize_query_templates()
        
        # Load existing cache metadata
        self._load_cache_metadata()
    
    def _load_cache_config(self) -> Dict[str, Any]:
        """Load cache configuration."""
        return {
            "enabled": True,
            "default_ttl_hours": 24,           # Default time-to-live
            "max_cache_size_mb": 500,          # Maximum cache size
            "semantic_similarity_threshold": 0.8,  # Threshold for semantic matches
            "template_similarity_threshold": 0.85, # Threshold for template matches
            "max_entries_per_strategy": 1000,  # Max entries per cache strategy
            "cleanup_interval_hours": 6,       # How often to run cleanup
            "memory_cache_enabled": True,
            "compression_enabled": True,       # Compress cache files
            "strategies": {
                "exact_match": {"enabled": True, "weight": 1.0},
                "semantic_match": {"enabled": True, "weight": 0.9},
                "template_match": {"enabled": True, "weight": 0.8}
            }
        }
    
    def _initialize_query_templates(self) -> List[Dict[str, Any]]:
        """Initialize common query templates for pattern matching."""
        return [
            {
                "template": "What is {concept}?",
                "pattern": r"what is (\\w+)\\?",
                "type": "definition",
                "priority": 0.9
            },
            {
                "template": "How does {process} work?",
                "pattern": r"how does (\\w+) work\\?",
                "type": "process_explanation",
                "priority": 0.9
            },
            {
                "template": "What are the main {aspects} of {topic}?",
                "pattern": r"what are the main (\\w+) of (\\w+)\\?",
                "type": "aspect_analysis",
                "priority": 0.8
            },
            {
                "template": "Compare {item1} and {item2}",
                "pattern": r"compare (\\w+) and (\\w+)",
                "type": "comparison",
                "priority": 0.8
            },
            {
                "template": "Summarize {content}",
                "pattern": r"summarize (\\w+)",
                "type": "summarization",
                "priority": 0.7
            }
        ]
    
    def _load_cache_metadata(self):
        """Load cache metadata and performance metrics."""
        metadata_file = self.metadata_dir / "cache_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Update performance metrics
                    if "performance_metrics" in data:
                        metrics_data = data["performance_metrics"]
                        for key, value in metrics_data.items():
                            if hasattr(self.performance_metrics, key):
                                setattr(self.performance_metrics, key, value)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata and performance metrics."""
        metadata_file = self.metadata_dir / "cache_metadata.json"
        try:
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "performance_metrics": asdict(self.performance_metrics),
                "config": self.config
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def generate_cache_key(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        strategy: CacheStrategy = CacheStrategy.HYBRID
    ) -> str:
        """Generate cache key for a query and context."""
        
        # Normalize query for consistent caching
        normalized_query = self._normalize_query(query)
        
        # Create base hash
        hash_input = normalized_query
        
        # Add context to hash if provided
        if context:
            # Sort context keys for consistent hashing
            sorted_context = json.dumps(context, sort_keys=True)
            hash_input += f"|{sorted_context}"
        
        # Add strategy to hash
        hash_input += f"|{strategy.value}"
        
        # Generate SHA-256 hash
        cache_key = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return cache_key
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation variations
        punctuation_replacements = {
            "?": "",
            "!": "",
            ".": "",
            ",": "",
            ";": "",
            ":": ""
        }
        
        for old, new in punctuation_replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    async def get_cached_response(
        self, 
        request: Union[ChatRequest, EnhancedChatRequest],
        strategy: CacheStrategy = CacheStrategy.HYBRID
    ) -> Optional[Union[ChatResponse, EnhancedChatResponse]]:
        """Retrieve cached response if available."""
        
        if not self.config["enabled"]:
            return None
        
        start_time = time.time()
        
        # Generate context for cache key
        context = self._extract_request_context(request)
        cache_key = self.generate_cache_key(request.question, context, strategy)
        
        # Try different cache strategies based on configuration
        cached_entry = None
        
        if strategy == CacheStrategy.HYBRID:
            # Try strategies in order of preference
            cached_entry = (
                await self._get_exact_match(request.question, context) or
                await self._get_semantic_match(request.question, context) or
                await self._get_template_match(request.question, context)
            )
        elif strategy == CacheStrategy.EXACT_MATCH:
            cached_entry = await self._get_exact_match(request.question, context)
        elif strategy == CacheStrategy.SEMANTIC_MATCH:
            cached_entry = await self._get_semantic_match(request.question, context)
        elif strategy == CacheStrategy.TEMPLATE_MATCH:
            cached_entry = await self._get_template_match(request.question, context)
        
        if cached_entry:
            # Update cache statistics
            cached_entry.access_count += 1
            cached_entry.last_accessed = datetime.now()
            
            # Update performance metrics
            self.performance_metrics.cache_hits += 1
            response_time = time.time() - start_time
            self._update_cached_response_time(response_time)
            
            # Convert cached data back to response object
            response = self._deserialize_response(cached_entry.response_data, request)
            
            # Update memory cache
            self.memory_cache[cache_key] = cached_entry
            self._manage_memory_cache()
            
            logger.info(f"Cache hit: {cached_entry.cache_key[:16]}... (score: {cached_entry.cache_hit_score:.3f})")
            return response
        
        # Cache miss
        self.performance_metrics.cache_misses += 1
        self.performance_metrics.total_requests += 1
        self._update_hit_rate()
        
        return None
    
    async def cache_response(
        self, 
        request: Union[ChatRequest, EnhancedChatRequest],
        response: Union[ChatResponse, EnhancedChatResponse],
        provider: str = "unknown",
        search_type: str = "unknown",
        ttl_hours: Optional[int] = None
    ):
        """Cache a response for future use."""
        
        if not self.config["enabled"]:
            return
        
        try:
            # Extract context and generate cache key
            context = self._extract_request_context(request)
            cache_key = self.generate_cache_key(request.question, context)
            
            # Serialize response data
            response_data = self._serialize_response(response)
            
            # Create cache entry
            ttl = ttl_hours or self.config["default_ttl_hours"]
            expiry_time = datetime.now() + timedelta(hours=ttl) if ttl > 0 else None
            
            cache_entry = CacheEntry(
                cache_key=cache_key,
                original_query=request.question,
                query_hash=hashlib.md5(request.question.encode()).hexdigest(),
                response_data=response_data,
                metadata={
                    "provider": provider,
                    "search_type": search_type,
                    "response_length": len(str(response_data)),
                    "source_count": len(getattr(response, 'sources', [])),
                    "cached_at": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                access_count=1,
                last_accessed=datetime.now(),
                expiry_time=expiry_time,
                cache_hit_score=1.0,  # Perfect score for exact matches
                provider=provider,
                search_type=search_type,
                document_context=context.get("document_ids") if context else None
            )
            
            # Save to appropriate cache directories
            await self._save_cache_entry(cache_entry)
            
            # Add to memory cache
            self.memory_cache[cache_key] = cache_entry
            self._manage_memory_cache()
            
            logger.info(f"Cached response: {cache_key[:16]}... (provider: {provider})")
            
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    async def _get_exact_match(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[CacheEntry]:
        """Get exact match from cache."""
        
        cache_key = self.generate_cache_key(query, context, CacheStrategy.EXACT_MATCH)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if not self._is_cache_entry_expired(entry):
                entry.cache_hit_score = 1.0  # Exact match
                return entry
        
        # Check disk cache
        cache_file = self.exact_cache_dir / f"{cache_key}.json.gz"
        if cache_file.exists():
            try:
                entry = await self._load_cache_entry(cache_file)
                if not self._is_cache_entry_expired(entry):
                    entry.cache_hit_score = 1.0
                    return entry
            except Exception as e:
                logger.warning(f"Failed to load exact cache entry: {e}")
        
        return None
    
    async def _get_semantic_match(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[CacheEntry]:
        """Get semantic similarity match from cache."""
        
        threshold = self.config["semantic_similarity_threshold"]
        
        # Get query embedding
        query_embedding = self.similarity_model.encode([query])[0]
        
        best_match = None
        best_score = 0.0
        
        # Check semantic cache directory
        for cache_file in self.semantic_cache_dir.glob("*.json.gz"):
            try:
                entry = await self._load_cache_entry(cache_file)
                
                if self._is_cache_entry_expired(entry):
                    continue
                
                # Calculate semantic similarity
                cached_query_embedding = self.similarity_model.encode([entry.original_query])[0]
                similarity = np.dot(query_embedding, cached_query_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_query_embedding)
                )
                
                # Check context compatibility
                context_compatible = self._is_context_compatible(context, entry.document_context)
                
                if similarity > threshold and similarity > best_score and context_compatible:
                    best_match = entry
                    best_score = similarity
                    
            except Exception as e:
                logger.warning(f"Error processing semantic cache entry {cache_file}: {e}")
                continue
        
        if best_match:
            best_match.cache_hit_score = best_score
            logger.info(f"Semantic match found with score: {best_score:.3f}")
        
        return best_match
    
    async def _get_template_match(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[CacheEntry]:
        """Get template-based match from cache."""
        import re
        
        # Find matching template
        matching_template = None
        for template in self.query_templates:
            pattern = template["pattern"]
            if re.search(pattern, query.lower()):
                matching_template = template
                break
        
        if not matching_template:
            return None
        
        # Look for cached responses of the same template type
        template_type = matching_template["type"]
        threshold = self.config["template_similarity_threshold"]
        
        best_match = None
        best_score = 0.0
        
        for cache_file in self.template_cache_dir.glob("*.json.gz"):
            try:
                entry = await self._load_cache_entry(cache_file)
                
                if self._is_cache_entry_expired(entry):
                    continue
                
                # Check if cached query matches the same template
                cached_template = self._identify_query_template(entry.original_query)
                if cached_template and cached_template["type"] == template_type:
                    
                    # Calculate template-based similarity
                    similarity = self._calculate_template_similarity(query, entry.original_query, matching_template)
                    
                    context_compatible = self._is_context_compatible(context, entry.document_context)
                    
                    if similarity > threshold and similarity > best_score and context_compatible:
                        best_match = entry
                        best_score = similarity
                        
            except Exception as e:
                logger.warning(f"Error processing template cache entry {cache_file}: {e}")
                continue
        
        if best_match:
            best_match.cache_hit_score = best_score * matching_template["priority"]
            logger.info(f"Template match found with score: {best_score:.3f}")
        
        return best_match
    
    def _identify_query_template(self, query: str) -> Optional[Dict[str, Any]]:
        """Identify which template a query matches."""
        import re
        
        for template in self.query_templates:
            pattern = template["pattern"]
            if re.search(pattern, query.lower()):
                return template
        return None
    
    def _calculate_template_similarity(
        self, 
        query1: str, 
        query2: str, 
        template: Dict[str, Any]
    ) -> float:
        """Calculate similarity between queries of the same template type."""
        import re
        
        # Extract template variables from both queries
        pattern = template["pattern"]
        
        match1 = re.search(pattern, query1.lower())
        match2 = re.search(pattern, query2.lower())
        
        if not match1 or not match2:
            return 0.0
        
        # Compare extracted variables using semantic similarity
        vars1 = match1.groups()
        vars2 = match2.groups()
        
        if len(vars1) != len(vars2):
            return 0.0
        
        # Calculate average similarity of template variables
        similarities = []
        for var1, var2 in zip(vars1, vars2):
            if var1 == var2:
                similarities.append(1.0)
            else:
                # Use word embeddings for variable similarity
                try:
                    emb1 = self.similarity_model.encode([var1])[0]
                    emb2 = self.similarity_model.encode([var2])[0]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(max(0, sim))
                except:
                    similarities.append(0.5)  # Neutral similarity if embedding fails
        
        return sum(similarities) / len(similarities)
    
    def _extract_request_context(self, request: Union[ChatRequest, EnhancedChatRequest]) -> Dict[str, Any]:
        """Extract context from request for cache key generation."""
        context = {}
        
        if hasattr(request, 'document_ids') and request.document_ids:
            context['document_ids'] = sorted(request.document_ids)
        
        if hasattr(request, 'search_type') and request.search_type:
            context['search_type'] = request.search_type
        
        if hasattr(request, 'include_entities'):
            context['include_entities'] = request.include_entities
        
        if hasattr(request, 'include_relationships'):
            context['include_relationships'] = request.include_relationships
        
        return context
    
    def _is_context_compatible(
        self, 
        current_context: Optional[Dict[str, Any]], 
        cached_context: Optional[List[str]]
    ) -> bool:
        """Check if current context is compatible with cached context."""
        
        if not current_context and not cached_context:
            return True
        
        if not current_context or not cached_context:
            return False
        
        # Check document context compatibility
        current_docs = current_context.get('document_ids', [])
        
        if not current_docs and not cached_context:
            return True
        
        if not current_docs or not cached_context:
            return False
        
        # Allow cache hit if there's significant overlap in documents
        current_set = set(current_docs)
        cached_set = set(cached_context)
        overlap = len(current_set.intersection(cached_set))
        total = len(current_set.union(cached_set))
        
        overlap_ratio = overlap / total if total > 0 else 0
        return overlap_ratio >= 0.7  # 70% overlap required
    
    def _serialize_response(self, response: Union[ChatResponse, EnhancedChatResponse]) -> Dict[str, Any]:
        """Serialize response object for caching."""
        
        response_dict = {
            "answer": response.answer,
            "timestamp": response.timestamp.isoformat() if response.timestamp else None,
            "response_type": type(response).__name__
        }
        
        # Handle sources
        if hasattr(response, 'sources') and response.sources:
            response_dict["sources"] = []
            for source in response.sources:
                source_dict = {
                    "document_id": source.document_id,
                    "document_name": source.document_name,
                    "chunk_content": source.chunk_content,
                    "relevance_score": source.relevance_score
                }
                
                # Handle enhanced source info
                if hasattr(source, 'entities') and source.entities:
                    source_dict["entities"] = [
                        {
                            "name": entity.name,
                            "entity_type": entity.entity_type,
                            "description": entity.description
                        } for entity in source.entities
                    ]
                
                if hasattr(source, 'relationships') and source.relationships:
                    source_dict["relationships"] = [
                        {
                            "source_entity": rel.source_entity,
                            "target_entity": rel.target_entity,
                            "relationship_type": rel.relationship_type
                        } for rel in source.relationships
                    ]
                
                response_dict["sources"].append(source_dict)
        
        # Handle enhanced response attributes
        if hasattr(response, 'search_type'):
            response_dict["search_type"] = response.search_type
        
        if hasattr(response, 'graph_insights'):
            response_dict["graph_insights"] = response.graph_insights.__dict__ if response.graph_insights else None
        
        return response_dict
    
    def _deserialize_response(
        self, 
        response_data: Dict[str, Any], 
        original_request: Union[ChatRequest, EnhancedChatRequest]
    ) -> Union[ChatResponse, EnhancedChatResponse]:
        """Deserialize cached response data back to response object."""
        
        # Import here to avoid circular imports
        from models import SourceInfo, EnhancedSourceInfo, EntityInfo, RelationshipInfo
        
        # Reconstruct sources
        sources = []
        if "sources" in response_data:
            for source_data in response_data["sources"]:
                
                # Check if this is enhanced source info
                if "entities" in source_data or "relationships" in source_data:
                    # Create enhanced source
                    entities = []
                    if "entities" in source_data:
                        for entity_data in source_data["entities"]:
                            entities.append(EntityInfo(
                                id=f"cached_{entity_data['name']}",
                                name=entity_data["name"],
                                entity_type=entity_data["entity_type"],
                                description=entity_data.get("description"),
                                source_document_id=source_data["document_id"],
                                source_document_name=source_data["document_name"]
                            ))
                    
                    relationships = []
                    if "relationships" in source_data:
                        for rel_data in source_data["relationships"]:
                            relationships.append(RelationshipInfo(
                                id=f"cached_{rel_data['source_entity']}_{rel_data['target_entity']}",
                                source_entity=rel_data["source_entity"],
                                target_entity=rel_data["target_entity"],
                                relationship_type=rel_data["relationship_type"],
                                source_document_id=source_data["document_id"]
                            ))
                    
                    source = EnhancedSourceInfo(
                        document_id=source_data["document_id"],
                        document_name=source_data["document_name"],
                        chunk_content=source_data["chunk_content"],
                        relevance_score=source_data["relevance_score"],
                        entities=entities,
                        relationships=relationships
                    )
                else:
                    # Create basic source
                    source = SourceInfo(
                        document_id=source_data["document_id"],
                        document_name=source_data["document_name"],
                        chunk_content=source_data["chunk_content"],
                        relevance_score=source_data["relevance_score"]
                    )
                
                sources.append(source)
        
        # Create appropriate response type
        if isinstance(original_request, EnhancedChatRequest):
            response = EnhancedChatResponse(
                answer=response_data["answer"],
                sources=sources,
                timestamp=datetime.now(),  # Use current timestamp for cached responses
                search_type=response_data.get("search_type", "cached")
            )
        else:
            response = ChatResponse(
                answer=response_data["answer"],
                sources=sources,
                timestamp=datetime.now()
            )
        
        return response
    
    async def _save_cache_entry(self, cache_entry: CacheEntry):
        """Save cache entry to appropriate directories."""
        
        # Save to exact match cache
        exact_file = self.exact_cache_dir / f"{cache_entry.cache_key}.json.gz"
        await self._write_cache_file(exact_file, cache_entry)
        
        # Save to semantic cache
        semantic_file = self.semantic_cache_dir / f"{cache_entry.cache_key}.json.gz"
        await self._write_cache_file(semantic_file, cache_entry)
        
        # Save to template cache if applicable
        template = self._identify_query_template(cache_entry.original_query)
        if template:
            template_file = self.template_cache_dir / f"{cache_entry.cache_key}.json.gz"
            await self._write_cache_file(template_file, cache_entry)
    
    async def _write_cache_file(self, file_path: Path, cache_entry: CacheEntry):
        """Write cache entry to compressed file."""
        try:
            cache_data = asdict(cache_entry)
            # Convert datetime objects to ISO strings
            cache_data['timestamp'] = cache_entry.timestamp.isoformat()
            cache_data['last_accessed'] = cache_entry.last_accessed.isoformat()
            if cache_entry.expiry_time:
                cache_data['expiry_time'] = cache_entry.expiry_time.isoformat()
            
            json_data = json.dumps(cache_data, indent=2).encode()
            
            if self.config["compression_enabled"]:
                with gzip.open(file_path, 'wb') as f:
                    f.write(json_data)
            else:
                with open(file_path.with_suffix('.json'), 'w') as f:
                    json.dump(cache_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to write cache file {file_path}: {e}")
    
    async def _load_cache_entry(self, file_path: Path) -> CacheEntry:
        """Load cache entry from file."""
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    data = json.loads(f.read().decode())
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            
            # Convert ISO strings back to datetime objects
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
            if data.get('expiry_time'):
                data['expiry_time'] = datetime.fromisoformat(data['expiry_time'])
            
            return CacheEntry(**data)
            
        except Exception as e:
            logger.error(f"Failed to load cache entry from {file_path}: {e}")
            raise
    
    def _is_cache_entry_expired(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        if not cache_entry.expiry_time:
            return False
        return datetime.now() > cache_entry.expiry_time
    
    def _manage_memory_cache(self):
        """Manage memory cache size and eviction."""
        if len(self.memory_cache) <= self.memory_cache_size:
            return
        
        # Evict least recently used entries
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries
        entries_to_remove = len(sorted_entries) - self.memory_cache_size
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.memory_cache[cache_key]
    
    def _update_cached_response_time(self, response_time: float):
        """Update cached response time statistics."""
        current_avg = self.performance_metrics.avg_response_time_cached
        hit_count = self.performance_metrics.cache_hits
        
        if hit_count == 1:
            self.performance_metrics.avg_response_time_cached = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_metrics.avg_response_time_cached = (
                alpha * response_time + (1 - alpha) * current_avg
            )
    
    def _update_hit_rate(self):
        """Update cache hit rate."""
        total = self.performance_metrics.total_requests
        hits = self.performance_metrics.cache_hits
        
        if total > 0:
            self.performance_metrics.hit_rate = hits / total
    
    async def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        start_time = time.time()
        
        cleaned_files = 0
        total_files = 0
        freed_space_mb = 0
        
        for cache_dir in [self.exact_cache_dir, self.semantic_cache_dir, self.template_cache_dir]:
            for cache_file in cache_dir.glob("*.json*"):
                total_files += 1
                try:
                    entry = await self._load_cache_entry(cache_file)
                    if self._is_cache_entry_expired(entry):
                        file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                        cache_file.unlink()
                        cleaned_files += 1
                        freed_space_mb += file_size
                        
                except Exception as e:
                    logger.warning(f"Error processing cache file {cache_file}: {e}")
                    continue
        
        # Update performance metrics
        self.performance_metrics.cleanup_operations += 1
        
        cleanup_time = time.time() - start_time
        
        return {
            "cleaned_files": cleaned_files,
            "total_files": total_files,
            "freed_space_mb": round(freed_space_mb, 2),
            "cleanup_time": round(cleanup_time, 2),
            "message": f"Cleaned {cleaned_files} expired entries, freed {freed_space_mb:.2f} MB"
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        # Calculate current cache size
        total_size_mb = 0
        file_counts = {}
        
        for cache_dir_name, cache_dir in [
            ("exact", self.exact_cache_dir),
            ("semantic", self.semantic_cache_dir), 
            ("template", self.template_cache_dir)
        ]:
            files = list(cache_dir.glob("*.json*"))
            file_counts[cache_dir_name] = len(files)
            
            for file_path in files:
                try:
                    total_size_mb += file_path.stat().st_size / (1024 * 1024)
                except:
                    continue
        
        self.performance_metrics.storage_size_mb = total_size_mb
        
        return {
            "performance_metrics": asdict(self.performance_metrics),
            "file_counts": file_counts,
            "memory_cache_entries": len(self.memory_cache),
            "configuration": self.config,
            "cache_strategies_enabled": [
                strategy for strategy, config in self.config["strategies"].items()
                if config["enabled"]
            ]
        }
    
    def update_cache_config(self, new_config: Dict[str, Any]):
        """Update cache configuration."""
        self.config.update(new_config)
        self._save_cache_metadata()
    
    async def invalidate_cache_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        invalidated_count = 0
        
        # Check pattern against cached queries
        for cache_dir in [self.exact_cache_dir, self.semantic_cache_dir, self.template_cache_dir]:
            for cache_file in cache_dir.glob("*.json*"):
                try:
                    entry = await self._load_cache_entry(cache_file)
                    if pattern.lower() in entry.original_query.lower():
                        cache_file.unlink()
                        invalidated_count += 1
                        
                        # Remove from memory cache
                        if entry.cache_key in self.memory_cache:
                            del self.memory_cache[entry.cache_key]
                            
                except Exception as e:
                    logger.warning(f"Error invalidating cache file {cache_file}: {e}")
                    continue
        
        return invalidated_count