"""Enhanced Vector Service with Optimized Embedding Strategies and Relevance Scoring."""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from config import config
from models import Document, SourceInfo
from services.query_analyzer import QueryAnalyzer


class EmbeddingStrategy(Enum):
    """Different embedding strategies for different content types."""
    SEMANTIC = "semantic"          # General semantic understanding
    TECHNICAL = "technical"        # Technical/scientific content
    LEGAL = "legal"               # Legal/formal documents
    CONVERSATIONAL = "conversational"  # Q&A style content
    MULTILINGUAL = "multilingual"  # Multiple languages


@dataclass
class RelevanceWeights:
    """Weights for different relevance scoring factors."""
    semantic_similarity: float = 0.40
    keyword_overlap: float = 0.20
    content_structure: float = 0.15
    document_authority: float = 0.10
    recency: float = 0.05
    length_quality: float = 0.10


@dataclass
class SearchConfiguration:
    """Configuration for search optimization."""
    embedding_strategy: EmbeddingStrategy
    relevance_weights: RelevanceWeights
    diversity_factor: float = 0.3
    rerank_results: bool = True
    use_query_expansion: bool = True
    max_results_per_doc: int = 3
    minimum_relevance_threshold: float = 0.1


class EnhancedVectorService:
    """Enhanced vector service with multiple embedding models and optimized relevance scoring."""
    
    def __init__(self):
        self.db_path = config.vectordb_dir
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize multiple embedding models for different strategies
        self.embedding_models = self._initialize_embedding_models()
        self.default_model = self.embedding_models[EmbeddingStrategy.SEMANTIC]
        
        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Create collections for different embedding strategies
        self.collections = self._initialize_collections()
        
        # Performance tracking
        self.performance_cache_dir = config.project_root / "data" / "vector_performance"
        self.performance_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Search configuration cache
        self.search_configs = self._load_search_configurations()
    
    def _initialize_embedding_models(self) -> Dict[EmbeddingStrategy, SentenceTransformer]:
        """Initialize different embedding models for various content types."""
        models = {}
        
        try:
            # Semantic (general purpose) - fast and accurate
            models[EmbeddingStrategy.SEMANTIC] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Technical/Scientific - better for academic content
            try:
                models[EmbeddingStrategy.TECHNICAL] = SentenceTransformer('allenai-specter')
            except:
                models[EmbeddingStrategy.TECHNICAL] = models[EmbeddingStrategy.SEMANTIC]
            
            # Legal/Formal - better for structured documents
            try:
                models[EmbeddingStrategy.LEGAL] = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
            except:
                models[EmbeddingStrategy.LEGAL] = models[EmbeddingStrategy.SEMANTIC]
            
            # Conversational - better for Q&A
            try:
                models[EmbeddingStrategy.CONVERSATIONAL] = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            except:
                models[EmbeddingStrategy.CONVERSATIONAL] = models[EmbeddingStrategy.SEMANTIC]
            
            # Multilingual - for multi-language content
            try:
                models[EmbeddingStrategy.MULTILINGUAL] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                models[EmbeddingStrategy.MULTILINGUAL] = models[EmbeddingStrategy.SEMANTIC]
                
        except Exception as e:
            print(f"Error initializing embedding models: {e}")
            # Fallback to single model
            default_model = SentenceTransformer('all-MiniLM-L6-v2')
            for strategy in EmbeddingStrategy:
                models[strategy] = default_model
        
        return models
    
    def _initialize_collections(self) -> Dict[EmbeddingStrategy, Any]:
        """Initialize ChromaDB collections for different embedding strategies."""
        collections = {}
        
        for strategy in EmbeddingStrategy:
            collection_name = f"documents_{strategy.value}"
            collections[strategy] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # Higher for better accuracy
                    "hnsw:search_ef": 200,        # Higher for better search quality
                    "hnsw:M": 16                  # Good balance of speed/accuracy
                }
            )
        
        return collections
    
    def _load_search_configurations(self) -> Dict[str, SearchConfiguration]:
        """Load optimized search configurations."""
        # Default configurations for different query types
        configs = {
            "factual": SearchConfiguration(
                embedding_strategy=EmbeddingStrategy.SEMANTIC,
                relevance_weights=RelevanceWeights(
                    semantic_similarity=0.45,
                    keyword_overlap=0.30,
                    content_structure=0.15,
                    document_authority=0.05,
                    recency=0.02,
                    length_quality=0.03
                ),
                diversity_factor=0.2,
                use_query_expansion=False
            ),
            
            "analytical": SearchConfiguration(
                embedding_strategy=EmbeddingStrategy.TECHNICAL,
                relevance_weights=RelevanceWeights(
                    semantic_similarity=0.35,
                    keyword_overlap=0.15,
                    content_structure=0.25,
                    document_authority=0.15,
                    recency=0.05,
                    length_quality=0.05
                ),
                diversity_factor=0.4,
                use_query_expansion=True
            ),
            
            "conversational": SearchConfiguration(
                embedding_strategy=EmbeddingStrategy.CONVERSATIONAL,
                relevance_weights=RelevanceWeights(
                    semantic_similarity=0.50,
                    keyword_overlap=0.25,
                    content_structure=0.10,
                    document_authority=0.05,
                    recency=0.05,
                    length_quality=0.05
                ),
                diversity_factor=0.3,
                rerank_results=True
            )
        }
        
        return configs
    
    def determine_optimal_strategy(self, query: str) -> SearchConfiguration:
        """Determine optimal search strategy based on query analysis."""
        intent = self.query_analyzer.analyze_query(query)
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Technical/analytical queries
        technical_indicators = [
            'methodology', 'analysis', 'framework', 'approach', 'systematic',
            'correlation', 'relationship', 'compare', 'evaluate', 'assess'
        ]
        
        # Factual queries
        factual_indicators = [
            'what', 'who', 'when', 'where', 'define', 'definition', 'meaning',
            'list', 'name', 'identify', 'describe'
        ]
        
        # Conversational/explanatory queries
        conversational_indicators = [
            'how', 'why', 'explain', 'help', 'understand', 'can you',
            'please', 'could you', 'tell me'
        ]
        
        # Score each type
        technical_score = sum(1 for indicator in technical_indicators if indicator in query_lower)
        factual_score = sum(1 for indicator in factual_indicators if indicator in query_lower)
        conversational_score = sum(1 for indicator in conversational_indicators if indicator in query_lower)
        
        # Determine strategy
        if technical_score > max(factual_score, conversational_score):
            return self.search_configs["analytical"]
        elif factual_score > conversational_score:
            return self.search_configs["factual"]
        else:
            return self.search_configs["conversational"]
    
    def add_document(self, document: Document) -> bool:
        """Add document to all relevant collections with optimized embeddings."""
        try:
            # Prepare chunks for vectorization
            chunk_texts = [chunk.content for chunk in document.chunks]
            chunk_ids = [chunk.id for chunk in document.chunks]
            
            # Determine document type for optimal embedding strategy
            doc_type = self._analyze_document_type(document)
            primary_strategy = self._get_strategy_for_document_type(doc_type)
            
            # Generate embeddings using multiple strategies
            embedding_results = {}
            for strategy in [primary_strategy, EmbeddingStrategy.SEMANTIC]:  # Always include semantic as fallback
                if strategy not in embedding_results:
                    model = self.embedding_models[strategy]
                    embeddings = model.encode(chunk_texts).tolist()
                    embedding_results[strategy] = embeddings
            
            # Prepare enhanced metadata
            metadatas = []
            for chunk in document.chunks:
                content_metrics = self._analyze_chunk_content(chunk.content)
                
                metadatas.append({
                    "document_id": document.id,
                    "document_name": document.name,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "document_type": doc_type,
                    "word_count": content_metrics["word_count"],
                    "sentence_count": content_metrics["sentence_count"],
                    "avg_sentence_length": content_metrics["avg_sentence_length"],
                    "content_density": content_metrics["content_density"],
                    "has_technical_terms": content_metrics["has_technical_terms"],
                    "has_numbers": content_metrics["has_numbers"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Add to all relevant collections
            success = True
            for strategy, embeddings in embedding_results.items():
                try:
                    collection = self.collections[strategy]
                    collection.add(
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas,
                        ids=[f"{strategy.value}_{chunk_id}" for chunk_id in chunk_ids]
                    )
                except Exception as e:
                    print(f"Error adding to {strategy.value} collection: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            print(f"Error adding document to vector service: {e}")
            return False
    
    def _analyze_document_type(self, document: Document) -> str:
        """Analyze document to determine its type for optimal embedding strategy."""
        # Analyze document content to classify type
        full_text = " ".join([chunk.content for chunk in document.chunks[:5]])  # Sample first 5 chunks
        text_lower = full_text.lower()
        
        # Technical/Academic indicators
        if any(term in text_lower for term in [
            'methodology', 'results', 'discussion', 'conclusion',
            'abstract', 'introduction', 'literature', 'hypothesis'
        ]):
            return "academic"
        
        # Legal indicators
        elif any(term in text_lower for term in [
            'whereas', 'therefore', 'pursuant', 'agreement',
            'contract', 'terms', 'conditions', 'liability'
        ]):
            return "legal"
        
        # Technical/Manual indicators
        elif any(term in text_lower for term in [
            'configuration', 'installation', 'setup', 'procedure',
            'step', 'process', 'system', 'operation'
        ]):
            return "technical"
        
        else:
            return "general"
    
    def _get_strategy_for_document_type(self, doc_type: str) -> EmbeddingStrategy:
        """Map document type to optimal embedding strategy."""
        mapping = {
            "academic": EmbeddingStrategy.TECHNICAL,
            "legal": EmbeddingStrategy.LEGAL,
            "technical": EmbeddingStrategy.TECHNICAL,
            "general": EmbeddingStrategy.SEMANTIC
        }
        return mapping.get(doc_type, EmbeddingStrategy.SEMANTIC)
    
    def _analyze_chunk_content(self, content: str) -> Dict[str, Any]:
        """Analyze chunk content for metadata enhancement."""
        words = content.split()
        sentences = content.split('.')
        
        # Technical terms detection
        technical_terms = [
            'algorithm', 'methodology', 'framework', 'analysis', 'correlation',
            'statistical', 'significant', 'hypothesis', 'experiment', 'model'
        ]
        has_technical = any(term in content.lower() for term in technical_terms)
        
        # Numbers and data detection
        has_numbers = any(char.isdigit() for char in content)
        
        # Content density (non-stop words ratio)
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'were', 'will', 'with'}
        
        content_words = [w.lower() for w in words if w.lower() not in stop_words]
        content_density = len(content_words) / len(words) if words else 0
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "content_density": content_density,
            "has_technical_terms": has_technical,
            "has_numbers": has_numbers
        }
    
    def enhanced_search(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> List[SourceInfo]:
        """Enhanced search with optimized relevance scoring and result diversity."""
        
        # Determine optimal search configuration
        config = self.determine_optimal_strategy(query)
        max_results = max_results or config_search_results
        
        try:
            # Step 1: Query expansion if enabled
            search_queries = [query]
            if config.use_query_expansion:
                search_queries.extend(self._expand_query(query)[:2])  # Limit expansion
            
            # Step 2: Multi-strategy search
            all_results = []
            for i, search_query in enumerate(search_queries):
                strategy_results = self._search_with_strategy(
                    search_query, config.embedding_strategy, document_ids
                )
                
                # Weight results by query importance
                weight = 1.0 - (i * 0.2)  # Diminishing weight for expanded queries
                for result in strategy_results:
                    result['query_weight'] = weight
                    all_results.append(result)
            
            # Step 3: Advanced relevance scoring
            scored_results = self._calculate_enhanced_relevance_scores(
                all_results, query, config.relevance_weights
            )
            
            # Step 4: Result re-ranking if enabled
            if config.rerank_results:
                scored_results = self._rerank_results(scored_results, query)
            
            # Step 5: Diversity optimization
            final_results = self._optimize_result_diversity(
                scored_results, config.diversity_factor, config.max_results_per_doc
            )
            
            # Step 6: Filter by minimum relevance threshold
            filtered_results = [
                r for r in final_results 
                if r.relevance_score >= config.minimum_relevance_threshold
            ]
            
            return filtered_results[:max_results]
            
        except Exception as e:
            print(f"Enhanced search error: {e}")
            # Fallback to basic search
            return self._basic_search(query, document_ids, max_results)
    
    def _search_with_strategy(
        self, 
        query: str, 
        strategy: EmbeddingStrategy, 
        document_ids: Optional[List[str]]
    ) -> List[Dict]:
        """Search using specific embedding strategy."""
        try:
            # Get embedding model and collection for strategy
            model = self.embedding_models[strategy]
            collection = self.collections[strategy]
            
            # Generate query embedding
            query_embedding = model.encode([query]).tolist()[0]
            
            # Prepare where clause
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(config.search_results * 2, 50),  # Get more for better diversity
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to result format
            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    search_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'strategy': strategy.value,
                        'base_relevance': max(0, 1 - distance)
                    })
            
            return search_results
            
        except Exception as e:
            print(f"Strategy search error for {strategy.value}: {e}")
            return []
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        # Simple query expansion - could be enhanced with word embeddings or thesaurus
        expansions = []
        
        # Synonym mapping for common terms
        synonyms = {
            'method': ['approach', 'technique', 'procedure'],
            'result': ['finding', 'outcome', 'conclusion'],
            'analysis': ['examination', 'study', 'evaluation'],
            'impact': ['effect', 'influence', 'consequence'],
            'research': ['study', 'investigation', 'examination']
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonyms:
                # Create expanded queries by replacing word with synonym
                for synonym in synonyms[word]:
                    expanded_query = query.replace(word, synonym, 1)
                    if expanded_query != query:
                        expansions.append(expanded_query)
        
        return expansions[:3]  # Limit expansions
    
    def _calculate_enhanced_relevance_scores(
        self, 
        results: List[Dict], 
        original_query: str, 
        weights: RelevanceWeights
    ) -> List[SourceInfo]:
        """Calculate enhanced relevance scores using multiple factors."""
        scored_sources = []
        
        for result in results:
            content = result['content']
            metadata = result['metadata']
            base_relevance = result['base_relevance']
            query_weight = result.get('query_weight', 1.0)
            
            # Factor 1: Semantic similarity (base relevance from vector search)
            semantic_score = base_relevance
            
            # Factor 2: Keyword overlap
            keyword_score = self._calculate_keyword_overlap(original_query, content)
            
            # Factor 3: Content structure quality
            structure_score = self._assess_content_structure(content, metadata)
            
            # Factor 4: Document authority (based on metadata signals)
            authority_score = self._assess_document_authority(metadata)
            
            # Factor 5: Recency (if timestamp available)
            recency_score = self._assess_content_recency(metadata)
            
            # Factor 6: Length quality (optimal length scoring)
            length_score = self._assess_content_length_quality(content)
            
            # Calculate weighted final score
            final_score = (
                semantic_score * weights.semantic_similarity +
                keyword_score * weights.keyword_overlap +
                structure_score * weights.content_structure +
                authority_score * weights.document_authority +
                recency_score * weights.recency +
                length_score * weights.length_quality
            ) * query_weight
            
            source = SourceInfo(
                document_id=metadata['document_id'],
                document_name=metadata['document_name'],
                chunk_content=content,
                relevance_score=final_score
            )
            
            scored_sources.append(source)
        
        return scored_sources
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return min(overlap / len(query_words), 1.0)
    
    def _assess_content_structure(self, content: str, metadata: Dict) -> float:
        """Assess content structure quality."""
        # Use pre-calculated metadata if available
        if 'content_density' in metadata:
            return metadata['content_density']
        
        # Fallback calculation
        words = content.split()
        sentences = content.split('.')
        
        if not words:
            return 0.0
        
        # Factors for good structure
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Optimal sentence length is around 15-25 words
        length_score = 1.0 - abs(avg_sentence_length - 20) / 30
        length_score = max(0.0, min(1.0, length_score))
        
        return length_score
    
    def _assess_document_authority(self, metadata: Dict) -> float:
        """Assess document authority based on metadata signals."""
        score = 0.5  # Neutral baseline
        
        # Technical content tends to be more authoritative
        if metadata.get('has_technical_terms', False):
            score += 0.2
        
        # Structured content (with numbers, data) tends to be more authoritative
        if metadata.get('has_numbers', False):
            score += 0.1
        
        # Academic documents tend to be more authoritative
        if metadata.get('document_type') == 'academic':
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_content_recency(self, metadata: Dict) -> float:
        """Assess content recency (newer is generally better)."""
        if 'timestamp' not in metadata:
            return 0.5  # Neutral score for unknown age
        
        try:
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            age_days = (datetime.now() - timestamp).days
            
            # Decay factor: newer content scores higher
            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.8
            elif age_days < 90:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5
    
    def _assess_content_length_quality(self, content: str) -> float:
        """Assess content length quality (optimal length scoring)."""
        word_count = len(content.split())
        
        # Optimal length is typically 50-300 words for search results
        if 50 <= word_count <= 300:
            return 1.0
        elif 20 <= word_count < 50 or 300 < word_count <= 500:
            return 0.8
        elif 10 <= word_count < 20 or 500 < word_count <= 800:
            return 0.6
        else:
            return 0.4
    
    def _rerank_results(self, results: List[SourceInfo], query: str) -> List[SourceInfo]:
        """Advanced re-ranking of results using cross-attention techniques."""
        # For now, implement a simple re-ranking based on content quality
        # In a full implementation, this could use a more sophisticated reranking model
        
        def rerank_score(source: SourceInfo) -> float:
            content = source.chunk_content
            
            # Boost score for comprehensive content
            completeness_boost = 0
            if len(content.split()) > 100:
                completeness_boost += 0.05
            
            # Boost score for query-specific content
            query_specific_boost = 0
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            # Exact phrase matches get higher boost
            if query.lower() in content.lower():
                query_specific_boost += 0.1
            
            # Multiple query word matches
            matches = len(query_words.intersection(content_words))
            if matches > len(query_words) * 0.5:
                query_specific_boost += 0.05
            
            return source.relevance_score + completeness_boost + query_specific_boost
        
        # Re-sort by enhanced scores
        reranked = sorted(results, key=rerank_score, reverse=True)
        
        # Update relevance scores with reranking
        for i, source in enumerate(reranked):
            # Slight boost for top results, slight penalty for lower results
            position_adjustment = (len(reranked) - i) / len(reranked) * 0.05
            source.relevance_score += position_adjustment
        
        return reranked
    
    def _optimize_result_diversity(
        self, 
        results: List[SourceInfo], 
        diversity_factor: float,
        max_per_doc: int
    ) -> List[SourceInfo]:
        """Optimize result diversity while maintaining relevance."""
        if not results:
            return results
        
        # Group results by document
        doc_groups = defaultdict(list)
        for source in results:
            doc_groups[source.document_id].append(source)
        
        # Sort each group by relevance
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Implement diversity-aware selection
        selected_results = []
        doc_counts = defaultdict(int)
        
        # First, select the highest scoring result from each document
        for doc_id, sources in doc_groups.items():
            if sources:
                selected_results.append(sources[0])
                doc_counts[doc_id] = 1
        
        # Then fill remaining slots with a balance of relevance and diversity
        remaining_sources = []
        for doc_id, sources in doc_groups.items():
            remaining_sources.extend(sources[doc_counts[doc_id]:])
        
        # Sort remaining by relevance
        remaining_sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Select additional results with diversity constraint
        for source in remaining_sources:
            if len(selected_results) >= config.search_results:
                break
            
            if doc_counts[source.document_id] < max_per_doc:
                selected_results.append(source)
                doc_counts[source.document_id] += 1
        
        # Final sort by relevance
        selected_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return selected_results
    
    def _basic_search(
        self, 
        query: str, 
        document_ids: Optional[List[str]],
        max_results: int
    ) -> List[SourceInfo]:
        """Basic fallback search using semantic strategy."""
        try:
            return self._search_with_strategy(
                query, EmbeddingStrategy.SEMANTIC, document_ids
            )[:max_results]
        except:
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from all collections."""
        success = True
        
        for strategy, collection in self.collections.items():
            try:
                collection.delete(where={"document_id": document_id})
            except Exception as e:
                print(f"Error deleting from {strategy.value} collection: {e}")
                success = False
        
        return success
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the enhanced vector service."""
        metrics = {}
        
        for strategy, collection in self.collections.items():
            try:
                count = collection.count()
                metrics[strategy.value] = {
                    "document_count": count,
                    "model_name": type(self.embedding_models[strategy]).__name__
                }
            except Exception as e:
                metrics[strategy.value] = {"error": str(e)}
        
        return {
            "collections": metrics,
            "total_strategies": len(self.collections),
            "default_strategy": self.default_model.__class__.__name__
        }