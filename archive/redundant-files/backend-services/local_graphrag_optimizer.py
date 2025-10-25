"""Local GraphRAG Optimization Service for efficient local model processing."""
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from config import config
from models import Document, EnhancedSourceInfo, EntityInfo, RelationshipInfo

logger = logging.getLogger(__name__)


@dataclass
class LocalGraphConfig:
    """Configuration for local GraphRAG optimization."""
    max_entities_per_chunk: int = 10
    max_relationships_per_entity: int = 5
    entity_similarity_threshold: float = 0.8
    relationship_confidence_threshold: float = 0.7
    use_lightweight_models: bool = True
    enable_caching: bool = True
    batch_size: int = 32
    max_community_size: int = 50


@dataclass
class EntityExtractionResult:
    """Result of entity extraction optimization."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    processing_time: float
    confidence_scores: Dict[str, float]
    optimization_notes: List[str]


@dataclass
class GraphOptimizationMetrics:
    """Metrics for graph processing optimization."""
    entity_extraction_time: float
    relationship_inference_time: float
    community_detection_time: float
    total_entities: int
    total_relationships: int
    optimization_ratio: float  # Reduction in processing time vs naive approach


class LocalGraphRAGOptimizer:
    """Optimized GraphRAG processing for local models with resource constraints."""
    
    def __init__(self):
        self.optimization_dir = config.project_root / "data" / "graphrag_optimization"
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = LocalGraphConfig()
        
        # Cache directories
        self.entity_cache_dir = self.optimization_dir / "entity_cache"
        self.relationship_cache_dir = self.optimization_dir / "relationship_cache"
        self.community_cache_dir = self.optimization_dir / "community_cache"
        
        for cache_dir in [self.entity_cache_dir, self.relationship_cache_dir, self.community_cache_dir]:
            cache_dir.mkdir(exist_ok=True)
        
        # Initialize lightweight models for local processing
        self.entity_model = self._initialize_entity_model()
        self.relationship_model = self._initialize_relationship_model()
        
        # Graph storage
        self.knowledge_graph = nx.Graph()
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.performance_metrics: List[GraphOptimizationMetrics] = []
        
        # Pre-computed patterns for efficient extraction
        self.entity_patterns = self._load_entity_patterns()
        self.relationship_patterns = self._load_relationship_patterns()
    
    def _initialize_entity_model(self) -> SentenceTransformer:
        """Initialize lightweight model for entity extraction."""
        try:
            if self.config.use_lightweight_models:
                # Use smaller, faster model for entity recognition
                return SentenceTransformer('all-MiniLM-L6-v2')
            else:
                # Use more accurate but slower model
                return SentenceTransformer('all-mpnet-base-v2')
        except Exception as e:
            logger.warning(f"Failed to load entity model: {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _initialize_relationship_model(self) -> SentenceTransformer:
        """Initialize lightweight model for relationship detection."""
        try:
            # Use same model as entity extraction for memory efficiency
            return self.entity_model
        except Exception as e:
            logger.warning(f"Failed to load relationship model: {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load pre-defined entity patterns for fast recognition."""
        return {
            "PERSON": [
                r"\\b[A-Z][a-z]+ [A-Z][a-z]+\\b",  # FirstName LastName
                r"\\bDr\\. [A-Z][a-z]+ [A-Z][a-z]+",  # Dr. FirstName LastName
                r"\\bProf\\. [A-Z][a-z]+ [A-Z][a-z]+",  # Prof. FirstName LastName
            ],
            "ORGANIZATION": [
                r"\\b[A-Z][a-z]+ (University|College|Institute|Corporation|Company|Inc\\.|Ltd\\.)\\b",
                r"\\b(Department|School|Faculty) of [A-Z][a-z]+\\b",
            ],
            "LOCATION": [
                r"\\b[A-Z][a-z]+ (City|State|Country|County)\\b",
                r"\\b(United States|Canada|Europe|Asia|Africa)\\b",
            ],
            "CONCEPT": [
                r"\\b[a-z]+ (theory|method|approach|framework|model)\\b",
                r"\\b(artificial intelligence|machine learning|deep learning)\\b",
            ],
            "TECHNOLOGY": [
                r"\\b[A-Z]+(\\.[a-z]+)*\\b",  # Acronyms
                r"\\b(software|hardware|algorithm|system|platform)\\b",
            ]
        }
    
    def _load_relationship_patterns(self) -> Dict[str, List[str]]:
        """Load pre-defined relationship patterns."""
        return {
            "COLLABORATES_WITH": [
                "works with", "collaborates with", "partners with", "co-authored with"
            ],
            "EMPLOYS": [
                "employs", "hires", "works at", "employed by"
            ],
            "DEVELOPS": [
                "develops", "creates", "builds", "designed", "invented"
            ],
            "USES": [
                "uses", "utilizes", "applies", "implements", "adopts"
            ],
            "STUDIES": [
                "studies", "researches", "investigates", "examines", "analyzes"
            ],
            "LOCATED_IN": [
                "located in", "based in", "situated in", "found in"
            ]
        }
    
    def optimize_entity_extraction(
        self, 
        document: Document,
        use_cache: bool = True
    ) -> EntityExtractionResult:
        """Optimized entity extraction with caching and batching."""
        start_time = time.time()
        optimization_notes = []
        
        # Check cache first
        cache_key = f"{document.id}_entities"
        cache_file = self.entity_cache_dir / f"{cache_key}.json"
        
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_result = json.load(f)
                    optimization_notes.append("Used cached entity extraction")
                    return EntityExtractionResult(
                        entities=cached_result["entities"],
                        relationships=cached_result["relationships"],
                        processing_time=time.time() - start_time,
                        confidence_scores=cached_result.get("confidence_scores", {}),
                        optimization_notes=optimization_notes
                    )
            except Exception as e:
                logger.warning(f"Failed to load entity cache: {e}")
        
        # Extract entities using optimized approach
        all_entities = []
        all_relationships = []
        confidence_scores = {}
        
        # Process chunks in batches for efficiency
        chunk_batches = [
            document.chunks[i:i + self.config.batch_size] 
            for i in range(0, len(document.chunks), self.config.batch_size)
        ]
        
        for batch_idx, chunk_batch in enumerate(chunk_batches):
            batch_entities, batch_relationships, batch_confidence = self._process_chunk_batch(
                chunk_batch, document.id
            )
            
            all_entities.extend(batch_entities)
            all_relationships.extend(batch_relationships)
            confidence_scores.update(batch_confidence)
            
            optimization_notes.append(f"Processed batch {batch_idx + 1}/{len(chunk_batches)}")
        
        # Deduplicate and merge similar entities
        deduplicated_entities = self._deduplicate_entities(all_entities)
        merged_relationships = self._merge_relationships(all_relationships)
        
        optimization_notes.append(f"Reduced {len(all_entities)} entities to {len(deduplicated_entities)}")
        optimization_notes.append(f"Merged to {len(merged_relationships)} relationships")
        
        # Cache results
        if use_cache:
            try:
                cache_data = {
                    "entities": deduplicated_entities,
                    "relationships": merged_relationships,
                    "confidence_scores": confidence_scores,
                    "timestamp": datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                optimization_notes.append("Cached results for future use")
            except Exception as e:
                logger.warning(f"Failed to cache entities: {e}")
        
        processing_time = time.time() - start_time
        
        return EntityExtractionResult(
            entities=deduplicated_entities,
            relationships=merged_relationships,
            processing_time=processing_time,
            confidence_scores=confidence_scores,
            optimization_notes=optimization_notes
        )
    
    def _process_chunk_batch(
        self, 
        chunk_batch: List[Any], 
        document_id: str
    ) -> Tuple[List[Dict], List[Dict], Dict[str, float]]:
        """Process a batch of chunks efficiently."""
        entities = []
        relationships = []
        confidence_scores = {}
        
        # Extract text from chunks
        chunk_texts = [chunk.content for chunk in chunk_batch]
        
        # Batch entity extraction
        batch_entities = self._extract_entities_batch(chunk_texts, document_id)
        entities.extend(batch_entities)
        
        # Extract relationships between entities in the same chunks
        for i, chunk in enumerate(chunk_batch):
            chunk_entities = [e for e in batch_entities if e.get('chunk_index') == chunk.chunk_index]
            chunk_relationships = self._extract_relationships_from_chunk(
                chunk.content, chunk_entities, document_id
            )
            relationships.extend(chunk_relationships)
            
            # Calculate confidence scores
            confidence_scores[f"chunk_{chunk.chunk_index}"] = self._calculate_chunk_confidence(
                chunk.content, chunk_entities, chunk_relationships
            )
        
        return entities, relationships, confidence_scores
    
    def _extract_entities_batch(
        self, 
        chunk_texts: List[str], 
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract entities from a batch of texts efficiently."""
        entities = []
        
        # Use pattern-based extraction first (fast)
        for i, text in enumerate(chunk_texts):
            pattern_entities = self._extract_entities_by_patterns(text, document_id, i)
            entities.extend(pattern_entities)
        
        # Use embedding-based extraction for missed entities (slower but more accurate)
        if len(entities) < self.config.max_entities_per_chunk * len(chunk_texts) * 0.5:
            # If pattern-based extraction didn't find enough entities, use embeddings
            embedding_entities = self._extract_entities_by_embeddings(chunk_texts, document_id)
            entities.extend(embedding_entities)
        
        return entities
    
    def _extract_entities_by_patterns(
        self, 
        text: str, 
        document_id: str, 
        chunk_index: int
    ) -> List[Dict[str, Any]]:
        """Fast entity extraction using pre-defined patterns."""
        import re
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:  # Filter very short matches
                        entity = {
                            "id": f"{document_id}_{entity_type}_{len(entities)}",
                            "name": entity_text,
                            "type": entity_type,
                            "description": f"{entity_type} entity found in document",
                            "source_document_id": document_id,
                            "chunk_index": chunk_index,
                            "extraction_method": "pattern",
                            "confidence": 0.8  # Pattern-based extraction has good confidence
                        }
                        entities.append(entity)
        
        return entities[:self.config.max_entities_per_chunk]
    
    def _extract_entities_by_embeddings(
        self, 
        chunk_texts: List[str], 
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Embedding-based entity extraction for missed entities."""
        entities = []
        
        # This is a simplified version - in practice would use NER models
        # For now, extract noun phrases as potential entities
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            return entities  # Skip if NLTK data not available
        
        for chunk_idx, text in enumerate(chunk_texts):
            try:
                # Simple noun phrase extraction
                words = text.split()
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if self._is_potential_entity(phrase):
                        entity = {
                            "id": f"{document_id}_embedding_{len(entities)}",
                            "name": phrase,
                            "type": "CONCEPT",
                            "description": "Entity extracted using embeddings",
                            "source_document_id": document_id,
                            "chunk_index": chunk_idx,
                            "extraction_method": "embedding",
                            "confidence": 0.6
                        }
                        entities.append(entity)
            except Exception as e:
                logger.warning(f"Embedding entity extraction failed: {e}")
                continue
        
        return entities
    
    def _is_potential_entity(self, phrase: str) -> bool:
        """Check if a phrase is likely to be an entity."""
        phrase = phrase.lower()
        
        # Skip common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = phrase.split()
        if all(word in stop_words for word in words):
            return False
        
        # Look for entity indicators
        entity_indicators = [
            'theory', 'method', 'approach', 'model', 'system', 'algorithm',
            'framework', 'technique', 'process', 'concept', 'principle'
        ]
        
        return any(indicator in phrase for indicator in entity_indicators)
    
    def _extract_relationships_from_chunk(
        self, 
        chunk_text: str, 
        entities: List[Dict], 
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities in the same chunk."""
        relationships = []
        
        # Check for relationships between all entity pairs
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationship = self._detect_relationship(
                    chunk_text, entity1, entity2, document_id
                )
                if relationship:
                    relationships.append(relationship)
                
                # Limit relationships per entity
                if len([r for r in relationships 
                       if r['source_entity'] == entity1['name'] or r['target_entity'] == entity1['name']]) >= self.config.max_relationships_per_entity:
                    break
        
        return relationships
    
    def _detect_relationship(
        self, 
        text: str, 
        entity1: Dict, 
        entity2: Dict, 
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Detect relationship between two entities in text."""
        entity1_name = entity1['name'].lower()
        entity2_name = entity2['name'].lower()
        text_lower = text.lower()
        
        # Find positions of entities in text
        pos1 = text_lower.find(entity1_name)
        pos2 = text_lower.find(entity2_name)
        
        if pos1 == -1 or pos2 == -1:
            return None
        
        # Extract text between entities
        start_pos = min(pos1, pos2)
        end_pos = max(pos1 + len(entity1_name), pos2 + len(entity2_name))
        between_text = text_lower[start_pos:end_pos]
        
        # Check for relationship patterns
        for relationship_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                if pattern in between_text:
                    return {
                        "id": f"{document_id}_{entity1['name']}_{entity2['name']}",
                        "source_entity": entity1['name'],
                        "target_entity": entity2['name'],
                        "relationship_type": relationship_type,
                        "confidence": 0.7,
                        "source_document_id": document_id,
                        "evidence_text": between_text[:100],  # Limit evidence text
                        "extraction_method": "pattern"
                    }
        
        # If no specific pattern found but entities are close, infer generic relationship
        distance = abs(pos1 - pos2)
        if distance < 200:  # Entities are close in text
            return {
                "id": f"{document_id}_{entity1['name']}_{entity2['name']}",
                "source_entity": entity1['name'],
                "target_entity": entity2['name'],
                "relationship_type": "RELATED_TO",
                "confidence": 0.5,
                "source_document_id": document_id,
                "evidence_text": between_text[:100],
                "extraction_method": "proximity"
            }
        
        return None
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate and similar entities."""
        if not entities:
            return entities
        
        # Group entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['type']].append(entity)
        
        deduplicated = []
        
        for entity_type, type_entities in entities_by_type.items():
            # Calculate embeddings for entity names
            entity_names = [e['name'] for e in type_entities]
            
            try:
                embeddings = self.entity_model.encode(entity_names)
                
                # Find similar entities using cosine similarity
                similarity_matrix = np.dot(embeddings, embeddings.T)
                norms = np.linalg.norm(embeddings, axis=1)
                similarity_matrix = similarity_matrix / np.outer(norms, norms)
                
                # Merge similar entities
                merged_indices = set()
                for i, entity in enumerate(type_entities):
                    if i in merged_indices:
                        continue
                    
                    # Find similar entities
                    similar_indices = np.where(
                        similarity_matrix[i] > self.config.entity_similarity_threshold
                    )[0]
                    
                    # Merge similar entities
                    merged_entity = self._merge_similar_entities(
                        [type_entities[idx] for idx in similar_indices]
                    )
                    deduplicated.append(merged_entity)
                    merged_indices.update(similar_indices)
                
            except Exception as e:
                logger.warning(f"Entity deduplication failed: {e}")
                # Fallback to simple name-based deduplication
                seen_names = set()
                for entity in type_entities:
                    name_lower = entity['name'].lower()
                    if name_lower not in seen_names:
                        deduplicated.append(entity)
                        seen_names.add(name_lower)
        
        return deduplicated
    
    def _merge_similar_entities(self, similar_entities: List[Dict]) -> Dict:
        """Merge similar entities into a single representative entity."""
        if len(similar_entities) == 1:
            return similar_entities[0]
        
        # Use the entity with highest confidence as base
        base_entity = max(similar_entities, key=lambda e: e.get('confidence', 0))
        
        # Merge information
        merged_entity = base_entity.copy()
        
        # Combine descriptions
        descriptions = [e.get('description', '') for e in similar_entities if e.get('description')]
        if descriptions:
            merged_entity['description'] = '; '.join(descriptions)
        
        # Average confidence
        confidences = [e.get('confidence', 0) for e in similar_entities]
        merged_entity['confidence'] = sum(confidences) / len(confidences)
        
        # Track merged entities
        merged_entity['merged_from'] = [e['id'] for e in similar_entities]
        merged_entity['merge_count'] = len(similar_entities)
        
        return merged_entity
    
    def _merge_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Merge duplicate relationships."""
        if not relationships:
            return relationships
        
        # Group relationships by entity pair and type
        relationship_groups = defaultdict(list)
        
        for rel in relationships:
            # Create a key for grouping (ensure consistent ordering)
            entity1, entity2 = sorted([rel['source_entity'], rel['target_entity']])
            key = f"{entity1}_{entity2}_{rel['relationship_type']}"
            relationship_groups[key].append(rel)
        
        merged_relationships = []
        for group in relationship_groups.values():
            if len(group) == 1:
                merged_relationships.append(group[0])
            else:
                # Merge multiple relationships of same type between same entities
                merged_rel = group[0].copy()
                
                # Average confidence
                confidences = [r.get('confidence', 0) for r in group]
                merged_rel['confidence'] = sum(confidences) / len(confidences)
                
                # Combine evidence text
                evidence_texts = [r.get('evidence_text', '') for r in group if r.get('evidence_text')]
                if evidence_texts:
                    merged_rel['evidence_text'] = '; '.join(evidence_texts[:3])  # Limit length
                
                merged_rel['merge_count'] = len(group)
                merged_relationships.append(merged_rel)
        
        return merged_relationships
    
    def _calculate_chunk_confidence(
        self, 
        chunk_text: str, 
        entities: List[Dict], 
        relationships: List[Dict]
    ) -> float:
        """Calculate confidence score for chunk processing."""
        if not entities:
            return 0.0
        
        # Base confidence on entity extraction quality
        entity_confidences = [e.get('confidence', 0) for e in entities]
        entity_score = sum(entity_confidences) / len(entity_confidences)
        
        # Adjust based on relationship extraction
        if relationships:
            rel_confidences = [r.get('confidence', 0) for r in relationships]
            rel_score = sum(rel_confidences) / len(rel_confidences)
            combined_score = (entity_score * 0.7) + (rel_score * 0.3)
        else:
            combined_score = entity_score * 0.8  # Penalty for no relationships
        
        # Adjust based on text quality indicators
        text_quality = self._assess_text_quality(chunk_text)
        final_score = combined_score * text_quality
        
        return min(final_score, 1.0)
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality for entity extraction."""
        if not text:
            return 0.0
        
        # Basic quality indicators
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Prefer medium-length texts
        if 50 <= word_count <= 300:
            length_score = 1.0
        elif 20 <= word_count < 50 or 300 < word_count <= 500:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Check for structured content indicators
        structure_indicators = [':', ';', '-', '(', ')', '[', ']', '"']
        structure_score = min(sum(text.count(ind) for ind in structure_indicators) / 10, 1.0)
        
        return (length_score * 0.7) + (structure_score * 0.3)
    
    def optimize_local_search(
        self, 
        query: str, 
        entities: List[Dict], 
        relationships: List[Dict],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Optimized local search for resource-constrained environments."""
        start_time = time.time()
        
        # Build lightweight knowledge graph
        graph = self._build_lightweight_graph(entities, relationships)
        
        # Find relevant entities for the query
        relevant_entities = self._find_relevant_entities(query, entities)
        
        # Expand search using graph traversal (limited hops)
        expanded_entities = self._expand_entities_with_graph(
            relevant_entities, graph, max_hops
        )
        
        # Generate context from expanded entities
        context = self._generate_local_context(expanded_entities, relationships)
        
        processing_time = time.time() - start_time
        
        return {
            "relevant_entities": relevant_entities,
            "expanded_entities": expanded_entities,
            "context": context,
            "processing_time": processing_time,
            "optimization_notes": [
                f"Built graph with {len(entities)} entities and {len(relationships)} relationships",
                f"Found {len(relevant_entities)} relevant entities",
                f"Expanded to {len(expanded_entities)} entities in {max_hops} hops"
            ]
        }
    
    def _build_lightweight_graph(
        self, 
        entities: List[Dict], 
        relationships: List[Dict]
    ) -> nx.Graph:
        """Build a lightweight networkx graph for local processing."""
        graph = nx.Graph()
        
        # Add entity nodes
        for entity in entities:
            graph.add_node(
                entity['name'], 
                type=entity['type'],
                confidence=entity.get('confidence', 0.5),
                id=entity['id']
            )
        
        # Add relationship edges
        for rel in relationships:
            if rel['source_entity'] in graph.nodes and rel['target_entity'] in graph.nodes:
                graph.add_edge(
                    rel['source_entity'],
                    rel['target_entity'],
                    relationship_type=rel['relationship_type'],
                    confidence=rel.get('confidence', 0.5),
                    id=rel['id']
                )
        
        return graph
    
    def _find_relevant_entities(self, query: str, entities: List[Dict]) -> List[Dict]:
        """Find entities most relevant to the query."""
        if not entities:
            return []
        
        query_lower = query.lower()
        relevant_entities = []
        
        for entity in entities:
            # Simple relevance scoring
            relevance_score = 0.0
            entity_name = entity['name'].lower()
            
            # Direct name match
            if entity_name in query_lower:
                relevance_score += 1.0
            
            # Word overlap
            entity_words = set(entity_name.split())
            query_words = set(query_lower.split())
            overlap = len(entity_words.intersection(query_words))
            if entity_words:
                relevance_score += overlap / len(entity_words) * 0.5
            
            # Type relevance (some types more likely to be queried)
            type_boost = {
                'PERSON': 0.3,
                'ORGANIZATION': 0.2,
                'CONCEPT': 0.4,
                'TECHNOLOGY': 0.3,
                'LOCATION': 0.1
            }.get(entity['type'], 0.1)
            relevance_score += type_boost
            
            if relevance_score > 0.3:  # Threshold for relevance
                entity_with_score = entity.copy()
                entity_with_score['relevance_score'] = relevance_score
                relevant_entities.append(entity_with_score)
        
        # Sort by relevance and return top entities
        relevant_entities.sort(key=lambda e: e['relevance_score'], reverse=True)
        return relevant_entities[:20]  # Limit for performance
    
    def _expand_entities_with_graph(
        self, 
        seed_entities: List[Dict], 
        graph: nx.Graph, 
        max_hops: int
    ) -> List[Dict]:
        """Expand entity set using graph traversal."""
        expanded_entities = {e['name']: e for e in seed_entities}
        
        # Breadth-first expansion
        for hop in range(max_hops):
            current_entities = list(expanded_entities.keys())
            for entity_name in current_entities:
                if entity_name in graph.nodes:
                    # Get neighbors
                    neighbors = list(graph.neighbors(entity_name))
                    for neighbor in neighbors:
                        if neighbor not in expanded_entities:
                            # Create entity dict for neighbor
                            neighbor_data = graph.nodes[neighbor]
                            expanded_entities[neighbor] = {
                                'name': neighbor,
                                'type': neighbor_data.get('type', 'UNKNOWN'),
                                'confidence': neighbor_data.get('confidence', 0.5),
                                'id': neighbor_data.get('id', f"expanded_{neighbor}"),
                                'expansion_hop': hop + 1
                            }
        
        return list(expanded_entities.values())
    
    def _generate_local_context(
        self, 
        entities: List[Dict], 
        relationships: List[Dict]
    ) -> str:
        """Generate context string for local GraphRAG search."""
        context_parts = []
        
        # Add entity information
        entity_names = [e['name'] for e in entities]
        context_parts.append(f"Relevant entities: {', '.join(entity_names[:10])}")
        
        # Add relationship information
        relevant_relationships = [
            r for r in relationships 
            if r['source_entity'] in entity_names and r['target_entity'] in entity_names
        ]
        
        if relevant_relationships:
            context_parts.append("Key relationships:")
            for rel in relevant_relationships[:10]:  # Limit for context size
                context_parts.append(
                    f"- {rel['source_entity']} {rel['relationship_type']} {rel['target_entity']}"
                )
        
        return "\n".join(context_parts)
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        if not self.performance_metrics:
            return {"message": "No optimization metrics available"}
        
        recent_metrics = self.performance_metrics[-10:]  # Last 10 operations
        
        return {
            "total_operations": len(self.performance_metrics),
            "average_entity_extraction_time": sum(m.entity_extraction_time for m in recent_metrics) / len(recent_metrics),
            "average_relationship_time": sum(m.relationship_inference_time for m in recent_metrics) / len(recent_metrics),
            "average_total_entities": sum(m.total_entities for m in recent_metrics) / len(recent_metrics),
            "average_total_relationships": sum(m.total_relationships for m in recent_metrics) / len(recent_metrics),
            "average_optimization_ratio": sum(m.optimization_ratio for m in recent_metrics) / len(recent_metrics),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for entity extraction."""
        # Simple approximation based on cache files
        total_cache_files = len(list(self.entity_cache_dir.glob("*.json")))
        if total_cache_files == 0:
            return 0.0
        
        # Estimate hit rate based on file access times
        recent_files = [
            f for f in self.entity_cache_dir.glob("*.json")
            if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < 1
        ]
        
        return len(recent_files) / max(total_cache_files, 1)
    
    def clear_optimization_cache(self) -> Dict[str, Any]:
        """Clear optimization caches to free up disk space."""
        cleared_files = 0
        
        for cache_dir in [self.entity_cache_dir, self.relationship_cache_dir, self.community_cache_dir]:
            for cache_file in cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    cleared_files += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        return {
            "cleared_files": cleared_files,
            "message": f"Cleared {cleared_files} cache files"
        }