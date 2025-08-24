"""Vector database service for Able."""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from collections import defaultdict
import math
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import config
from models import Document, SourceInfo
from services.query_analyzer import QueryAnalyzer

def mmr_select(query_vec, cand_vecs, lambda_mult: float, top_k: int) -> List[int]:
    selected = []
    remaining = list(range(len(cand_vecs)))
    sims_to_query = cosine_similarity(cand_vecs, query_vec.reshape(1, -1)).reshape(-1)
    
    while remaining and len(selected) < top_k:
        mmr_scores = []
        for i in remaining:
            if not selected:
                div = 0.0
            else:
                div = max(cosine_similarity(cand_vecs[i].reshape(1, -1), cand_vecs[selected]).reshape(-1))
            mmr = lambda_mult * sims_to_query[i] - (1 - lambda_mult) * div
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected

def cap_by_doc(cands: List[Dict[str, any]], max_per_doc: int) -> List[Dict[str, any]]:
    seen = {}
    out = []
    for c in cands:
        did = c["meta"].get("doc_id", "_")
        seen[did] = seen.get(did, 0)
        if seen[did] < max_per_doc:
            out.append(c)
            seen[did] += 1
    return out

class VectorService:
    """Enhanced vector database with document diversity."""
    
    def __init__(self):
        self.db_path = config.vectordb_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, document: Document) -> bool:
        """Add document to vector database."""
        try:
            # Prepare chunks for vectorization
            chunk_texts = [chunk.content for chunk in document.chunks]
            chunk_ids = [chunk.id for chunk in document.chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Prepare metadata
            metadatas = []
            for chunk in document.chunks:
                metadatas.append({
                    "document_id": document.id,
                    "document_name": document.name,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                })
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            return True
        except Exception:
            return False
    
    def strategic_search(self, query: str, document_ids: Optional[List[str]] = None) -> List[SourceInfo]:
        """Agentic search with query analysis and strategic retrieval."""
        try:
            # Step 1: Analyze the query to understand intent
            intent = self.query_analyzer.analyze_query(query)
            
            # Step 2: Execute multiple search strategies
            all_results = []
            
            for i, search_query in enumerate(intent.search_strategies[:4]):  # Limit to 4 strategies
                try:
                    query_embedding = self.embedding_model.encode([search_query]).tolist()[0]
                    
                    where_clause = None
                    if document_ids:
                        where_clause = {"document_id": {"$in": document_ids}}
                    
                    # Search with this strategy
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=config.search_results,
                        where=where_clause,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Weight results based on strategy importance
                    strategy_weight = 1.0 - (i * 0.1)  # First strategy gets highest weight
                    
                    if results['documents'] and results['documents'][0]:
                        for doc, metadata, distance in zip(
                            results['documents'][0],
                            results['metadatas'][0], 
                            results['distances'][0]
                        ):
                            relevance_score = max(0, (1 - distance) * strategy_weight)
                            
                            # Content type scoring based on query intent
                            content_score = self._score_content_relevance(doc, intent)
                            final_score = relevance_score * content_score
                            
                            all_results.append({
                                'source': SourceInfo(
                                    document_id=metadata['document_id'],
                                    document_name=metadata['document_name'],
                                    chunk_content=doc,
                                    relevance_score=final_score
                                ),
                                'strategy': search_query,
                                'content_type': self._identify_content_type(doc)
                            })
                            
                except Exception:
                    continue
            
            # Step 3: Rank and diversify results
            final_sources = self._rank_and_diversify_strategic_results(all_results, intent)
            
            return final_sources[:config.search_results]
            
        except Exception:
            # Fallback to simple search
            return self._simple_search(query, document_ids)
    
    def _score_content_relevance(self, content: str, intent) -> float:
        """Score content based on query intent and content type."""
        content_lower = content.lower()
        score = 1.0
        
        # Boost score for preferred content types
        for preference in intent.content_preferences:
            preference_indicators = {
                'introduction': ['introduction', 'background', 'overview', 'abstract'],
                'methodology': ['method', 'approach', 'procedure', 'design', 'technique'],
                'results': ['results', 'findings', 'data', 'analysis', 'outcome'],
                'conclusion': ['conclusion', 'summary', 'implications', 'discussion'],
                'theoretical': ['theory', 'model', 'concept', 'framework']
            }
            
            indicators = preference_indicators.get(preference, [])
            for indicator in indicators:
                if indicator in content_lower:
                    score *= 1.3  # Boost preferred content
                    break
        
        # Penalize reference sections and bibliographies
        reference_indicators = [
            'references', 'bibliography', 'cited', 'et al.', 'doi:', 'arxiv:',
            'journal of', 'proceedings of', 'conference on'
        ]
        
        reference_count = sum(1 for indicator in reference_indicators if indicator in content_lower)
        if reference_count > 2:
            score *= 0.6  # Penalize heavily reference-heavy content
        elif reference_count > 0:
            score *= 0.8  # Light penalty for some references
        
        # Boost substantive content
        if len(content.split()) > 100:  # Longer chunks often have more substance
            score *= 1.1
        
        return min(score, 2.0)  # Cap the boost
    
    def _identify_content_type(self, content: str) -> str:
        """Identify the type of content in a chunk."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['method', 'approach', 'procedure']):
            return 'methodology'
        elif any(word in content_lower for word in ['results', 'findings', 'data']):
            return 'results'
        elif any(word in content_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in content_lower for word in ['conclusion', 'summary']):
            return 'conclusion'
        elif any(word in content_lower for word in ['references', 'bibliography']):
            return 'references'
        else:
            return 'content'
    
    def _rank_and_diversify_strategic_results(self, results: List[Dict], intent) -> List[SourceInfo]:
        """Rank results strategically and ensure diversity."""
        # Remove duplicates (same chunk from different strategies)
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_key = (result['source'].document_id, result['source'].chunk_content[:100])
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x['source'].relevance_score, reverse=True)
        
        # Ensure document diversity
        final_sources = []
        document_counts = defaultdict(int)
        max_per_doc = max(2, config.search_results // 3)  # Allow 2-3 chunks per document
        
        for result in unique_results:
            doc_id = result['source'].document_id
            if document_counts[doc_id] < max_per_doc:
                final_sources.append(result['source'])
                document_counts[doc_id] += 1
            
            if len(final_sources) >= config.search_results:
                break
        
        return final_sources
    
    def _simple_search(self, query: str, document_ids: Optional[List[str]] = None) -> List[SourceInfo]:
        """Fallback simple search."""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=config.search_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            sources = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    relevance_score = max(0, 1 - distance)
                    sources.append(SourceInfo(
                        document_id=metadata['document_id'],
                        document_name=metadata['document_name'],
                        chunk_content=doc,
                        relevance_score=relevance_score
                    ))
            
            return sources
        except Exception:
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from vector database."""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(
                    where={"document_id": document_id}
                )
                return True
            return False
        except Exception:
            return False
    
    def cleanup_orphaned_documents(self, valid_document_ids: List[str]) -> Dict[str, int]:
        """Remove documents from vector database that are not in the valid list."""
        try:
            # Get all documents in vector database
            all_results = self.collection.get(include=["metadatas"])
            
            # Find orphaned documents
            orphaned_ids = set()
            valid_ids_set = set(valid_document_ids)
            
            for metadata in all_results['metadatas']:
                doc_id = metadata.get('document_id')
                if doc_id and doc_id not in valid_ids_set:
                    orphaned_ids.add(doc_id)
            
            # Delete orphaned documents
            deleted_count = 0
            for doc_id in orphaned_ids:
                if self.delete_document(doc_id):
                    deleted_count += 1
            
            return {
                "total_checked": len(all_results['metadatas']),
                "orphaned_found": len(orphaned_ids),
                "deleted": deleted_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        try:
            total_chunks = self.collection.count()
            
            # Get unique documents
            results = self.collection.get(include=["metadatas"])
            document_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    document_ids.add(metadata['document_id'])
            
            return {
                "status": "healthy",
                "total_chunks": total_chunks,
                "unique_documents": len(document_ids),
                "embedding_model": "all-MiniLM-L6-v2"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _ensure_document_diversity(self, results) -> List[SourceInfo]:
        """Ensure results include chunks from different documents."""
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # Group results by document
        document_groups = defaultdict(list)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            relevance_score = max(0, 1 - distance)
            
            source = SourceInfo(
                document_id=metadata['document_id'],
                document_name=metadata['document_name'],
                chunk_content=doc,
                relevance_score=relevance_score
            )
            
            document_groups[metadata['document_id']].append(source)
        
        # Select best chunks from each document
        diverse_sources = []
        max_per_document = max(1, config.search_results // len(document_groups)) if document_groups else 1
        
        # Sort documents by their best score
        sorted_docs = sorted(
            document_groups.items(),
            key=lambda x: max(s.relevance_score for s in x[1]),
            reverse=True
        )
        
        for doc_id, sources in sorted_docs:
            # Sort sources within document by relevance
            sorted_sources = sorted(sources, key=lambda x: x.relevance_score, reverse=True)
            diverse_sources.extend(sorted_sources[:max_per_document])
        
        # Fill remaining slots with best overall scores if needed
        if len(diverse_sources) < config.search_results:
            all_sources = []
            for sources in document_groups.values():
                all_sources.extend(sources)
            
            remaining_sources = [s for s in all_sources if s not in diverse_sources]
            remaining_sources.sort(key=lambda x: x.relevance_score, reverse=True)
            
            slots_remaining = config.search_results - len(diverse_sources)
            diverse_sources.extend(remaining_sources[:slots_remaining])
        
        return diverse_sources