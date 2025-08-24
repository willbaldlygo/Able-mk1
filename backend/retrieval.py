import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from lexical_index import BM25Index
from reranker import CrossEncoderReranker
from services.vector_service import mmr_select, cap_by_doc

class HybridRetriever:
    def __init__(self, vector_service, bm25_index: BM25Index):
        self.vector_service = vector_service
        self.bm25_index = bm25_index
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoderReranker(device=None)  # CPU for M4 Pro
        
        # Load env config
        self.k0 = int(os.getenv("RETRIEVAL_INITIAL_K", 20))
        self.klex = int(os.getenv("HYBRID_LEXICAL_K", 20))
        self.kfinal = int(os.getenv("RERANK_FINAL_K", 8))
        self.lam = float(os.getenv("MMR_LAMBDA", 0.5))
        self.cap = int(os.getenv("DIVERSITY_MAX_PER_DOC", 3))
    
    def get_candidates(self, query: str) -> List[Dict[str, Any]]:
        # Step A: Vector retrieve top k0
        query_embedding = self.embedder.encode([query]).tolist()[0]
        vector_results = self.vector_service.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k0,
            include=["embeddings", "metadatas", "documents", "ids"]
        )
        
        vector_candidates = []
        if vector_results['documents'] and vector_results['documents'][0]:
            for i, (doc, metadata, embedding, doc_id, distance) in enumerate(zip(
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['embeddings'][0],
                vector_results['ids'][0],
                vector_results['distances'][0]
            )):
                vector_candidates.append({
                    "id": doc_id,
                    "text": doc,
                    "vec": np.array(embedding),
                    "meta": metadata,
                    "vec_score": max(0, 1 - distance)
                })
        
        # Step B: Lexical retrieve top klex
        lexical_results = self.bm25_index.query(query, self.klex)
        
        # Step C: Union by id, compute embeddings for lexical-only
        candidates_dict = {c["id"]: c for c in vector_candidates}
        
        for lex_result in lexical_results:
            if lex_result["id"] not in candidates_dict:
                # Compute embedding for lexical-only item
                embedding = self.embedder.encode([lex_result["text"]])
                candidates_dict[lex_result["id"]] = {
                    "id": lex_result["id"],
                    "text": lex_result["text"],
                    "vec": embedding[0],
                    "meta": {"doc_id": lex_result["id"].split("_")[0]},  # Extract doc_id
                    "lex_score": lex_result["lex_score"],
                    "vec_score": 0.0
                }
        
        unioned = list(candidates_dict.values())
        
        # Step D: Rerank
        rerank_k = max(self.kfinal * 3, self.kfinal + 8)
        reranked = self.reranker.rerank(query, unioned, rerank_k)
        
        # Step E: MMR selection
        if reranked:
            query_vec = np.array(query_embedding)
            cand_vecs = np.array([c["vec"] for c in reranked])
            mmr_indices = mmr_select(query_vec, cand_vecs, self.lam, self.kfinal)
            mmr_selected = [reranked[i] for i in mmr_indices]
            
            # Apply diversity cap
            capped = cap_by_doc(mmr_selected, self.cap)
            
            # Refill if needed
            if len(capped) < self.kfinal:
                remaining = [c for c in reranked if c not in capped]
                needed = self.kfinal - len(capped)
                capped.extend(remaining[:needed])
        else:
            capped = []
        
        # Step F: Attach parent context
        for c in capped:
            c["attached_context"] = c["meta"].get("parent_text", c["text"])
        
        return capped