from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device=None):
        self.model = CrossEncoder(model_name, device=device)
    
    def rerank(self, query: str, candidates: list, top_k: int):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]