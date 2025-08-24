from rank_bm25 import BM25Okapi
import re

TOKEN = re.compile(r"\w+")

def _tok(s: str):
    return TOKEN.findall(s.lower())

class BM25Index:
    def __init__(self):
        self.docs = []
        self.ids = []
        self.toks = []
        self.bm = None
    
    def build(self, records):
        self.docs = [r["text"] for r in records]
        self.ids = [r["id"] for r in records]
        self.toks = [_tok(t) for t in self.docs]
        self.bm = BM25Okapi(self.toks)
    
    def query(self, q: str, k: int):
        if not self.bm:
            return []
        scores = self.bm.get_scores(_tok(q))
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{"id": self.ids[i], "text": self.docs[i], "lex_score": float(scores[i])} for i in idxs]