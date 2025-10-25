#!/usr/bin/env python3
"""Debug script to test vector search functionality."""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from services.vector_service import VectorService
import chromadb
from chromadb.config import Settings

def main():
    print("🔍 Debugging Vector Search for Theory of Mind")
    print("=" * 50)
    
    # Initialize vector service
    try:
        vector_service = VectorService()
        print("✅ Vector service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize vector service: {e}")
        return
    
    # Check collection stats
    try:
        collection = vector_service.collection
        total_chunks = collection.count()
        print(f"📊 Total chunks in database: {total_chunks}")
    except Exception as e:
        print(f"❌ Failed to get collection stats: {e}")
        return
    
    # Check for Theory of Mind document chunks
    try:
        # Query for theory of mind content
        results = collection.query(
            query_texts=["theory of mind"],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\n🔍 Search Results for 'theory of mind':")
        print(f"Found {len(results['documents'][0]) if results['documents'] else 0} results")
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0][:3],  # Show first 3
                results['metadatas'][0][:3],
                results['distances'][0][:3]
            )):
                print(f"\n--- Result {i+1} ---")
                print(f"Document: {meta.get('document_name', 'Unknown')}")
                print(f"Distance: {dist:.4f} (lower = better match)")
                print(f"Content preview: {doc[:200]}...")
        
        # Check specifically for Theory_of_Mind_Prediction.pdf
        theory_results = collection.query(
            query_texts=["ChatGPT theory of mind prediction"],
            n_results=5,
            where={"document_name": "Theory_of_Mind_Prediction.pdf"},
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"\n📄 Specific search in Theory_of_Mind_Prediction.pdf:")
        print(f"Found {len(theory_results['documents'][0]) if theory_results['documents'] else 0} results")
        
        if theory_results['documents'] and theory_results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                theory_results['documents'][0],
                theory_results['metadatas'][0],
                theory_results['distances'][0]
            )):
                print(f"\n--- Theory Doc Result {i+1} ---")
                print(f"Distance: {dist:.4f}")
                print(f"Content: {doc[:300]}...")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    # List all documents in the collection
    try:
        print(f"\n📚 All documents in collection:")
        all_results = collection.get(include=["metadatas"])
        doc_names = set()
        for meta in all_results['metadatas']:
            doc_names.add(meta.get('document_name', 'Unknown'))
        
        for doc_name in sorted(doc_names):
            print(f"  • {doc_name}")
            
    except Exception as e:
        print(f"❌ Failed to list documents: {e}")

if __name__ == "__main__":
    main()