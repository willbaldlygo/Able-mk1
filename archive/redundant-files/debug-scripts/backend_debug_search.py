#!/usr/bin/env python3
"""Debug script for vector search."""

from services.vector_service import VectorService

def main():
    print("🔍 Debugging Vector Search")
    print("=" * 40)
    
    # Initialize
    vector_service = VectorService()
    collection = vector_service.collection
    
    # Get counts
    total_chunks = collection.count()
    print(f"📊 Total chunks: {total_chunks}")
    
    # Test theory of mind search
    print(f"\n🔍 Searching for 'theory of mind'...")
    results = collection.query(
        query_texts=["theory of mind prediction ChatGPT"],
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )
    
    found = len(results['documents'][0]) if results['documents'] else 0
    print(f"Found {found} results")
    
    if results['documents'] and results['documents'][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0][:5],
            results['metadatas'][0][:5], 
            results['distances'][0][:5]
        )):
            print(f"\n--- Result {i+1} ---")
            print(f"Doc: {meta.get('document_name', 'Unknown')}")
            print(f"Distance: {dist:.3f}")
            print(f"Preview: {doc[:150]}...")
    
    # List all document names
    print(f"\n📚 Documents in collection:")
    all_results = collection.get(include=["metadatas"])
    doc_names = set()
    for meta in all_results['metadatas']:
        doc_names.add(meta.get('document_name', 'Unknown'))
    
    for doc_name in sorted(doc_names):
        print(f"  • {doc_name}")
        
    # Test strategic search
    print(f"\n🎯 Testing strategic search...")
    try:
        sources = vector_service.strategic_search("theory of mind studies in ChatGPT")
        print(f"Strategic search found {len(sources)} sources:")
        for i, source in enumerate(sources[:3]):
            print(f"  {i+1}. {source.document_name} ({source.relevance_score:.3f})")
    except Exception as e:
        print(f"Strategic search failed: {e}")

if __name__ == "__main__":
    main()