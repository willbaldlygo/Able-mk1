#!/usr/bin/env python3
"""Test GraphRAG integration with existing documents."""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from services.document_service import DocumentService
from services.storage_service import StorageService

async def test_graphrag_with_existing_document():
    """Test GraphRAG processing with an existing document."""
    
    print("🧠 Testing GraphRAG Integration")
    print("=" * 50)
    
    # Initialize services
    doc_service = DocumentService()
    storage_service = StorageService()
    
    # Check GraphRAG availability
    if not doc_service.is_graphrag_available():
        print("❌ GraphRAG is not available")
        return
    
    print("✅ GraphRAG is available")
    
    # Get existing documents
    documents = storage_service.load_metadata()
    if not documents:
        print("❌ No documents found")
        return
    
    print(f"📚 Found {len(documents)} documents")
    
    # Pick the first document for testing
    doc_id, document = next(iter(documents.items()))
    print(f"🔍 Testing with document: {document.name}")
    
    # Test GraphRAG processing
    try:
        print("🚀 Processing document with GraphRAG...")
        graph_result = await doc_service.process_document_with_graph(document)
        
        if graph_result["success"]:
            print("✅ GraphRAG processing successful!")
            print(f"   📊 Entities extracted: {graph_result.get('entity_count', 0)}")
            print(f"   🔗 Relationships found: {graph_result.get('relationship_count', 0)}")
            
            # Show some entities if available
            entities = graph_result.get('entities', [])
            if entities:
                print("   🏷️  Sample entities:")
                for entity in entities[:5]:  # Show first 5
                    print(f"      - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
            
            # Show some relationships if available
            relationships = graph_result.get('relationships', [])
            if relationships:
                print("   🔗 Sample relationships:")
                for rel in relationships[:3]:  # Show first 3
                    print(f"      - {rel.get('source_entity', 'Unknown')} → {rel.get('target_entity', 'Unknown')}")
        else:
            print(f"❌ GraphRAG processing failed: {graph_result.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error during GraphRAG processing: {e}")
    
    # Test graph statistics
    try:
        print("\n📈 Graph Statistics:")
        stats = doc_service.get_graph_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error getting graph statistics: {e}")
    
    print("\n🎉 GraphRAG test completed!")

if __name__ == "__main__":
    asyncio.run(test_graphrag_with_existing_document())