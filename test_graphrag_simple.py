#!/usr/bin/env python3
"""Simple GraphRAG test with mock data."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from services.graphrag_service import GraphRAGService

async def test_graphrag_simple():
    """Test GraphRAG with simple mock data."""
    
    print("🧠 Testing GraphRAG Core Functionality")
    print("=" * 50)
    
    # Initialize GraphRAG service
    graphrag_service = GraphRAGService()
    
    # Check availability
    if not graphrag_service.is_available():
        print("❌ GraphRAG is not available")
        return
    
    print("✅ GraphRAG is available")
    
    # Test with mock entities and relationships
    mock_entities = [
        {
            "id": "entity_1",
            "name": "Stanford University",
            "type": "ORGANIZATION",
            "description": "Research university",
            "source_document_id": "test_doc",
            "source_document_name": "Test Document",
            "attributes": {}
        },
        {
            "id": "entity_2", 
            "name": "Machine Learning",
            "type": "CONCEPT",
            "description": "AI research field",
            "source_document_id": "test_doc",
            "source_document_name": "Test Document",
            "attributes": {}
        }
    ]
    
    mock_relationships = [
        {
            "id": "rel_1",
            "source_entity": "Stanford University",
            "target_entity": "Machine Learning",
            "relationship_type": "researches",
            "description": "Stanford conducts ML research",
            "weight": 1.0,
            "source_document_id": "test_doc",
            "attributes": {}
        }
    ]
    
    print("🔧 Testing graph update with mock data...")
    
    # Test graph update
    try:
        graphrag_service._update_graph_with_extracted_data(mock_entities, mock_relationships)
        print("✅ Graph update successful")
        
        # Test statistics
        stats = graphrag_service.get_graph_statistics()
        print(f"📊 Graph nodes: {stats['graph_nodes']}")
        print(f"📊 Graph edges: {stats['graph_edges']}")
        
        # Test entity relationships
        relationships = graphrag_service.get_entity_relationships("Stanford University")
        print(f"🔗 Stanford relationships: {len(relationships)}")
        
        if relationships:
            for rel in relationships:
                print(f"   - {rel['target']} ({rel['relationship']})")
        
        # Test query analysis
        query_types = [
            "What are the main themes in the documents?",
            "How is Stanford University connected to machine learning?",
            "Tell me about specific research methods"
        ]
        
        print("\n🔍 Testing query analysis:")
        for query in query_types:
            query_type = graphrag_service.analyze_query_type(query)
            print(f"   '{query[:40]}...' → {query_type}")
        
        print("\n✅ GraphRAG core functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during GraphRAG test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graphrag_simple())