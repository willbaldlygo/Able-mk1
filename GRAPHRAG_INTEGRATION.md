# GraphRAG Integration Guide for Avi

## Overview

Avi now includes Microsoft GraphRAG integration, transforming it from a simple vector search system into an intelligent knowledge synthesis platform. This integration enables multi-hop reasoning, entity relationship analysis, and holistic document understanding.

## What is GraphRAG?

GraphRAG (Graph Retrieval-Augmented Generation) combines traditional vector search with knowledge graph analysis to provide:

- **Entity Extraction**: Automatically identifies people, organizations, concepts, technologies, and other entities
- **Relationship Mapping**: Discovers connections and relationships between entities
- **Community Detection**: Groups related entities and concepts into communities
- **Global Search**: Broad research questions using community summaries
- **Local Search**: Specific queries leveraging entity relationships
- **Multi-hop Reasoning**: Complex questions requiring connections across multiple documents

## New Features Added

### 1. Enhanced Backend Services

#### GraphRAG Service (`graphrag_service.py`)
- Microsoft GraphRAG framework integration
- Entity and relationship extraction
- Knowledge graph construction and management
- Global and local search capabilities
- Query type analysis and routing

#### Hybrid Search Service (`hybrid_search_service.py`)
- Intelligent query routing (global, local, or vector search)
- Automatic search strategy selection
- Enhanced source attribution with entity context
- Fallback mechanisms for reliability

#### Enhanced AI Service
- GraphRAG-aware prompt engineering
- Entity and relationship context integration
- Multi-search strategy support
- Enhanced response generation

### 2. New Data Models

#### Entity Information
```python
class EntityInfo(BaseModel):
    id: str
    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, etc.
    description: str
    source_document_id: str
    attributes: Dict[str, Any]
```

#### Relationship Information
```python
class RelationshipInfo(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    weight: float
    source_document_id: str
```

### 3. Enhanced API Endpoints

#### Enhanced Chat Endpoint
```
POST /chat/enhanced
```
Intelligent chat with automatic search strategy selection.

#### Knowledge Graph Statistics
```
GET /graph/statistics
```
Returns comprehensive graph statistics including entity counts, relationships, and communities.

#### Document Entities
```
GET /documents/{doc_id}/entities
```
Get all entities extracted from a specific document.

#### Entity Relationships
```
GET /entities/{entity_name}/relationships
```
Get all relationships for a specific entity.

#### Search Capabilities
```
GET /search/capabilities
```
Information about available search strategies and GraphRAG status.

#### Document Reprocessing
```
POST /documents/{doc_id}/reprocess-graph
```
Reprocess a document for GraphRAG extraction.

## Search Strategies

### Automatic Strategy Selection

The system automatically selects the best search strategy based on query analysis:

#### Global Search
**Best for**: Broad research questions, overviews, comparisons
**Indicators**: "overview", "summary", "trends", "compare", "across all documents"
**Example**: "What are the main themes across all my research papers?"

#### Local Search  
**Best for**: Specific entity queries, relationship exploration
**Indicators**: "relationship between", "specific", "who is", "connection"
**Example**: "What is the relationship between OpenAI and Microsoft?"

#### Vector Search
**Best for**: Simple factual queries, specific information lookup
**Indicators**: "define", "what does", "specifically mentions"
**Example**: "What does the document say about machine learning?"

### Manual Strategy Selection

You can force a specific search strategy:

```json
{
    "question": "Your question here",
    "search_type": "global",  // "global", "local", "vector", or "auto"
    "document_ids": ["doc1", "doc2"],  // optional
    "include_entities": true,
    "include_relationships": true
}
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# GraphRAG Configuration
GRAPHRAG_ENABLED=true
GRAPHRAG_ENTITY_TYPES=PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,METHOD
GRAPHRAG_MAX_GLEANINGS=1
GRAPHRAG_COMMUNITY_MAX_LENGTH=1500
GRAPHRAG_CHUNK_SIZE=1200
GRAPHRAG_CHUNK_OVERLAP=100
```

### Data Storage

GraphRAG data is stored in:
- `data/graphrag/input/` - Document input files for processing
- `data/graphrag/output/` - Generated knowledge graphs and communities
- `data/graphrag/cache/` - Processing cache
- `data/graphrag/logs/` - Processing logs

## Installation and Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

New dependencies added:
- `graphrag>=0.3.0`
- `networkx>=3.0`
- `pandas>=2.0.0` 
- `tiktoken>=0.5.0`
- `numpy>=1.24.0`

### 2. Verify Installation

Check GraphRAG availability:
```bash
GET /search/capabilities
```

### 3. Process Existing Documents

Reprocess existing documents for GraphRAG:
```bash
POST /documents/{doc_id}/reprocess-graph
```

## Usage Examples

### Enhanced Chat Query

```javascript
// Automatic strategy selection
const response = await fetch('/chat/enhanced', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: "What are the key relationships between AI companies mentioned in my documents?",
        search_type: "auto",
        include_entities: true,
        include_relationships: true
    })
});
```

### Get Document Entities

```javascript
const entities = await fetch('/documents/doc123/entities');
console.log(entities.entities); // Array of EntityInfo objects
```

### Explore Entity Relationships

```javascript
const relationships = await fetch('/entities/OpenAI/relationships');
console.log(relationships.relationships); // Array of RelationshipInfo objects
```

## Response Format

### Enhanced Chat Response

```json
{
    "answer": "Based on the knowledge graph analysis...",
    "sources": [
        {
            "document_id": "doc123",
            "document_name": "AI Research Paper",
            "chunk_content": "...",
            "relevance_score": 0.95,
            "entities": [
                {
                    "id": "entity1",
                    "name": "OpenAI",
                    "entity_type": "ORGANIZATION",
                    "description": "AI research company"
                }
            ],
            "relationships": [
                {
                    "source_entity": "OpenAI",
                    "target_entity": "GPT-4",
                    "relationship_type": "DEVELOPED",
                    "weight": 0.9
                }
            ]
        }
    ],
    "search_type": "local",
    "graph_insights": {
        "search_type": "local",
        "entities": [...],
        "relationships": [...],
        "completion_time": 2.3,
        "llm_calls": 3
    }
}
```

## Benefits

### For Users
- **Better Answers**: Multi-hop reasoning provides more comprehensive responses
- **Entity Understanding**: Clear identification of key people, organizations, and concepts
- **Relationship Discovery**: Understand connections across documents
- **Research Synthesis**: Holistic view of knowledge across document collection

### For Developers
- **Intelligent Routing**: Automatic selection of best search strategy
- **Rich Context**: Entity and relationship information for enhanced responses
- **Scalable Architecture**: Graceful fallback to vector search when needed
- **Comprehensive APIs**: Full access to knowledge graph data

## Performance Considerations

### Initial Processing
- GraphRAG processing adds ~30-60 seconds per document during upload
- Processing happens asynchronously to avoid blocking uploads
- Existing vector search remains available during GraphRAG processing

### Query Performance
- Global search: 2-5 seconds (community-based analysis)
- Local search: 1-3 seconds (entity relationship analysis)  
- Vector search: 0.5-1 second (traditional similarity search)
- Automatic fallback ensures reliability

### Resource Usage
- Additional ~200MB RAM for GraphRAG services
- ~50-100MB disk space per processed document
- CPU intensive during initial document processing

## Troubleshooting

### GraphRAG Not Available
1. Check dependencies: `pip install graphrag>=0.3.0`
2. Verify configuration: `GET /search/capabilities`
3. Check logs: `data/graphrag/logs/`

### Poor Entity Extraction
1. Ensure documents have clear text content
2. Consider reprocessing: `POST /documents/{doc_id}/reprocess-graph`
3. Check entity types in configuration

### Slow Performance
1. Monitor with: `GET /graph/statistics`
2. Use vector search fallback for simple queries
3. Consider processing fewer documents simultaneously

## Migration from Vector-Only Search

### Backward Compatibility
- All existing endpoints continue to work
- `/chat` endpoint uses enhanced vector search
- `/chat/enhanced` provides GraphRAG capabilities

### Gradual Migration
1. Start with automatic search strategy (`search_type: "auto"`)
2. Monitor performance and accuracy
3. Gradually enable GraphRAG features in UI
4. Use GraphRAG for complex research questions

### Data Migration
- Existing documents work with vector search immediately
- Reprocess documents for GraphRAG: `POST /documents/{doc_id}/reprocess-graph`
- New uploads automatically include GraphRAG processing

## Future Enhancements

- Real-time entity relationship visualization
- Advanced community analysis and clustering
- Cross-document citation and reference mapping
- Interactive knowledge graph exploration
- Batch processing for large document collections

---

This integration transforms Avi into a powerful research synthesis platform while maintaining the simplicity and reliability of the original vector search system.