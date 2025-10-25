# Metadata Optimization Design

## Current Bloated Structure
```json
{
  "doc_id": {
    "id": "...",
    "name": "Theory_of_Mind_Prediction.pdf",
    "file_path": "...",
    "summary": "...",
    "chunks": [
      {
        "id": "chunk_id",
        "document_id": "doc_id",
        "content": "FULL 4KB CONTENT HERE...", // BLOAT!
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 4031
      }
    ],
    "created_at": "...",
    "file_size": 12345
  }
}
```

## NEW Optimized Structure
```json
{
  "doc_id": {
    "id": "doc_id",
    "name": "Theory_of_Mind_Prediction.pdf", 
    "file_path": "/path/to/file.pdf",
    "summary": "Brief document summary...",
    "created_at": "2025-08-24T10:45:00",
    "file_size": 12345,
    "chunk_count": 14,
    "chunk_ids": ["chunk_1", "chunk_2", "chunk_3", ...],
    "source_type": "pdf",
    "original_url": null
  }
}
```

## Key Changes
1. **Remove** `chunks` array with full content
2. **Add** `chunk_count` for quick stats
3. **Add** `chunk_ids` array for references
4. **Keep** essential metadata for document management
5. **Content access** via ChromaDB only (single source of truth)

## Storage Impact
- **Before**: 226KB for 4 docs (56 chunks × 3.8KB content each)
- **After**: ~3-5KB for same 4 docs (metadata only)
- **Reduction**: 98% storage reduction
- **Content access**: Same via vector_service.collection.get()

## API Compatibility
- DocumentSummary model unchanged (already lean)
- Document model used only for internal processing
- Frontend gets DocumentSummary (no chunks needed)
- Chat uses vector search (already content-independent)