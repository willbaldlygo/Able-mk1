# Able mk I - Architecture Quick Reference

## System Overview

**Type:** Multimodal AI Research Assistant  
**Status:** Production-ready, actively maintained  
**Last Major Update:** October 2025 (Skills integration)  

## Technology Stack

### Backend
- **Framework:** FastAPI (Python 3.11+)
- **Vector Database:** ChromaDB with sentence-transformers
- **Knowledge Graph:** Microsoft GraphRAG 2.6.0 + NetworkX
- **Search:** Hybrid (Vector + BM25 + Cross-encoder reranking)
- **Document Processing:** PyMuPDF + pdf2image
- **AI Models:** 
  - Anthropic Claude (text understanding)
  - LLava 7b via Ollama (vision analysis)

### Frontend
- **Type:** Vanilla JavaScript SPA
- **Location:** `new-ui/`
- **Styling:** Custom CSS with modern layouts
- **Icons:** Custom Able brain icon

### Infrastructure
- **Ports:**
  - 3001: Frontend (Python HTTP server)
  - 8000: Backend API (FastAPI/Uvicorn)
  - 11434: Ollama (optional, for LLava)
- **Data Storage:**
  - `sources/`: Original PDF files
  - `sources/images/`: Extracted/uploaded images
  - `data/vectordb/`: ChromaDB collections
  - `data/graphrag/`: Knowledge graph data
  - `data/document_metadata.json`: Lean metadata (~5.6KB per document set)

## Core Architecture Patterns

### Service Layer (backend/services/)

```
ai_service.py              → Claude API integration, prompt handling
document_service.py        → PDF processing, chunking, coordination
vector_service.py          → ChromaDB, embeddings, MMR diversity
multimodal_service.py      → LLava integration, image analysis
image_extractor.py         → PDF image extraction via pdf2image
graphrag_service.py        → Entity extraction, relationships
hybrid_search_service.py   → Coordinated search strategies
mcp_service.py             → MCP server lifecycle management
mcp_integration_service.py → MCP tool execution, chat integration
storage_service.py         → File operations, metadata persistence
transcription_service.py   → Voice-to-text (if enabled)
```

### Data Flow: Document Upload

```
1. User uploads PDF via /upload/multimodal
2. DocumentService orchestrates processing:
   a. Text extraction (PyMuPDF)
   b. Image extraction (ImageExtractor + pdf2image)
   c. Visual analysis (MultimodalService + LLava)
3. Text + image descriptions → Child chunks (700 tokens)
4. Child chunks → Parent chunks (2000 tokens)
5. GraphRAG entity extraction
6. Embeddings generation (sentence-transformers)
7. Storage in ChromaDB + metadata.json
8. Knowledge graph update (NetworkX)
```

### Data Flow: Chat Query

```
1. User submits query via /chat or /chat/enhanced
2. Query analysis (simple vs complex)
3. Search strategy selection:
   - Simple: Vector search
   - Complex: Hybrid (Vector + BM25)
4. Retrieval pipeline:
   a. Parallel vector + BM25 search (K=20)
   b. Union and deduplication
   c. Cross-encoder reranking (Top K=10)
   d. MMR diversification (Final K=8, λ=0.7)
   e. Parent context attachment
5. Context packing (token budget management)
6. Claude API call with system prompt
7. Response with source attribution
```

### MCP Integration Architecture

```
MCP Servers (backend/mcp/):
- filesystem_server.py → File read/write/list
- git_server.py        → Repository operations
- sqlite_server.py     → Database queries

Session Management:
- UUID-based isolation
- Path validation and restrictions
- Graceful shutdown with force-kill fallback
- 60-second timeout per command
```

## Key Design Decisions

### Always-On Multimodal
- No user toggles for vision processing
- Graceful degradation when LLava unavailable
- Visual context automatically integrated into chunks and search

### Child/Parent Chunking
- **Child:** 700 tokens (for precise retrieval)
- **Parent:** 2000 tokens (for full context)
- Each child stores parent reference
- Parent chunks provided to Claude for context

### Hybrid Retrieval Strategy
1. **Vector Search:** Semantic similarity (sentence-transformers)
2. **BM25 Lexical:** Exact keyword matching (rank-bm25)
3. **Cross-Encoder Reranking:** Query-passage relevance (BAAI/bge-reranker-base)
4. **MMR Diversification:** Balance relevance vs diversity (λ=0.7)
5. **Document Diversity Cap:** Max 3 results per document

### Metadata Optimization (August 2024)
- **Old:** 226KB for 4 documents (full chunk content stored)
- **New:** 5.6KB for 4 documents (97.5% reduction)
- **Strategy:** Store references only, content in ChromaDB

### Launch System
- **Primary:** `launch_able.py` (with automatic port management)
- **Legacy:** `start_able.py` (maintained for compatibility)
- **Features:**
  - Checks ports 3001, 8000, 11434
  - Gracefully terminates conflicting processes
  - Starts Ollama if needed
  - Downloads LLava model automatically
  - Opens browser to frontend

## API Endpoint Structure

### Core Endpoints
```
GET  /health                  → System health check
GET  /documents               → List documents with metadata
POST /upload/multimodal       → Upload PDFs or images
POST /chat                    → Simple chat with retrieval
POST /chat/enhanced           → Intelligent search routing
POST /chat/multimodal         → Chat with visual context
DELETE /documents/{id}        → Delete document + images
```

### Multimodal Endpoints
```
GET  /multimodal/capabilities      → Vision model status
GET  /documents/{id}/processing-info → Multimodal details
GET  /sources/images/{filename}    → Access extracted images
```

### GraphRAG Endpoints
```
GET  /graph/statistics                    → Knowledge graph metrics
GET  /documents/{id}/entities             → Document entities
GET  /entities/{name}/relationships       → Entity connections
POST /documents/{id}/reprocess-graph      → Rebuild graph
```

### MCP Endpoints
```
GET  /mcp/status    → Session and server status
POST /mcp/toggle    → Enable/disable MCP
POST /mcp/config    → Configure MCP servers
```

### Model Management
```
GET    /models/available   → List all models
POST   /models/switch      → Change active model
GET    /models/status      → Current model info
GET    /models/health      → Provider health checks
```

## Configuration Reference (.env)

### Required
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Optional - AI Providers
```bash
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
DEFAULT_AI_PROVIDER=anthropic
FALLBACK_ENABLED=true
```

### Optional - Processing
```bash
# Multimodal
IMAGE_PROCESSING_ENABLED=true
IMAGE_MAX_SIZE=2048
IMAGE_QUALITY=85

# Chunking
CHUNK_SIZE=700
PARENT_CHUNK_SIZE=2000
CHUNK_OVERLAP=50

# GraphRAG
GRAPHRAG_ENABLED=true
GRAPHRAG_ENTITY_TYPES=PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,CHART,DIAGRAM
GRAPHRAG_CHUNK_SIZE=1200
GRAPHRAG_CHUNK_OVERLAP=100

# Retrieval
RETRIEVAL_K=20
RETRIEVAL_RERANK_TOP_K=10
RETRIEVAL_FINAL_K=8
RETRIEVAL_MMR_LAMBDA=0.7
RETRIEVAL_DIVERSITY_CAP=3

# MCP
MCP_ENABLED=true
MCP_DEFAULT_ROOT=/path/to/allowed/directory
MCP_SESSION_TIMEOUT=3600
```

## File Structure

```
/
├── backend/
│   ├── main.py              # FastAPI app, routes
│   ├── config.py            # Configuration loading
│   ├── models.py            # Pydantic models
│   ├── services/            # Core business logic
│   ├── mcp/                 # MCP server implementations
│   ├── requirements.txt     # Python dependencies
│   └── venv/                # Virtual environment
├── new-ui/
│   ├── index.html           # SPA structure
│   ├── script.js            # Client-side logic
│   └── styles.css           # Styling
├── .claude/
│   └── skills/              # Claude Code skills
│       └── able-project-review/
├── data/
│   ├── vectordb/            # ChromaDB storage
│   ├── graphrag/            # GraphRAG data
│   └── document_metadata.json
├── sources/
│   ├── *.pdf                # Original documents
│   └── images/              # Extracted images
├── docs/
│   ├── claude-code/         # Claude Code instructions
│   └── AGENT_STATUS.md      # Changelog
├── archive/
│   ├── build-logs/          # Historical build records
│   ├── mothballed-react-frontend/
│   └── documentation/       # Archived docs
├── README.md                # Project overview
├── CLAUDE.md                # Claude Code guidance (693 lines)
└── launch_able.py           # Main launcher

```

## Common Operations

### Start Application
```bash
python3 launch_able.py
# or
python3 launch_able.py --force  # Force kill conflicting ports
```

### Check Status
```bash
python3 status_able.py
```

### Shutdown
```bash
python3 shutdown_able.py
```

### Access Points
- Frontend: http://localhost:3001
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Ollama: http://localhost:11434

## Dependencies Overview

### Critical Python Packages
```
fastapi==0.109.0          # Web framework
uvicorn==0.27.0           # ASGI server
anthropic==0.18.1         # Claude API
chromadb==0.4.22          # Vector database
sentence-transformers>=3.0.1  # Embeddings
pymupdf==1.23.26          # PDF processing
pdf2image==1.16.3         # PDF to images
Pillow==10.2.0            # Image processing
graphrag==2.6.0           # Knowledge graphs
networkx==3.2.1           # Graph operations
rank-bm25>=0.2.2          # Lexical search
pydantic==2.5.3           # Data validation
```

### Optional Packages
```
ollama (via pip)          # Local models
pytesseract               # OCR if needed
opencv-python             # Advanced image processing
```

## Security Considerations

### Path Restrictions
- MCP filesystem operations restricted to configured directories
- No access to system directories without explicit permission
- Path traversal attacks prevented via validation

### API Key Management
- Keys in `.env` only (never committed)
- `.gitignore` configured to exclude sensitive files
- API key validation on startup

### Session Isolation
- MCP sessions use UUID-based identification
- No cross-session data leakage
- Graceful cleanup on session end

## Performance Characteristics

### Document Processing Speed
- Text extraction: ~1-2 seconds per PDF page
- Image extraction: ~2-3 seconds per PDF page
- LLava analysis: ~5-10 seconds per image
- Embedding generation: ~0.5 seconds per chunk batch
- GraphRAG extraction: ~10-20 seconds per document

### Query Response Time
- Simple vector search: 100-300ms
- Hybrid search with reranking: 500-1000ms
- GraphRAG global search: 1-2 seconds
- Claude API call: 1-5 seconds (depends on context size)

### Resource Usage
- Memory: 8-16GB recommended (4GB ChromaDB + 4-8GB LLava)
- Storage: ~100MB per document (including vectors)
- CPU: Multi-core recommended for parallel processing

## Known Limitations

1. **LLava Dependency:** Requires 8GB+ RAM for vision model
2. **Sync Processing:** Some operations block request thread (optimization opportunity)
3. **No Real-time Updates:** Document reprocessing required for content changes
4. **Single-user:** No multi-tenancy or user authentication
5. **Local Only:** Not designed for distributed deployment

## Future Enhancement Opportunities

- Async LLava processing with job queue
- Incremental document updates (no full reprocessing)
- Multi-user support with authentication
- Distributed deployment configuration
- Advanced GraphRAG queries (multi-hop reasoning)
- Real-time collaboration features
