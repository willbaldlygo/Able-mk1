# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Able is a PDF Research Assistant that combines a Python FastAPI backend with a React frontend. It allows users to upload multiple PDFs, processes them using vector embeddings, and enables intelligent question-answering using Claude AI with source attribution.

## Recent Updates

### Version History - Able mk I (GraphRAG Integration)
- **Advanced Knowledge Synthesis**: Microsoft GraphRAG integration for intelligent research
- **Multi-hop Reasoning**: Complex queries spanning multiple documents with entity relationships
- **Intelligent Search Routing**: Automatic selection between global, local, and vector search
- **Entity & Relationship Extraction**: Automatic identification of people, organizations, concepts
- **Enhanced API Endpoints**: New GraphRAG-specific endpoints for advanced research capabilities

### Version History - Previous Version  
- **Branding Update**: System renamed back to Able throughout the application
- **UI Enhancements**: 
  - Custom brain icon implementation (`Able Icon.png`) as favicon and app logo
  - Wider UI layout (1600px max width) with 60/40 split favoring chat interface
  - Consistent button styling with proper height alignment and border weights
  - Send button color states: red (disabled) → white with red icon (active)
  - Voice input integration with visual feedback states

## Architecture

- **Backend** (`backend/`): FastAPI server with Pydantic models, document processing via PyMuPDF, vector storage with ChromaDB, GraphRAG knowledge graphs, and Claude AI integration
- **Frontend** (`new-ui/`): Modern web interface built with vanilla JavaScript, HTML, and CSS (port 3001)
- **Data Storage**: 
  - `sources/`: PDF file storage
  - `data/vectordb/`: ChromaDB vector database storage  
  - `data/graphrag/`: Microsoft GraphRAG knowledge graphs and community summaries
  - `document_metadata.json`: Document metadata tracking
- **Archive** (`archive/`): Historical components and development artifacts (see archive/README.md)

## Development Commands

### Quick Start (Recommended)
```bash
# Quick launcher with automatic port management
python3 launch_able.py

# Force kill processes on occupied ports if needed
python3 launch_able.py --force

# Check current port status
python3 launch_able.py --check
```

### Advanced Startup Options
```bash
# Full-featured startup script
python3 start_able.py

# Shell script version
./start_able.sh

# Manual port management
python3 port_manager.py --check      # Check port status
python3 port_manager.py --prepare    # Prepare all ports
```

### Manual Development Commands
```bash
# Frontend (New UI) - Manual start
cd new-ui
python3 -m http.server 3001   # Serve static files (port 3001)

# Backend (Python) - Manual start
cd backend
python -m venv venv && source venv/bin/activate  # Setup virtual environment
pip install -r requirements.txt                  # Install dependencies
python main.py                                   # Start FastAPI server (port 8000)
```

### Port Management
Able now includes automatic port management that:
- Checks required ports (3001, 8000) before startup
- Gracefully terminates conflicting processes
- Provides detailed status information
- Skips optional ports (11434 for Ollama) by default

Required ports:
- **3001**: New UI Frontend Server (primary interface)
- **8000**: Backend FastAPI Server
- **11434**: Ollama Local AI Server (optional)

**Note**: The new UI (port 3001) is the only active frontend. The old React frontend has been archived.

### macOS Dock App
Create a native macOS app that appears in the dock and Applications folder:

```bash
# Create the app bundle
python3 create_dock_app.py

# Create and install to Applications folder
./install_dock_app.sh

# Or create and manually install to Applications
python3 create_dock_app.py --install
```

The dock app provides:
- Native macOS app experience with custom Able icon
- Automatic port management and startup
- Browser auto-launch to http://localhost:3001 (New UI)
- Integration with macOS Spotlight, Launchpad, and Applications folder
- Proper app bundle structure for professional deployment

## Key Components

### Backend Architecture
- `main.py`: FastAPI application entry point with CORS configuration
- `models.py`: Pydantic data models (ChatRequest, ChatResponse, DocumentSummary, etc.)
- `document_processor.py`: PDF processing using PyMuPDF
- `vector_store.py`: ChromaDB vector storage management
- `llm_client.py`: Anthropic Claude AI client
- `document_manager.py`: Document lifecycle management

### Frontend Architecture (new-ui/)
- `index.html`: Main application structure with modern design
- `script.js`: JavaScript handling API communication, document management, and chat
- `styles.css`: CSS styling with gradient header and responsive layout
- Vanilla JavaScript implementation for maximum compatibility

## Environment Setup

### AI Provider Configuration

Required environment variables:
- `ANTHROPIC_API_KEY`: Claude AI API key (required for Anthropic functionality)

Optional Ollama configuration:
- `OLLAMA_ENABLED`: Enable/disable Ollama local models (default: true)
- `OLLAMA_HOST`: Ollama server host (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default Ollama model (default: llama3.1:8b)
- `DEFAULT_AI_PROVIDER`: Default AI provider - "anthropic" or "ollama" (default: anthropic)
- `FALLBACK_ENABLED`: Enable fallback between providers (default: true)

### Ollama Setup Instructions

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows - Download from https://ollama.ai/download
   ```

2. **Start Ollama Service:**
   ```bash
   ollama serve
   ```

3. **Download Models:**
   ```bash
   # Lightweight model for testing
   ollama pull tinyllama
   
   # Recommended models for production
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   ollama pull codellama:13b  # For code-related queries
   ```

4. **Verify Installation:**
   ```bash
   ollama list  # Should show downloaded models
   curl http://localhost:11434/api/tags  # API test
   ```

### GraphRAG Configuration
- `GRAPHRAG_ENABLED`: Enable/disable GraphRAG (default: true)
- `GRAPHRAG_ENTITY_TYPES`: Entity types to extract (default: PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,METHOD)
- `GRAPHRAG_CHUNK_SIZE`: GraphRAG chunk size (default: 1200)
- `GRAPHRAG_CHUNK_OVERLAP`: GraphRAG chunk overlap (default: 100)

## API Endpoints

### Core Endpoints
- `GET /health`: Health check
- `GET /documents`: List all documents
- `POST /upload`: Upload PDFs (now includes GraphRAG processing)
- `DELETE /documents/{doc_id}`: Delete document
- `POST /chat`: Ask questions about documents (traditional vector search)

### GraphRAG-Enhanced Endpoints  
- `POST /chat/enhanced`: Intelligent chat with automatic search routing
- `GET /graph/statistics`: Knowledge graph statistics
- `GET /documents/{doc_id}/entities`: Get entities from a document
- `GET /entities/{entity_name}/relationships`: Get entity relationships
- `GET /search/capabilities`: Available search strategies
- `POST /documents/{doc_id}/reprocess-graph`: Reprocess document for GraphRAG

### Voice & Web Endpoints
- `POST /api/transcribe`: Voice transcription
- `POST /scrape-url`: Web content scraping

### Model Management Endpoints
- `GET /models/available`: List available models from all providers
- `POST /models/switch`: Switch active AI model and provider
- `GET /models/status`: Current model status and health information
- `GET /models/health`: Comprehensive health check for all AI providers
- `GET /models/performance`: Performance metrics for all models
- `POST /models/download`: Download/pull a model from Ollama registry
- `DELETE /models/{model_name}`: Delete a model from local storage
- `GET /system/status`: System status and resource usage

- API documentation available at `http://localhost:8000/docs`

## Development Notes

### AI Provider Management
- Able supports dual AI providers: Anthropic Claude (remote) and Ollama (local)
- Default provider is Anthropic, with automatic fallback to Ollama if enabled
- Model switching is supported via API endpoints and can be done at runtime
- Performance metrics are tracked for all providers and models
- Health checks monitor both local Ollama service and Anthropic API connectivity

### Local vs Remote AI Models
- **Anthropic Claude**: High-quality remote models, requires API key and internet
- **Ollama Local**: Privacy-focused local models, runs entirely offline
- **Automatic Fallback**: If one provider fails, automatically tries the other
- **Model Performance**: Different models optimized for different use cases

### System Requirements for Ollama
- **Memory**: 8GB RAM minimum, 16GB+ recommended for larger models
- **Storage**: 4-50GB per model depending on size
- **CPU**: Modern multi-core processor recommended
- **GPU**: Optional but significantly improves performance

### Technical Implementation
- Frontend proxies API requests to `http://localhost:8000` via package.json proxy setting
- Vector embeddings use sentence-transformers for semantic search
- Document processing extracts text and metadata from PDFs
- Chat responses include source attribution with page references
- Model switching preserves conversation context across providers

## Troubleshooting

### Ollama Issues

**Ollama not connecting:**
1. Check if Ollama service is running: `ps aux | grep ollama`
2. Restart Ollama service: `ollama serve`
3. Verify port 11434 is available: `lsof -i :11434`
4. Check logs: `ollama logs`

**Models not downloading:**
1. Check internet connection for model downloads
2. Verify disk space: `df -h`
3. Try downloading manually: `ollama pull model_name`
4. Clear cache if needed: `rm -rf ~/.ollama/models/*`

**Performance issues:**
1. Monitor memory usage: Use `/system/status` endpoint
2. Consider smaller models if memory limited
3. Enable GPU acceleration if available
4. Increase system memory if possible

**API connection errors:**
1. Test direct API: `curl http://localhost:11434/api/tags`
2. Check firewall settings
3. Verify OLLAMA_HOST configuration
4. Restart both Ollama and Able services

### Provider Switching

**Fallback not working:**
1. Ensure `FALLBACK_ENABLED=true`
2. Check both providers are configured
3. Test individual connections via `/models/health`
4. Review logs for specific error messages

**Model switching fails:**
1. Verify model exists: Use `/models/available`
2. Check model is downloaded for Ollama
3. Validate API keys for Anthropic
4. Test connection before switching

## Archive System

The project includes an `archive/` directory containing historical components and development artifacts that are no longer actively used but preserved for reference and recovery:

### Archive Contents
- **`archive/mothballed-react-frontend/`**: Complete React frontend (formerly port 3000)
- **`archive/development-coordination/`**: Multi-agent development coordination system
- **`archive/build-logs/`**: Historical build logs and implementation documentation
- **`archive/prototypes/`**: UI prototypes and design assets
- **`archive/testing-scripts/`**: Development testing utilities

### Full Backup
A complete backup of the project state before archival is available at:
`/Users/will/AVI BUILD/Able3_Main_WithVoiceMode_BACKUP_20250824_103200`

### Recovery
See `archive/README.md` for detailed recovery instructions and archive contents. All archived components were fully functional when archived and can be restored if needed.

## Recent Improvements (August 24, 2025)

### 🔧 Metadata Optimization - MAJOR PERFORMANCE IMPROVEMENT
**Problem**: metadata.json was 226KB for only 4 documents due to storing full chunk content redundantly
**Solution**: Implemented lean metadata structure storing only references

**Changes Made**:
- Created new `DocumentMetadata` model without chunk content
- Updated `storage_service.py` to save optimized metadata
- Added backward compatibility for legacy format
- Created migration script (`migrate_metadata.py`)

**Results**:
- **97.5% storage reduction**: 226KB → 5.6KB  
- **220KB space saved** per document set
- Eliminated content duplication between metadata.json and ChromaDB
- Improved scalability for large document libraries

**Files Modified**:
- `backend/models.py`: Added DocumentMetadata model
- `backend/services/storage_service.py`: Optimized save/load methods
- `migrate_metadata.py`: Automated migration tool
- `data/metadata.json`: Now contains lean references only

### 🐛 Bug Fixes - CRITICAL ISSUES RESOLVED

#### Source Attribution Bug Fix
**Problem**: UI showed "undefined (48%)" instead of document names
**Root Cause**: Frontend accessed `source.document_title` but backend sends `document_name`
**Fix**: Updated `new-ui/script.js:221` to use correct field name
**File**: `new-ui/script.js`

#### Content Retrieval Verification  
**Issue**: System appeared to claim no theory-of-mind studies despite having relevant document
**Resolution**: Verified vector search working correctly - false alarm
**Evidence**: ChromaDB contains 56 chunks with proper theory-of-mind content retrieval

### 🎯 User Experience Enhancements

#### Model Selection Persistence
**Problem**: Selected LLM reset to Claude 3.5 Sonnet after browser refresh
**Solution**: Added localStorage persistence for model selection
**Implementation**:
- Save model selection to `localStorage` on switch
- Restore preferred model on page load  
- Automatic model switching on startup

**Files Modified**:
- `new-ui/script.js`: Added localStorage save/restore in `selectModel()` and `loadCurrentModel()`

#### UI Layout Improvements
**Problem**: Header elements (Able mk I, LLM indicator, Shutdown button) not properly aligned
**Solution**: Fixed flexbox layout with proper alignment
**Changes**:
- Added `.header-left` and `.header-right` flexbox containers
- Used `gap: 15px` for clean spacing
- Ensured all elements on same horizontal baseline

**Files Modified**:
- `new-ui/styles.css`: Added header layout rules
- `new-ui/index.html`: Cache busting with `?v=1.2`

### 🛠 Service Management System - COMPREHENSIVE SOLUTION

#### Problem Analysis
Multiple Able processes running without clear shutdown method:
- Able Launcher (bash script)
- Python launcher process  
- Frontend server (port 3001)
- Backend server (port 8000) - often missing
- Ollama (port 11434)

#### New Service Management Tools

**1. Comprehensive Shutdown Script (`shutdown_able.py`)**
- Finds all Able-related processes automatically
- Terminates in proper dependency order (backend → frontend → launchers)
- Creates detailed session logs with document counts
- Graceful termination with fallback to force-kill
- Clears ports and provides final status report

**2. Service Status Checker (`status_able.py`)**  
- Real-time status of all ports (3001, 8000, 11434)
- API health checking with fallback methods
- Document and data directory status
- Last session information
- Overall system health assessment
- Clear guidance on next actions

**3. Improved UI Shutdown Button**
- Added visual shutdown button in header (matches existing design)
- Confirmation dialog for safety
- Session logging integration
- Clean UI transition after shutdown
- Backend `/shutdown` endpoint for proper service termination

#### Usage Commands
```bash
# Check service status
python3 status_able.py

# Start all services  
python3 launch_able.py

# Stop all services cleanly
python3 shutdown_able.py
```

### 📊 Session Logging
- **Automatic logging**: All shutdowns logged to `session_log.jsonl`
- **Tracked data**: Timestamp, document count, processes terminated
- **Recovery info**: Session history for troubleshooting
- **Format**: JSON Lines for easy parsing and analysis

### 🔄 Backward Compatibility
- All existing functionality preserved
- Legacy metadata format still readable  
- Gradual migration without breaking changes
- Full backup system for recovery

### 🎉 Summary of Improvements
- **Performance**: 97.5% metadata storage reduction
- **Reliability**: Fixed source attribution bug
- **Usability**: Model persistence, better UI layout  
- **Management**: Comprehensive service control system
- **Logging**: Full session tracking and recovery
- **Scalability**: Optimized for large document libraries

All improvements maintain full backward compatibility while significantly enhancing performance, reliability, and user experience.

### 🔧 Multiple Document Upload Fix (January 2025)
**Problem**: Upload endpoint expected single file but frontend sent multiple files
**Root Cause**: Backend `upload_document(file: UploadFile)` vs frontend `formData.append('files', file)`
**Solution**: Updated backend to handle `files: List[UploadFile] = File(...)` with batch processing
**Result**: Multiple PDF upload now works correctly with proper error handling per file

### 🧠 Staged Reasoning Implementation (January 2025)
**Feature**: Advanced multi-stage response generation for complex queries
**Components**:
- **Stage 1**: Outline generation with key aspects and perspectives identification
- **Stage 2**: Multi-pass retrieval (overview → aspects → perspectives)
- **Stage 3**: Source diversification and comprehensive synthesis

**Technical Implementation**:
- New service: `StagedReasoningService` with 3-stage pipeline
- New endpoint: `/chat/staged` for comprehensive responses
- Frontend auto-detection: Complex questions (>10 words or analysis keywords) use staged reasoning
- Source diversification: Max 3 sources per document, ensures multi-document coverage

**Benefits**:
- Enhanced analysis for complex research questions
- Better source diversity across multiple documents
- Structured reasoning process with outline-first approach
- Automatic routing between fast chat and deep analysis

### 🔧 Launch Script Fix (Post-Archive)
**Problem**: `launch_able.py` failing because still looking for archived `frontend/` directory
**Root Cause**: `start_able.py` still configured for old React frontend instead of new-ui
**Solution**: Updated launch scripts to use new-ui structure

**Changes Made**:
- Updated `start_able.py` frontend directory path: `frontend/` → `new-ui/`
- Changed dependency checks: `package.json` → `index.html`
- Replaced npm environment check with new-ui file validation
- Updated frontend startup to use existing Python HTTP server method

**Files Modified**:
- `start_able.py`: Updated paths and dependency checks for new-ui

**Result**: Launch script now properly starts with new-ui architecture

### 🔍 Comprehensive Hybrid Retrieval Pipeline (January 2025)
**Feature**: Advanced multi-modal retrieval system combining vector, lexical, and semantic search
**Architecture**: Complete end-to-end pipeline with intelligent ranking and diversification

**Core Components**:
- **Child/Parent Chunking**: 700-token children with 2000-token parent context for better retrieval
- **BM25 Lexical Search**: Keyword-based search using `rank-bm25` for exact term matching
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base for semantic relevance scoring
- **MMR Diversification**: Maximum Marginal Relevance for result diversity
- **Token Budget Management**: Smart context packing with 30% answer, 10% instruction, 60% context allocation
- **Document Diversity**: Caps results per document to ensure multi-source coverage

**Technical Implementation**:
- **New Services**: `lexical_index.py`, `reranker.py`, `prompt_budget.py`, `retrieval.py`
- **Enhanced Services**: Updated `document_service.py`, `vector_service.py`, `main.py`
- **Dependencies**: Added `sentence-transformers>=3.0.1`, `rank-bm25>=0.2.2`, `scikit-learn>=1.4`
- **Debug Endpoint**: `/chat/debug` for pipeline analysis and performance monitoring

**Pipeline Flow**:
1. **Dual Search**: Parallel vector similarity + BM25 lexical search
2. **Union & Dedup**: Combine results with intelligent deduplication
3. **Cross-Encoder Rerank**: Semantic relevance scoring for query-passage pairs
4. **MMR Diversification**: Balance relevance vs diversity (λ=0.7)
5. **Document Diversity**: Cap results per document (max 3)
6. **Parent Context**: Attach full parent chunks for comprehensive context
7. **Token Budget**: Pack context within model limits with smart truncation

**Configuration** (`.env`):
```bash
# Retrieval Pipeline
RETRIEVAL_K=20
RETRIEVAL_RERANK_TOP_K=10
RETRIEVAL_FINAL_K=8
RETRIEVAL_MMR_LAMBDA=0.7
RETRIEVAL_DIVERSITY_CAP=3

# Token Management
TOKEN_BUDGET_TOTAL=200000
TOKEN_BUDGET_ANSWER_RATIO=0.3
TOKEN_BUDGET_INSTRUCTION_RATIO=0.1
TOKEN_BUDGET_CONTEXT_RATIO=0.6
```

**Benefits**:
- **Improved Recall**: BM25 catches exact terms missed by vector search
- **Better Ranking**: Cross-encoder provides superior relevance scoring
- **Enhanced Diversity**: MMR prevents redundant results
- **Optimal Context**: Parent chunks provide comprehensive information
- **Smart Budgeting**: Maximizes context within token limits
- **Debug Visibility**: Full pipeline analysis for optimization

**Files Modified**:
- `backend/main.py`: BM25 initialization, hybrid retriever setup, debug endpoint
- `backend/requirements.txt`: Added retrieval dependencies
- `backend/.env`: Comprehensive retrieval configuration
- `backend/services/document_service.py`: Child/parent chunking implementation
- `backend/services/vector_service.py`: MMR and diversity utilities
- `backend/lexical_index.py`: BM25 search implementation
- `backend/reranker.py`: Cross-encoder reranking
- `backend/prompt_budget.py`: Token budget management
- `backend/retrieval.py`: Hybrid retrieval orchestration