# Able mk I - Advanced PDF Research Assistant Build Description

## Project Overview

**Able mk I** is a sophisticated PDF Research Assistant that combines cutting-edge AI technologies with intuitive user interfaces to enable intelligent document analysis and question-answering. This build represents a complete evolution from traditional document processing systems to an advanced knowledge synthesis platform.

## Core Architecture

### Technology Stack
- **Backend**: Python FastAPI with advanced AI integration
- **Frontend**: Modern vanilla JavaScript web interface (port 3001)
- **AI Integration**: Anthropic Claude + Local Ollama models with automatic fallback
- **Document Processing**: PyMuPDF for PDF extraction
- **Vector Storage**: ChromaDB for semantic search
- **Knowledge Graphs**: Microsoft GraphRAG for entity relationships
- **Voice Processing**: Web Speech API with transcription capabilities
- **Model Context Protocol**: Full MCP integration for filesystem, git, and SQLite operations

### Advanced Features

#### 🧠 Intelligent Search Routing
- **Vector Search**: Semantic similarity using sentence-transformers
- **Lexical Search**: BM25 keyword-based exact term matching  
- **GraphRAG**: Entity relationship and community-based retrieval
- **Hybrid Pipeline**: Automatic selection of optimal search strategy
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base for relevance scoring

#### 🔄 Staged Reasoning System
- **Stage 1**: Outline generation with key aspects identification
- **Stage 2**: Multi-pass retrieval (overview → aspects → perspectives) 
- **Stage 3**: Source diversification and comprehensive synthesis
- **Auto-Detection**: Complex queries automatically use staged reasoning

#### 🔧 Comprehensive Retrieval Pipeline
- **Child/Parent Chunking**: 700-token children with 2000-token parent context
- **MMR Diversification**: Maximum Marginal Relevance for result variety
- **Token Budget Management**: Smart context packing within model limits
- **Document Diversity**: Caps results per document for multi-source coverage

## System Capabilities

### Document Management
- **Multi-PDF Upload**: Batch processing with per-file error handling
- **Metadata Optimization**: 97.5% storage reduction (226KB → 5.6KB per document set)
- **Source Attribution**: Page-level citations with confidence scoring
- **GraphRAG Processing**: Automatic entity and relationship extraction

### AI Provider Management
- **Dual Provider Support**: Anthropic Claude (remote) + Ollama (local)
- **Automatic Fallback**: Seamless switching between providers on failure
- **Model Performance Tracking**: Real-time metrics and health monitoring
- **Runtime Model Switching**: Change AI models without service restart

### Voice Integration
- **Web Speech API**: Real-time voice transcription
- **Visual Feedback**: Voice input states with UI indicators
- **Transcription Endpoint**: `/api/transcribe` for voice processing

### MCP (Model Context Protocol) Integration
- **Filesystem Operations**: Secure file read/write/list within boundaries
- **Git Integration**: Repository operations with access control
- **SQLite Queries**: Database access with path restrictions
- **Session Isolation**: UUID-based sessions with timeout management
- **Security Model**: Path validation and process isolation

## Performance Optimizations

### Storage Efficiency
- **Lean Metadata**: Eliminated content duplication between metadata.json and ChromaDB
- **220KB Space Saved**: Per document set through optimized structure
- **Backward Compatibility**: Legacy format still readable with automatic migration

### Retrieval Performance  
- **Hybrid Search**: BM25 + Vector + GraphRAG combination
- **Smart Context Packing**: 30% answer, 10% instruction, 60% context allocation
- **Cross-Encoder Ranking**: Superior relevance scoring vs. traditional methods

### Service Management
- **Automatic Port Management**: Graceful process termination and port cleanup
- **Session Logging**: Complete session tracking in `session_log.jsonl`
- **Health Monitoring**: Real-time system status and resource usage

## Development Infrastructure

### Quick Start System
```bash
# One-command startup with automatic port management
python3 launch_able.py

# Force cleanup if ports occupied  
python3 launch_able.py --force

# Service status checking
python3 status_able.py

# Clean shutdown with session logging
python3 shutdown_able.py
```

### macOS Integration
- **Native Dock App**: Professional app bundle for Applications folder
- **Spotlight Integration**: Searchable via macOS Spotlight
- **Auto-Launch**: Browser opens to http://localhost:3001 automatically

### Port Architecture
- **3001**: New UI Frontend Server (primary interface)
- **8000**: Backend FastAPI Server  
- **11434**: Ollama Local AI Server (optional)

## Configuration Management

### Environment Variables
```bash
# AI Provider Configuration
ANTHROPIC_API_KEY=your_key_here
DEFAULT_AI_PROVIDER=anthropic
FALLBACK_ENABLED=true

# Ollama Configuration  
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# GraphRAG Settings
GRAPHRAG_ENABLED=true
GRAPHRAG_ENTITY_TYPES=PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,METHOD

# Retrieval Pipeline
RETRIEVAL_K=20
RETRIEVAL_RERANK_TOP_K=10
RETRIEVAL_FINAL_K=8
RETRIEVAL_MMR_LAMBDA=0.7

# MCP Configuration
MCP_ENABLED=true
MCP_DEFAULT_ROOT=/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data
```

## API Ecosystem

### Core Endpoints
- **Document Management**: `/upload`, `/documents`, `/documents/{doc_id}`
- **Traditional Chat**: `/chat` for standard vector search
- **Enhanced Chat**: `/chat/enhanced` with intelligent routing
- **Staged Reasoning**: `/chat/staged` for complex analysis
- **Debug Pipeline**: `/chat/debug` for retrieval analysis

### AI Model Management
- **Model Operations**: `/models/available`, `/models/switch`, `/models/status`
- **Health Monitoring**: `/models/health`, `/models/performance`
- **Ollama Management**: `/models/download`, `/models/{model_name}` (DELETE)

### GraphRAG Integration
- **Graph Statistics**: `/graph/statistics`
- **Entity Relationships**: `/entities/{entity_name}/relationships`
- **Document Entities**: `/documents/{doc_id}/entities`
- **Search Capabilities**: `/search/capabilities`

### MCP Protocol
- **MCP Control**: `/mcp/status`, `/mcp/toggle`, `/mcp/config`
- **Tool Access**: Filesystem, Git, and SQLite operations via MCP

## Security Architecture

### MCP Security Model
- **Path Validation**: All operations restricted to configured boundaries
- **Session Isolation**: UUID-based sessions with independent process spaces
- **Graceful Termination**: Clean shutdown with force-kill fallback
- **Access Control**: Restricted filesystem, git repository, and database access

### Data Protection
- **Source Attribution**: All responses include document source tracking
- **Input Validation**: Comprehensive parameter validation on all endpoints
- **Process Isolation**: Independent sessions prevent cross-contamination

## Build Quality Assurance

### Automated Systems
- **Health Checks**: Continuous monitoring of all AI providers
- **Performance Metrics**: Real-time tracking across all models
- **Session Logging**: Complete session history for troubleshooting
- **Error Recovery**: Automatic fallback and retry mechanisms

### User Experience
- **Model Persistence**: Selected AI model remembered across sessions
- **Visual Feedback**: Real-time status indicators for all operations
- **Responsive Design**: 1600px max width with 60/40 chat-favoring split
- **Professional Branding**: Custom Able icon throughout interface

## Archive System

Complete historical preservation system:
- **`archive/mothballed-react-frontend/`**: Complete React frontend preservation
- **`archive/development-coordination/`**: Multi-agent development history
- **Full Backup**: Complete project state at `/Users/will/AVI BUILD/Able3_Main_WithVoiceMode_BACKUP_20250824_103200`

## Version Control Integration

### GitHub Repository
- **Repository**: https://github.com/willbaldlygo/Able-mk1
- **Status**: Fully synchronized with local development
- **Collaboration**: Complete version control capability

## Recent Major Improvements

### January 2025 Enhancements
- **Hybrid Retrieval Pipeline**: Complete BM25 + Vector + Cross-Encoder system
- **Staged Reasoning**: Multi-stage response generation for complex queries
- **Multiple Document Upload**: Fixed batch processing with per-file error handling
- **Token Budget Management**: Smart context allocation within model limits

### August 2025 MCP Integration
- **Complete MCP Protocol**: Filesystem, Git, and SQLite operations
- **Security Implementation**: Full session isolation and path validation
- **UI Integration**: Native MCP controls in web interface
- **Production Ready**: Tested and operational MCP system

## System Requirements

### Hardware Recommendations
- **Memory**: 16GB+ RAM for optimal Ollama performance
- **Storage**: 50GB+ for multiple AI models and document libraries
- **CPU**: Modern multi-core processor (Apple Silicon recommended)
- **Network**: Stable internet for Anthropic Claude API

### Software Dependencies
- **Python 3.8+**: Core runtime environment
- **Node.js**: Development tooling (optional)
- **Ollama**: Local AI model server
- **Git**: Version control integration

## Future Development Roadmap

### Intelligent MCP Integration
- **Auto-Detection**: Remove manual MCP toggle, intelligent tool usage
- **Context-Aware**: LLM-driven MCP tool selection based on queries
- **Seamless Integration**: Background MCP operations without user configuration

### Enhanced Analytics
- **Usage Metrics**: Comprehensive tracking and export capabilities
- **Performance Analytics**: Detailed retrieval pipeline optimization
- **User Behavior**: Interface usage patterns and optimization opportunities

## Build Status: Production Ready

**Able mk I** represents a complete, production-ready PDF research assistant with advanced AI capabilities, comprehensive security measures, and professional user experience. The system successfully combines multiple cutting-edge technologies into a cohesive, reliable platform for intelligent document analysis and research.

**Key Achievement**: 97.5% metadata optimization, complete MCP integration, and hybrid retrieval pipeline delivering superior research capabilities compared to traditional document processing systems.

---

*Generated on January 17, 2025 - Build Documentation*