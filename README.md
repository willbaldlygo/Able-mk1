# Able - Advanced PDF Research Assistant

An intelligent PDF research assistant with GraphRAG-inspired knowledge synthesis, voice input capabilities, and advanced multi-document reasoning.

> **Status**: ✅ Fully operational with lightweight GraphRAG alternative (Python 3.13 compatible)

## 🚀 Advanced Features

### **GraphRAG-Inspired Knowledge Synthesis**
- **Entity Extraction**: AI-powered identification of people, organizations, concepts (via Claude)
- **Relationship Mapping**: Automated discovery of connections between entities
- **Multi-hop Reasoning**: Complex queries spanning multiple documents with graph context
- **Intelligent Routing**: Automatic selection between global, local, and vector search strategies

### **Enhanced User Experience**
- **Voice Input**: Speech-to-text capabilities for hands-free operation
- **Professional UI**: 1600px layout with optimized design
- **Custom Branding**: Able identity with brain icon
- **Dynamic Interactions**: Smart button states and visual feedback

### **Advanced Search Capabilities**
- **Global Search**: Community-based insights across all documents
- **Local Search**: Entity-focused queries with relationship analysis
- **Vector Search**: Traditional similarity matching as fallback
- **Hybrid Intelligence**: Best of all search methods combined
- **Better Error Handling**: Graceful failures with user feedback
- **Modern Stack**: Latest React patterns with hooks and clean components

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
cd /Users/will/AVI\ BUILD/Able3_Main_WithVoiceMode
# Add your Anthropic API key to .env file
```

### 2. **Launch Application**
```bash
./start_able.sh
```

This will:
- Install all dependencies automatically
- Start backend server (port 8000)
- Start frontend server (port 3000)
- Open your browser to the application

### 3. **Start Researching**
- Upload PDF documents (supports batch upload)
- Ask questions about your documents
- Get intelligent responses with source attribution

## 🏗️ Architecture

### Backend Services
```
backend/
├── main.py              # Clean FastAPI application
├── config.py            # Environment configuration
├── models.py            # Pydantic data models
└── services/
    ├── storage_service.py    # File & metadata management
    ├── document_service.py   # PDF processing
    ├── vector_service.py     # Enhanced vector search
    └── ai_service.py         # Claude AI integration
```

### Frontend Components
```
frontend/src/
├── App.js                    # Main application
├── components/
│   ├── DocumentManager.js   # Upload & document management
│   └── ChatInterface.js     # Enhanced chat with sources
├── hooks/
│   └── useDocuments.js      # Document state management
└── services/
    └── api.js               # Clean API client
```

## ✨ Features

### **Document Management**
- Drag-and-drop upload for multiple files
- Readable filename preservation
- Document metadata with summaries
- Clean deletion with proper cleanup

### **Enhanced Search**
- Document diversity enforcement
- Configurable result count (default: 8)
- Relevance scoring and filtering
- Smart chunking with overlap

### **Intelligent Chat**
- Clear document vs excerpt distinction
- Source attribution with relevance scores
- Multi-document analysis
- Error recovery and user feedback

### **System Reliability**
- Atomic upload/delete operations
- Graceful error handling
- Clean state management
- Self-healing capabilities

## 🔧 Configuration

Environment variables in `.env`:
```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here

# GraphRAG Configuration
GRAPHRAG_ENABLED=true
GRAPHRAG_ENTITY_TYPES=PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,METHOD
GRAPHRAG_CHUNK_SIZE=1200
GRAPHRAG_CHUNK_OVERLAP=100

# Optional (with defaults)
CLAUDE_MODEL=claude-3-5-sonnet-20241022
SEARCH_RESULTS=8
CHUNK_SIZE=600
CHUNK_OVERLAP=100
MAX_TOKENS=1000
TEMPERATURE=0.1
```

## 📋 Technical Notes

**GraphRAG Implementation**: Currently using a lightweight GraphRAG alternative implemented with Claude AI for entity/relationship extraction and NetworkX for graph storage. This provides full GraphRAG functionality while maintaining Python 3.13 compatibility.

**Future Migration**: When official GraphRAG supports Python 3.13, the existing API structure will seamlessly support full GraphRAG with minimal changes.

## 📊 System Health

### **Access Points**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### **Health Indicators**
- Green dot in header = system online
- Document count display in header
- Upload/delete success feedback
- Chat response with source count

## 🔄 Development

### **Clean Reset**
```bash
rm -rf data backend/venv frontend/node_modules
```

### **Testing**
- Upload multiple documents
- Test batch upload
- Verify document diversity in search results
- Check source attribution accuracy

## 🚦 Next Steps

Able provides an advanced platform for research and knowledge synthesis:
- Document preview/viewer
- Export functionality
- Performance monitoring
- Advanced AI capabilities

---

**Built for reliability, designed for efficiency, enhanced for research.**