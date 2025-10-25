# Able mk I - Advanced Multimodal Research Assistant

An intelligent document research assistant with **multimodal processing capabilities**, GraphRAG knowledge synthesis, voice input, and advanced AI integration. Able mk I processes both text and images from PDFs, providing comprehensive visual analysis alongside traditional document understanding.

> **Status**: ✅ Fully operational with **multimodal document processing** and llava vision model integration

## 🧠 Core Multimodal Capabilities

### **Visual Document Understanding**
- **PDF Image Extraction**: Automatic extraction and analysis of images from uploaded PDFs
- **Direct Image Upload**: Support for PNG, JPG, JPEG files with AI-powered visual descriptions
- **LLava Vision Model**: Advanced image analysis using locally-hosted llava model via Ollama
- **Visual Context Integration**: Image descriptions embedded into document chunks for enhanced search
- **Graceful Degradation**: System operates with metadata-only processing when vision model unavailable

### **Enhanced Document Processing**
- **Dual Processing Pipeline**: Simultaneous text extraction and image analysis
- **Child/Parent + Visual Chunking**: 700/2000 token chunking enhanced with visual descriptions
- **GraphRAG Entity Extraction**: Automatic extraction of entities (PERSON, ORGANIZATION, CONCEPT, etc.) and relationships
- **Knowledge Graph Integration**: Entities and relationships stored in NetworkX graphs for advanced querying
- **Multimodal Search**: Visual context integrated into semantic search results
- **Source Attribution**: Documents with images clearly marked with visual indicators

### **Intelligent Multimodal Chat**
- **Visual Q&A**: Ask questions about charts, diagrams, and images in your documents
- **Cross-Modal Retrieval**: Find content based on text queries that reference visual elements
- **GraphRAG-Enhanced Responses**: AI responses incorporate entity relationships and cross-document connections
- **Intelligent Search Routing**: Automatic selection of global, local, or vector search based on query type
- **Enhanced Responses**: AI responses incorporate both textual and visual context
- **Visual Source Indicators**: Chat responses show which sources contain visual content

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
cd /Users/will/AVI\ BUILD/Able3_Main_WithVoiceMode
# Add your Anthropic API key to .env file
# Ensure Ollama is installed: brew install ollama
```

### 2. **Launch Application**
```bash
python3 launch_able.py
```

This automatically:
- Starts Ollama service (port 11434)
- Downloads llava model if needed
- Starts backend server (port 8000) with multimodal endpoints
- Starts frontend server (port 3001) with enhanced UI
- Opens browser to the multimodal interface

### 3. **Start Multimodal Research**
- **Upload PDFs**: System automatically extracts text AND images
- **Upload Images**: Direct image analysis with visual descriptions
- **Ask Visual Questions**: "What does the chart in the document show?"
- **Get Enhanced Responses**: Answers incorporating both text and visual context

## 🏗️ Advanced Architecture

### **Multimodal Backend Services**
```
backend/
├── main.py                    # FastAPI with multimodal endpoints
├── services/
│   ├── multimodal_service.py  # Core llava integration and image processing
│   ├── image_extractor.py     # PDF image extraction using pdf2image
│   ├── document_service.py    # Enhanced with multimodal processing
│   ├── ai_service.py          # Multimodal prompt handling
│   └── vector_service.py      # Visual context in embeddings
└── models.py                  # Multimodal data models
```

### **Enhanced Frontend Interface**
```
new-ui/
├── index.html    # Multimodal upload support (PDFs + images)
├── script.js     # Enhanced with visual indicators and image galleries
└── styles.css    # Visual badges, thumbnails, and modal styling
```

### **AI Integration Stack**
- **Text Analysis**: Anthropic Claude for document understanding
- **Vision Analysis**: LLava model via Ollama for image descriptions
- **Hybrid Processing**: Automatic fallback between vision and metadata-only modes
- **Model Management**: Runtime switching between text and vision models

## ✨ Multimodal Features

### **Document Library with Visual Indicators**
- **📸 Visual Badges**: Documents with images show multimodal indicators
- **Image Count Display**: Number of extracted/analyzed images per document
- **👁️ Description Badges**: Visual description availability indicators
- **Image Gallery**: "View Images" buttons open modal galleries with thumbnails

### **Enhanced Upload Experience**
- **Smart Detection**: Automatic routing to multimodal processing endpoints
- **Visual Feedback**: Processing indicators show when images are being analyzed
- **Batch Processing**: Handle mixed PDF and image uploads simultaneously
- **Progress Indicators**: Real-time feedback on vision model processing

### **Multimodal Search & Chat**
- **Visual Source Markers**: 🖼️ indicators show sources with visual content
- **Enhanced Context**: Responses incorporate image descriptions when relevant
- **Cross-Modal Queries**: Find documents based on visual content descriptions
- **Visual Analysis Status**: Shows when visual processing was used in responses

## 🔧 Multimodal Configuration

Environment variables in `.env`:
```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Multimodal Processing
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llava:7b

# GraphRAG + Visual Context
GRAPHRAG_ENABLED=true
GRAPHRAG_ENTITY_TYPES=PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,CHART,DIAGRAM

# Enhanced Search
RETRIEVAL_K=20
RETRIEVAL_FINAL_K=8
RETRIEVAL_MMR_LAMBDA=0.7

# MCP Integration
MCP_ENABLED=true
```

## 📋 System Requirements

### **Multimodal Processing Requirements**
- **Memory**: 16GB+ RAM (llava model requires ~8GB)
- **Storage**: 10GB+ for llava model and image processing
- **CPU**: Modern multi-core processor (Apple Silicon recommended)
- **GPU**: Optional but significantly improves llava performance

### **Model Dependencies**
- **Ollama**: Local AI model server
- **LLava Model**: Vision model for image analysis
- **Anthropic Claude**: Text processing and reasoning
- **ChromaDB**: Vector storage with visual metadata

## 🎯 API Endpoints

### **Multimodal Processing**
- **`POST /upload/multimodal`**: Enhanced document upload with vision processing
- **`POST /chat/multimodal`**: Chat with visual context integration
- **`GET /documents/{id}/processing-info`**: Multimodal document details
- **`GET /multimodal/capabilities`**: System vision capabilities status

### **Traditional Endpoints** (Enhanced)
- **`POST /upload`**: Now routes to multimodal processing
- **`POST /chat`**: Enhanced with visual context when available
- **`POST /chat/enhanced`**: Intelligent routing with multimodal support

### **Visual Content Management**
- **`GET /sources/images/{filename}`**: Access extracted/uploaded images
- **Image Gallery API**: Thumbnail generation and modal viewing

## 🚦 Development & Troubleshooting

### **Multimodal Status Checking**
```bash
# Check all services including llava
python3 status_able.py

# Test multimodal capabilities
curl http://localhost:8000/multimodal/capabilities

# Check ollama models
ollama list
```

### **Model Management**
```bash
# Install llava model
ollama pull llava:7b

# Check model status
ollama list | grep llava

# Test llava directly
ollama run llava:7b "Describe this image" --image path/to/image.jpg
```

### **Visual Processing Pipeline**
1. **PDF Upload** → Image extraction → LLava analysis → Visual descriptions → Enhanced chunks
2. **Image Upload** → Direct llava analysis → Visual descriptions → Metadata storage
3. **Chat Query** → Multimodal search → Visual context integration → Enhanced response

## 🎉 Recent Multimodal Implementation

### **January 2025 - Multimodal Integration Complete**
- **✅ 3-Agent Development**: Architect, Backend, Frontend specialists
- **✅ Core Services**: `MultimodalService`, `ImageExtractor` implementation
- **✅ LLava Integration**: Complete vision model integration via Ollama
- **✅ Enhanced UI**: Visual indicators, image galleries, processing feedback
- **✅ API Endpoints**: Full multimodal API with graceful degradation
- **✅ Documentation**: Comprehensive multimodal documentation

### **January 2025 - GraphRAG Integration Complete**
- **✅ Microsoft GraphRAG 2.6.0**: Installed with Python 3.11 compatibility
- **✅ Configuration Setup**: GraphRAG initialized in `/data/graphrag/` with proper settings
- **✅ Service Integration**: `GraphRAGService` fully integrated into `DocumentService` and API
- **✅ Entity Extraction**: Automatic entity and relationship extraction from uploaded PDFs
- **✅ Knowledge Graph**: NetworkX-based graph with entities, relationships, and statistics
- **✅ Enhanced Search**: Global/local/vector search routing with entity context
- **✅ API Endpoints**: Complete GraphRAG API (`/graph/statistics`, `/entities/{name}/relationships`)
- **✅ AI Service Fix**: Disabled response formatting for reliable JSON extraction

### **Architecture Decisions**
- **Always-On Multimodal**: No toggle needed - core functionality
- **Graceful Degradation**: Works with or without vision model
- **Intelligent Processing**: Automatic visual analysis when available
- **Enhanced Storage**: Visual metadata integrated with text chunks
- **GraphRAG Integration**: Automatic entity extraction on document upload
- **Lightweight Implementation**: JSON-based storage with NetworkX graphs for performance

## 📊 System Health

### **Access Points**
- **Frontend**: http://localhost:3001 (Enhanced multimodal interface)
- **Backend API**: http://localhost:8000 (Multimodal endpoints)
- **API Documentation**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434 (Vision model server)

### **Health Indicators**
- **🟢 System Online**: All services operational
- **🧠 Multimodal Ready**: LLava model available
- **📸 Visual Processing**: Image analysis active
- **🔄 Graceful Mode**: Text-only fallback when needed

## 🔄 Advanced Capabilities

### **Knowledge Synthesis**
- **GraphRAG Integration**: Entity and relationship extraction from text AND images
- **Multi-hop Reasoning**: Complex queries spanning visual and textual content
- **Hybrid Intelligence**: Vector + Lexical + Visual search combination
- **MCP Protocol**: Filesystem, Git, SQLite operations for enhanced research

### **Professional Features**
- **Session Management**: Complete session logging with visual processing metrics
- **Performance Monitoring**: Real-time tracking of vision model performance
- **Model Switching**: Runtime switching between text and vision models
- **Archive System**: Complete project history and recovery capabilities

---

**Able mk I - Where Text Meets Vision for Advanced Research**

*Built with multimodal-first architecture, powered by Claude + LLava, designed for comprehensive document understanding.*