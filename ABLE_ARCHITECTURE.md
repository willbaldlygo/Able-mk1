# Avi Architecture Plan
**Advanced PDF Research Assistant with GraphRAG**

## Design Principles

### 1. **Simplicity First**
- Minimal, clean codebase without defensive over-engineering
- Single responsibility for each component
- Clear, readable code structure

### 2. **Robust by Design** 
- Build reliability in from the start, not as patches
- Atomic operations without complex validation layers
- Self-contained components with clear interfaces

### 3. **Advanced Intelligence**
- Microsoft GraphRAG knowledge synthesis
- Intelligent search routing (global/local/vector)
- Multi-hop reasoning capabilities
- Voice input integration

## Core Architecture

### Backend Structure
```
backend/
├── main.py              # FastAPI app with clean endpoint definitions
├── models.py            # Pydantic models (simplified, focused)
├── services/
│   ├── document_service.py    # Document processing & management
│   ├── vector_service.py      # ChromaDB operations
│   ├── ai_service.py          # Claude AI integration
│   └── storage_service.py     # File system operations
├── config.py            # Configuration management
├── utils.py             # Shared utilities
└── requirements.txt     # Dependencies
```

### Frontend Structure
```
frontend/
├── src/
│   ├── App.js                 # Main app component
│   ├── components/
│   │   ├── DocumentManager.js # Combined upload/list/management
│   │   ├── ChatInterface.js   # Enhanced chat with features
│   │   ├── SearchResults.js   # Better result display
│   │   └── DocumentViewer.js  # Preview capability
│   ├── services/
│   │   └── api.js            # Clean API client
│   ├── hooks/
│   │   └── useDocuments.js   # Document state management
│   └── styles/
│       └── globals.css       # Clean, modern styling
├── package.json
└── tailwind.config.js
```

### Data Management
```
data/
├── vectordb/           # ChromaDB storage
├── sources/           # PDF files (readable names)
├── metadata.json      # Document metadata
└── config.json       # App configuration
```

## Key Improvements Over Able2

### 1. **Enhanced Search**
- **Document Diversity**: Ensure results from multiple documents
- **Relevance Scoring**: Configurable thresholds
- **Result Count**: 8-10 chunks vs 5
- **Smart Chunking**: Better overlap and sizing

### 2. **Better User Experience**
- **Batch Upload**: Multiple files at once
- **Document Preview**: View PDFs inline
- **Search Highlighting**: Show relevant sections
- **Export Options**: Results to text/markdown

### 3. **Clean Data Flow**
```
Upload → Validate → Process → Store → Index → Confirm
Delete → Remove → Cleanup → Update UI → Confirm
Search → Query → Diversify → Rank → Present
```

### 4. **Modern UI Features**
- Drag-and-drop for multiple files
- Document thumbnails/previews
- Search result highlighting
- Progress indicators
- Error boundaries

### 5. **Robust Configuration**
- Environment-based settings
- Configurable search parameters
- Claude model selection
- Performance tuning options

## Implementation Strategy

### Phase 1: Core Foundation
1. Clean backend with services architecture
2. Robust data management built-in
3. Basic frontend with modern components
4. Automated setup and testing

### Phase 2: Enhanced Features
1. Document diversity in search
2. Batch upload capability
3. Document preview/viewer
4. Export functionality

### Phase 3: Advanced Features
1. Document comparison
2. Automated summarization  
3. Performance optimization
4. Advanced configuration

## Technical Specifications

### Backend Technologies
- **FastAPI**: Latest version with async support
- **ChromaDB**: Vector database with proper configuration
- **PyMuPDF**: PDF processing with error handling
- **Anthropic**: Claude 3.5 Sonnet with proper context management
- **Pydantic**: V2 for data validation

### Frontend Technologies  
- **React 18**: With hooks and modern patterns
- **Tailwind CSS**: Clean, responsive design
- **React Query**: Server state management
- **React Dropzone**: Enhanced file upload
- **React PDF**: Document preview capability

### Development Tools
- **pytest**: Comprehensive testing
- **Black**: Code formatting
- **ESLint/Prettier**: Frontend linting
- **Docker**: Optional containerization

## Quality Assurance

### Built-in Testing
- Unit tests for all services
- Integration tests for API endpoints
- Frontend component testing
- End-to-end workflow testing

### Performance Monitoring
- Upload/processing time tracking
- Search response time monitoring
- Memory usage optimization
- Database performance tuning

### Error Handling
- Graceful failure handling
- User-friendly error messages
- Automatic recovery where possible
- Comprehensive logging

This architecture ensures Avi will be intelligent, scalable, and feature-complete with GraphRAG integration, building on the stable foundation from previous versions.