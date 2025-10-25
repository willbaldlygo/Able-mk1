# Avi WebUI Integration Guide

## Overview
This guide provides complete specifications for creating a new webUI that seamlessly integrates with Avi's backend API without damaging existing functionality.

## Backend API Specification

### Base Configuration
- **Base URL**: `http://localhost:8000`
- **CORS**: Pre-configured for `localhost:3000` (update if using different port)
- **Content-Type**: JSON for most endpoints, multipart/form-data for uploads

### Core API Endpoints

#### 1. Upload Documents
```
POST /upload
Content-Type: multipart/form-data
Field Name: 'files' (supports multiple files)

Response:
{
  "message": "X documents uploaded successfully",
  "document_ids": ["doc_id_1", "doc_id_2"]
}
```

#### 2. List Documents
```
GET /documents

Response:
[
  {
    "id": "unique_doc_id",
    "title": "readable_filename.pdf", 
    "summary": "AI generated summary",
    "upload_date": "2024-01-01T12:00:00Z",
    "chunk_count": 15
  }
]
```

#### 3. Delete Document
```
DELETE /documents/{doc_id}

Response:
{
  "message": "Document deleted successfully"
}
```

#### 4. Chat Query
```
POST /chat
Content-Type: application/json

Request:
{
  "message": "What is the main topic?",
  "document_ids": ["optional", "filter", "array"]
}

Response:
{
  "response": "AI generated answer...",
  "sources": [
    {
      "document_title": "filename.pdf",
      "content": "relevant excerpt...",
      "relevance_score": 0.85
    }
  ]
}
```

#### 5. Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "documents_count": 4
}
```

## Required UI Components

### 1. Document Upload Area
- **Drag & Drop**: Support for PDF files
- **File Picker**: Alternative upload method
- **Progress Indicator**: Show upload status
- **Error Handling**: Display upload failures

### 2. Document Management
- **Document List**: Display all uploaded documents
- **Document Info**: Show title, summary, upload date
- **Delete Function**: Remove documents with confirmation
- **Document Count**: Display total number of documents

### 3. Chat Interface
- **Message Input**: Text area for user questions
- **Send Button**: Submit queries to backend
- **Chat History**: Display conversation thread
- **Loading States**: Show when processing queries

### 4. Source Attribution
- **Source Display**: Show document sources for each response
- **Relevance Scores**: Display confidence levels
- **Content Excerpts**: Show relevant text snippets
- **Document Links**: Connect sources to document list

### 5. System Status
- **Health Indicator**: Green dot when backend online
- **Connection Status**: Show API connectivity
- **Error Messages**: Display system errors clearly

## Implementation Requirements

### JavaScript API Integration
```javascript
// Upload files
const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  const response = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};

// Get documents
const getDocuments = async () => {
  const response = await fetch('http://localhost:8000/documents');
  return response.json();
};

// Send chat message
const sendMessage = async (message, documentIds = []) => {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      document_ids: documentIds
    })
  });
  
  return response.json();
};

// Delete document
const deleteDocument = async (docId) => {
  const response = await fetch(`http://localhost:8000/documents/${docId}`, {
    method: 'DELETE'
  });
  
  return response.json();
};

// Check health
const checkHealth = async () => {
  const response = await fetch('http://localhost:8000/health');
  return response.json();
};
```

### Error Handling
- **Network Errors**: Handle connection failures gracefully
- **API Errors**: Display backend error messages to user
- **Validation**: Ensure only PDF files are uploaded
- **Timeouts**: Handle long-running operations

### State Management
- **Document State**: Track uploaded documents
- **Chat State**: Maintain conversation history
- **UI State**: Loading indicators, error states
- **Sync State**: Keep UI synchronized with backend

## Critical Implementation Notes

### File Handling
- Backend expects `multipart/form-data` with field name `files`
- Supports multiple file upload in single request
- Only PDF files are processed
- File size limits handled by backend

### Document IDs
- Always use the `id` field from document list responses
- Never use filenames as identifiers
- Document IDs are required for deletion operations

### Source Attribution
- Display both `document_title` and `content` from sources
- Show `relevance_score` to indicate confidence
- Sources array may be empty for some responses

### CORS Configuration
- Backend pre-configured for `localhost:3000`
- Update backend CORS settings if using different port
- Ensure proper headers for cross-origin requests

## Optional Enhancements

### Voice Input
- Backend supports voice transcription
- Integrate speech-to-text for hands-free operation
- Add microphone button to chat interface

### Advanced Features
- Document filtering in chat queries
- Export functionality for responses
- Real-time upload progress bars
- Document preview capabilities

### Performance Optimizations
- Implement request caching where appropriate
- Add debouncing for search inputs
- Optimize re-renders for large document lists

## Testing Checklist

### Upload Functionality
- [ ] Single file upload works
- [ ] Multiple file upload works
- [ ] Error handling for invalid files
- [ ] Progress indication during upload

### Document Management
- [ ] Document list displays correctly
- [ ] Document deletion works
- [ ] Document count updates properly
- [ ] Metadata displays accurately

### Chat Interface
- [ ] Messages send successfully
- [ ] Responses display with sources
- [ ] Error messages show for failures
- [ ] Loading states work properly

### System Integration
- [ ] Health check indicates status
- [ ] All API endpoints respond correctly
- [ ] Error handling works across all features
- [ ] UI stays synchronized with backend

## Framework Compatibility

This API specification works with any web framework:
- **React**: Use hooks for state management
- **Vue**: Use composition API or options API
- **Angular**: Use services for API calls
- **Vanilla JS**: Direct fetch API usage
- **Other**: Any framework supporting HTTP requests

The backend is completely framework-agnostic and will work seamlessly with any properly implemented frontend.