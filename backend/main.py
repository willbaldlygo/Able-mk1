"""Able FastAPI application with GraphRAG integration."""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from models import (
    ChatRequest, ChatResponse, DocumentSummary, HealthResponse,
    UploadResponse, WebScrapingRequest, WebScrapingResponse,
    # GraphRAG models
    EnhancedChatRequest, EnhancedChatResponse, GraphStatistics,
    DocumentWithGraph, GraphProcessingResult,
    # Model Management models
    ModelInfo, ModelSwitchRequest, ModelDownloadRequest,
    ModelStatusResponse, ModelPerformanceMetrics, SystemStatusResponse,
    # MCP models
    MCPConfig, MCPStatusResponse, MCPToggleRequest, MCPSession,
    MCPToolResult, MCPEnhancedChatResponse
)
from services.storage_service import StorageService
from services.document_service import DocumentService
from services.vector_service import VectorService
from services.ai_service import AIService
from services.transcription_service import TranscriptionService
from services.simple_web_library_service import SimpleWebLibraryService
from services.hybrid_search_service import HybridSearchService
from services.model_service import model_service
from services.staged_reasoning_service import StagedReasoningService
from services.mcp_integration_service import MCPIntegrationService
from services.mcp_service import mcp_manager
from lexical_index import BM25Index
from retrieval import HybridRetriever
from prompt_budget import plan_budget, pack_context
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Able - Advanced PDF Research Assistant",
    description="Intelligent PDF research assistant with GraphRAG knowledge synthesis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
storage_service = StorageService()
document_service = DocumentService()
vector_service = VectorService()
ai_service = AIService()
transcription_service = TranscriptionService()
web_library_service = SimpleWebLibraryService()
hybrid_search_service = HybridSearchService()
staged_reasoning_service = StagedReasoningService()
mcp_integration_service = MCPIntegrationService()
bm25_index = BM25Index()
hybrid_retriever = None

# MCP session management - use actual manager
current_mcp_session_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize application."""
    try:
        # Test AI provider connections
        connection_results = ai_service.test_connection()
        for provider, status in connection_results.items():
            status_msg = "connected" if status else "failed"
            logger.info(f"AI provider {provider}: {status_msg}")
            
        # Check model service status
        model_status = model_service.test_connection()
        if model_status:
            available_models = model_service.get_available_models()
            total_models = sum(len(models) for models in available_models.values())
            logger.info(f"Model service: {total_models} models available across all providers")
        
        # Load existing documents
        documents = storage_service.load_metadata()
        logger.info(f"Loaded {len(documents)} documents on startup")
        
        # Build BM25 index from existing chunks
        global hybrid_retriever
        try:
            # Get all chunks from vector database
            all_results = vector_service.collection.get(include=["documents", "ids", "metadatas"])
            if all_results['documents']:
                records = []
                for doc, doc_id in zip(all_results['documents'], all_results['ids']):
                    records.append({"id": doc_id, "text": doc})
                bm25_index.build(records)
                hybrid_retriever = HybridRetriever(vector_service, bm25_index)
                logger.info(f"Built BM25 index with {len(records)} chunks")
            else:
                logger.info("No existing chunks found for BM25 index")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
        
        logger.info("Able with Ollama integration startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Able - Advanced PDF Research Assistant with GraphRAG", "status": "running"}

@app.post("/shutdown")
async def shutdown_services():
    """Shutdown all Able services cleanly."""
    import os
    import signal
    from datetime import datetime
    
    try:
        # Log shutdown
        shutdown_time = datetime.now().isoformat()
        logger.info(f"Shutdown requested at {shutdown_time}")
        
        # Create simple session log
        session_log = {
            "shutdown_time": shutdown_time,
            "documents_processed": len(storage_service.load_metadata()),
            "status": "clean_shutdown"
        }
        
        # Save session log
        log_file = Path("session_shutdown.log")
        with open(log_file, "a") as f:
            f.write(f"{shutdown_time}: Clean shutdown - {session_log['documents_processed']} documents\n")
        
        # Schedule shutdown after response
        def shutdown_after_response():
            import time
            time.sleep(1)  # Give time for response to send
            os.kill(os.getpid(), signal.SIGTERM)
        
        import threading
        threading.Thread(target=shutdown_after_response, daemon=True).start()
        
        return {"message": "Shutting down Able services...", "session_log": session_log}
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")
        raise HTTPException(status_code=500, detail="Shutdown failed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vector_stats = vector_service.get_stats()
        documents = storage_service.get_document_summaries()
        
        return HealthResponse(
            status="healthy" if vector_stats["status"] == "healthy" else "degraded",
            documents_count=len(documents),
            vector_db_status=vector_stats["status"],
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            documents_count=0,
            vector_db_status="error",
            timestamp=datetime.now()
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents."""
    try:
        if not files:
            return UploadResponse(
                success=False,
                message="No files provided"
            )
        
        # Validate all files first
        for file in files:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                return UploadResponse(
                    success=False,
                    message=f"Only PDF files are supported: {file.filename}"
                )
            
            if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
                return UploadResponse(
                    success=False,
                    message=f"File size exceeds 50MB limit: {file.filename}"
                )
        
        uploaded_docs = []
        
        # Process each file
        for file in files:
            # Read file content
            file_content = await file.read()
            
            # Atomic upload process
            try:
                # 1. Save file
                file_path = storage_service.save_file(file_content, file.filename)
                
                # 2. Validate PDF
                if not document_service.validate_pdf(file_path):
                    storage_service.delete_file(file_path)
                    return UploadResponse(
                        success=False,
                        message=f"Invalid or corrupted PDF file: {file.filename}"
                    )
                
                # 3. Process document
                document = document_service.process_pdf(file_path, file.filename)
                
                # 4. Add to vector database
                if not vector_service.add_document(document):
                    storage_service.delete_file(file_path)
                    return UploadResponse(
                        success=False,
                        message=f"Failed to process document for search: {file.filename}"
                    )
                
                # 5. Save metadata
                if not storage_service.save_metadata(document):
                    storage_service.delete_file(file_path)
                    vector_service.delete_document(document.id)
                    return UploadResponse(
                        success=False,
                        message=f"Failed to save document metadata: {file.filename}"
                    )
                
                # 6. Process with GraphRAG (asynchronous, non-blocking)
                graph_result = None
                if document_service.is_graphrag_available():
                    try:
                        graph_result = await document_service.process_document_with_graph(document)
                    except Exception as e:
                        logger.warning(f"GraphRAG processing failed for {file.filename}: {e}")
                
                # Success - create summary
                summary = DocumentSummary(
                    id=document.id,
                    name=document.name,
                    summary=document.summary,
                    created_at=document.created_at,
                    file_size=document.file_size,
                    chunk_count=len(document.chunks)
                )
                
                uploaded_docs.append(summary)
                
                success_message = f"Successfully uploaded {file.filename}"
                if graph_result and graph_result.get("success"):
                    success_message += f" (GraphRAG: {graph_result.get('entity_count', 0)} entities, {graph_result.get('relationship_count', 0)} relationships)"
                
                logger.info(success_message)
                
            except Exception as e:
                # Cleanup on any failure
                try:
                    if 'file_path' in locals():
                        storage_service.delete_file(file_path)
                    if 'document' in locals():
                        vector_service.delete_document(document.id)
                except:
                    pass
                raise e
        
        # Return success with first document (for compatibility)
        return UploadResponse(
            success=True,
            document=uploaded_docs[0] if uploaded_docs else None,
            message=f"Successfully uploaded {len(uploaded_docs)} document{'s' if len(uploaded_docs) != 1 else ''}"
        )
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return UploadResponse(
            success=False,
            message=f"Upload failed: {str(e)}"
        )

@app.get("/documents", response_model=List[DocumentSummary])
async def get_documents():
    """Get all uploaded documents."""
    try:
        return storage_service.get_document_summaries()
    except Exception as e:
        logger.error(f"Get documents error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document."""
    try:
        # Load document metadata
        documents = storage_service.load_metadata()
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = documents[doc_id]
        
        # Clean deletion with error tracking
        errors = []
        
        # 1. Delete from vector database
        if not vector_service.delete_document(doc_id):
            errors.append("vector database")
        
        # 2. Delete file
        if not storage_service.delete_file(Path(document.file_path)):
            errors.append("file system")
        
        # 3. Delete metadata
        if not storage_service.delete_metadata(doc_id):
            errors.append("metadata")
            
        if errors:
            logger.warning(f"Partial deletion of {doc_id}: failed to delete from {', '.join(errors)}")
            return {"message": f"Document partially deleted - errors with: {', '.join(errors)}", "document_id": doc_id}
        else:
            logger.info(f"Successfully deleted document: {doc_id}")
            return {"message": "Document deleted successfully", "document_id": doc_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(e)}"
        )

def is_document_library_question(question: str) -> bool:
    """Check if the question is asking about the document library itself."""
    question_lower = question.lower().strip()
    
    # Specific phrases that clearly indicate library questions
    library_phrases = [
        'what documents do you have',
        'which documents do you have',
        'list documents',
        'show documents',
        'list all documents',
        'show all documents',
        'what files do you have',
        'which files do you have',
        'list files',
        'show files',
        'document library',
        'file library',
        'uploaded documents',
        'available documents',
        'how many documents',
        'how many files',
        'document titles',
        'file names'
    ]
    
    # Check for exact phrase matches
    for phrase in library_phrases:
        if phrase in question_lower:
            return True
    
    # Check for very specific patterns only
    if question_lower.startswith(('list ', 'show ', 'display ')) and ('document' in question_lower or 'file' in question_lower):
        return True
        
    return False

def generate_document_library_response() -> ChatResponse:
    """Generate a response listing all documents in the library."""
    try:
        # Get all documents from storage
        documents = list(storage_service.load_metadata().values())
        
        if not documents:
            answer = "The document library is currently empty. No documents have been uploaded yet."
        else:
            # Create a formatted list of documents
            doc_list = []
            for i, doc in enumerate(documents, 1):
                upload_date = doc.created_at.strftime("%Y-%m-%d") if doc.created_at else "Unknown"
                pages = f"{len(doc.chunks)} chunks" if hasattr(doc, 'chunks') and doc.chunks else "Unknown chunks"
                doc_list.append(f"{i}. **{doc.name}** ({pages}, uploaded {upload_date})")
            
            answer = f"Here are all the documents currently in your library:\n\n" + "\n".join(doc_list)
            answer += f"\n\n**Total: {len(documents)} document{'s' if len(documents) != 1 else ''}**"
        
        return ChatResponse(
            answer=answer,
            sources=[],
            timestamp=datetime.now()
        )
    except Exception as e:
        return ChatResponse(
            answer="I encountered an error while retrieving the document library information. Please try again.",
            sources=[],
            timestamp=datetime.now()
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with enhanced search and document diversity."""
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Check if this is a question about the document library itself
        if is_document_library_question(request.question):
            logger.info(f"Document library query detected: '{request.question[:50]}...'")
            return generate_document_library_response()
        
        # Strategic search with query analysis
        sources = vector_service.strategic_search(
            query=request.question,
            document_ids=request.document_ids
        )
        
        if not sources:
            return ChatResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you have uploaded relevant PDF documents.",
                sources=[],
                timestamp=datetime.now()
            )
        
        # Generate response using legacy method
        response = ai_service.generate_response_with_sources(request.question, sources)
        
        logger.info(f"Chat query processed: '{request.question[:50]}...' -> {len(sources)} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

# GraphRAG-Enhanced Endpoints

@app.post("/chat/enhanced", response_model=EnhancedChatResponse)
async def enhanced_chat_with_documents(request: EnhancedChatRequest):
    """Enhanced chat with GraphRAG intelligent search routing and optional MCP integration."""
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Use hybrid search system for intelligent query routing
        response = await hybrid_search_service.intelligent_search(request)
        
        # Check if MCP is enabled and should be integrated
        global current_mcp_session_id
        if current_mcp_session_id:
            session_status = mcp_manager.get_session_status(current_mcp_session_id)
            if session_status and session_status["active"] and session_status["servers"]:
                try:
                    # Add MCP tool context to the response if tools are available
                    # Note: This is a placeholder for actual MCP tool execution
                    # In a full implementation, you would execute MCP tools based on the query
                    logger.info(f"MCP tools available for enhanced chat: {len(session_status['servers'])} servers")
                    
                except Exception as e:
                    logger.warning(f"MCP integration failed for enhanced chat: {e}")
                    # Continue without MCP integration
        
        logger.info(f"Enhanced chat query processed: '{request.question[:50]}...' using {response.search_type} search")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced chat failed: {str(e)}"
        )

@app.post("/chat/enhanced/mcp", response_model=MCPEnhancedChatResponse)
async def mcp_enhanced_chat_with_documents(request: EnhancedChatRequest):
    """Enhanced chat with full MCP tool integration."""
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Check if MCP is enabled
        global current_mcp_session_id
        if not current_mcp_session_id:
            raise HTTPException(
                status_code=400,
                detail="MCP is not enabled. Enable MCP first via /mcp/toggle"
            )
        
        session_status = mcp_manager.get_session_status(current_mcp_session_id)
        if not session_status or not session_status["active"]:
            raise HTTPException(
                status_code=400,
                detail="MCP session is not active"
            )
        
        # Get standard enhanced response first
        response = await hybrid_search_service.intelligent_search(request)
        
        # Placeholder for MCP tool execution
        # In a full implementation, this would:
        # 1. Analyze the query to determine which MCP tools to use
        # 2. Execute the appropriate tools
        # 3. Format the results
        mcp_results = []  # List[MCPToolResult]
        mcp_tools_used = []
        
        # Example: If query mentions files or code, execute filesystem tools
        query_lower = request.question.lower()
        if any(keyword in query_lower for keyword in ["file", "code", "directory", "folder"]):
            # Placeholder for filesystem tool execution
            logger.info("Query suggests filesystem access - would execute filesystem tools")
            mcp_tools_used.append("filesystem_read")
        
        # Enhanced sources with MCP results
        enhanced_sources = mcp_integration_service.enhance_sources_with_mcp(
            [source for source in response.sources],
            mcp_results
        )
        
        # Create MCP-enhanced response
        mcp_response = MCPEnhancedChatResponse(
            answer=response.answer,
            sources=enhanced_sources,
            timestamp=response.timestamp,
            search_type=response.search_type,
            mcp_tools_used=mcp_tools_used,
            mcp_results=mcp_results,
            graph_insights=response.graph_insights
        )
        
        logger.info(f"MCP-enhanced chat query processed: '{request.question[:50]}...' with {len(mcp_tools_used)} MCP tools")
        return mcp_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP-enhanced chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP-enhanced chat failed: {str(e)}"
        )

@app.post("/chat/staged")
async def staged_reasoning_chat(request: ChatRequest):
    """Chat with staged reasoning approach for comprehensive responses with optional MCP integration."""
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Check if MCP is enabled for staged reasoning enhancement
        global current_mcp_session_id
        mcp_context = ""
        if current_mcp_session_id:
            session_status = mcp_manager.get_session_status(current_mcp_session_id)
            if session_status and session_status["active"] and session_status["servers"]:
                try:
                    # Note: This is a placeholder for actual MCP tool execution
                    logger.info(f"MCP tools available for staged reasoning: {len(session_status['servers'])} servers")
                except Exception as e:
                    logger.warning(f"MCP integration failed for staged reasoning: {e}")
        
        # Use staged reasoning for comprehensive response
        # Note: The staged reasoning service would need to be enhanced to accept MCP context
        response = await staged_reasoning_service.generate_staged_response(
            question=request.question,
            document_ids=request.document_ids
        )
        
        logger.info(f"Staged reasoning query processed: '{request.question[:50]}...' with {len(response['sources'])} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Staged reasoning error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Staged reasoning failed: {str(e)}"
        )

@app.get("/graph/statistics", response_model=GraphStatistics)
async def get_graph_statistics():
    """Get knowledge graph statistics."""
    try:
        stats = document_service.get_graph_statistics()
        return GraphStatistics(**stats)
    except Exception as e:
        logger.error(f"Graph statistics error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph statistics: {str(e)}"
        )

@app.get("/documents/{doc_id}/entities")
async def get_document_entities(doc_id: str):
    """Get entities extracted from a specific document."""
    try:
        entities = document_service.get_document_entities(doc_id)
        return {"document_id": doc_id, "entities": entities, "count": len(entities)}
    except Exception as e:
        logger.error(f"Get document entities error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document entities: {str(e)}"
        )

@app.get("/entities/{entity_name}/relationships")
async def get_entity_relationships(entity_name: str):
    """Get relationships for a specific entity."""
    try:
        relationships = document_service.get_entity_relationships(entity_name)
        return {"entity": entity_name, "relationships": relationships, "count": len(relationships)}
    except Exception as e:
        logger.error(f"Get entity relationships error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get entity relationships: {str(e)}"
        )

@app.get("/search/capabilities")
async def get_search_capabilities():
    """Get information about available search capabilities."""
    try:
        capabilities = hybrid_search_service.get_search_statistics()
        return capabilities
    except Exception as e:
        logger.error(f"Search capabilities error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get search capabilities: {str(e)}"
        )

@app.post("/documents/{doc_id}/reprocess-graph")
async def reprocess_document_graph(doc_id: str):
    """Reprocess a document for GraphRAG extraction."""
    try:
        # Load document
        documents = storage_service.load_metadata()
        if doc_id not in documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        document = documents[doc_id]
        
        # Reprocess with GraphRAG
        graph_result = await document_service.process_document_with_graph(document)
        
        return {
            "document_id": doc_id,
            "reprocessing_result": graph_result,
            "message": "Document reprocessed for GraphRAG extraction"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reprocess document error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reprocess document: {str(e)}"
        )

@app.post("/admin/cleanup-vector-database")
async def cleanup_vector_database():
    """Clean up orphaned documents in vector database."""
    try:
        # Get valid document IDs (only for documents with existing files)
        summaries = storage_service.get_document_summaries()
        valid_ids = [doc.id for doc in summaries]
        
        # Cleanup vector database
        cleanup_result = vector_service.cleanup_orphaned_documents(valid_ids)
        
        logger.info(f"Vector database cleanup completed: {cleanup_result}")
        return {
            "success": True,
            "cleanup_result": cleanup_result,
            "valid_documents": len(valid_ids),
            "message": f"Cleaned up {cleanup_result.get('deleted', 0)} orphaned documents"
        }
        
    except Exception as e:
        logger.error(f"Vector database cleanup error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Only audio files are supported"
            )
        
        # Read audio data
        audio_data = await file.read()
        
        # Transcribe audio
        transcribed_text = transcription_service.transcribe_audio(audio_data)
        
        if transcribed_text is None:
            raise HTTPException(
                status_code=500,
                detail="Transcription failed"
            )
        
        logger.info(f"Audio transcribed successfully: {len(transcribed_text)} characters")
        return {"text": transcribed_text}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/scrape-url", response_model=WebScrapingResponse)
async def scrape_url(request: WebScrapingRequest):
    """Scrape URL and create document."""
    try:
        url = str(request.url)
        custom_title = request.title
        
        # Validate URL
        if not web_library_service.validate_url(url):
            return WebScrapingResponse(
                success=False,
                message="URL is not accessible or invalid"
            )
        
        # Check if content is scrapable
        if not web_library_service.is_scrapable_content(url):
            return WebScrapingResponse(
                success=False,
                message="URL does not contain scrapable content (HTML)"
            )
        
        # Atomic scraping process
        try:
            # 1. Scrape content
            title, text_content, text_filename = web_library_service.scrape_url(url, custom_title)
            
            if not title or not text_content or not text_filename:
                return WebScrapingResponse(
                    success=False,
                    message="Failed to extract content from URL"
                )
            
            # 2. Save text file
            file_path = storage_service.save_file(text_content.encode('utf-8'), text_filename)
            
            # 3. Process document
            document = document_service.process_text_content(
                file_path, title, text_content, url
            )
            
            # 4. Add to vector database
            if not vector_service.add_document(document):
                storage_service.delete_file(file_path)
                return WebScrapingResponse(
                    success=False,
                    message="Failed to process content for search"
                )
            
            # 5. Save metadata
            if not storage_service.save_metadata(document):
                storage_service.delete_file(file_path)
                vector_service.delete_document(document.id)
                return WebScrapingResponse(
                    success=False,
                    message="Failed to save document metadata"
                )
            
            # Success - create summary
            summary = DocumentSummary(
                id=document.id,
                name=document.name,
                summary=document.summary,
                created_at=document.created_at,
                file_size=document.file_size,
                chunk_count=len(document.chunks),
                source_type=document.source_type,
                original_url=document.original_url
            )
            
            logger.info(f"Successfully scraped URL: {url}")
            return WebScrapingResponse(
                success=True,
                document=summary,
                message=f"Successfully scraped and processed: {title}"
            )
            
        except Exception as e:
            # Cleanup on any failure
            try:
                if 'file_path' in locals():
                    storage_service.delete_file(file_path)
                if 'document' in locals():
                    vector_service.delete_document(document.id)
            except:
                pass
            raise e
            
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        return WebScrapingResponse(
            success=False,
            message=f"Scraping failed: {str(e)}"
        )

# Model Management Endpoints

@app.get("/models/available")
async def get_available_models():
    """Get list of available AI models from all providers."""
    try:
        models = model_service.get_available_models()
        
        # Flatten the structure for the API response
        all_models = []
        for provider, provider_models in models.items():
            for model in provider_models:
                all_models.append(ModelInfo(**model))
        
        return {
            "models": all_models,
            "providers": models,
            "total_count": len(all_models)
        }
        
    except Exception as e:
        logger.error(f"Get available models error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available models: {str(e)}"
        )

@app.get("/models/current")
async def get_current_model():
    """Get the currently active AI model."""
    try:
        status = await model_service.get_model_status()
        return {
            "model": {
                "name": status["current"]["model"],
                "display_name": status["current"]["model"].replace(':', ' ').replace('-', ' ').title(),
                "provider": status["current"]["provider"],
                "type": "local" if status["current"]["provider"] == "ollama" else "remote",
                "status": "online" if status["current"]["active"] else "offline",
                "available": status["current"]["active"]
            }
        }
    except Exception as e:
        logger.error(f"Get current model error: {str(e)}")
        # Return a fallback model
        return {
            "model": {
                "name": "claude-3-5-sonnet-20241022",
                "display_name": "Claude 3.5 Sonnet",
                "provider": "anthropic",
                "type": "remote",
                "status": "online",
                "available": True
            }
        }

@app.post("/models/switch")
async def switch_ai_provider(request: ModelSwitchRequest):
    """Switch AI provider and/or model."""
    try:
        success = ai_service.switch_provider(request.provider, request.model)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to provider: {request.provider}"
            )
        
        # Get updated status
        status = await get_model_status()
        
        logger.info(f"Switched AI provider to {request.provider}" + 
                   (f" with model {request.model}" if request.model else ""))
        
        return {
            "success": True,
            "message": f"Successfully switched to {request.provider}",
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Switch provider error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch provider: {str(e)}"
        )

@app.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get current AI provider and model status."""
    try:
        available_providers = ai_service.get_available_providers()
        connection_status = ai_service.test_connection()
        
        return ModelStatusResponse(
            current_provider=ai_service.default_provider.value,
            current_model=(ai_service.claude_model if ai_service.default_provider.value == "anthropic" 
                          else ai_service.ollama_model),
            available_providers=available_providers,
            fallback_enabled=ai_service.fallback_enabled,
            provider_status=connection_status
        )
        
    except Exception as e:
        logger.error(f"Get model status error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )

@app.post("/models/download")
async def download_ollama_model(request: ModelDownloadRequest):
    """Download an Ollama model."""
    try:
        if not config.ollama_enabled:
            raise HTTPException(
                status_code=400,
                detail="Ollama is not enabled"
            )
        
        result = model_service.download_model(request.model_name)
        
        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result['message']
            )
        
        logger.info(f"Downloaded Ollama model: {request.model_name}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download model error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download model: {str(e)}"
        )

@app.delete("/models/{model_name}")
async def delete_ollama_model(model_name: str):
    """Delete an Ollama model."""
    try:
        if not config.ollama_enabled:
            raise HTTPException(
                status_code=400,
                detail="Ollama is not enabled"
            )
        
        result = model_service.delete_model(model_name)
        
        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result['message']
            )
        
        logger.info(f"Deleted Ollama model: {model_name}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete model error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )

@app.get("/models/performance")
async def get_model_performance():
    """Get performance metrics for all models."""
    try:
        metrics = model_service.get_performance_metrics()
        
        performance_list = []
        for model_name, data in metrics.items():
            # Determine provider from model name
            provider = "ollama" if ":" in model_name or model_name in [m['name'] for m in model_service.get_available_models().get('ollama', [])] else "anthropic"
            
            performance_list.append(ModelPerformanceMetrics(
                model_name=model_name,
                provider=provider,
                total_requests=data['total_requests'],
                avg_response_time=data['avg_response_time'],
                avg_memory_usage=data['avg_memory_usage'],
                last_used=data['last_used']
            ))
        
        return {
            "performance_metrics": performance_list,
            "total_models_used": len(performance_list)
        }
        
    except Exception as e:
        logger.error(f"Get model performance error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model performance: {str(e)}"
        )

@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and resource usage."""
    try:
        status = model_service.get_system_status()
        
        if 'error' in status:
            raise HTTPException(
                status_code=500,
                detail=status['error']
            )
        
        # Test connections
        anthropic_status = model_service.test_anthropic_connection() if hasattr(model_service, 'test_anthropic_connection') else False
        ollama_status = model_service.test_connection()
        
        return SystemStatusResponse(
            cpu_percent=status['cpu_percent'],
            memory=status['memory'],
            disk=status['disk'],
            ollama_connection=ollama_status,
            anthropic_connection=anthropic_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get system status error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )

# MCP (Model Context Protocol) Endpoints

@app.get("/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status():
    """Get current MCP session status."""
    try:
        global current_mcp_session_id
        
        if not current_mcp_session_id:
            return MCPStatusResponse(
                enabled=False,
                session_active=False,
                available_tools=[],
                filesystem_access=False,
                git_access=False,
                sqlite_access=False,
                session_id=None,
                last_activity=None
            )
        
        # Get session status from manager
        session_status = mcp_manager.get_session_status(current_mcp_session_id)
        
        if not session_status:
            current_mcp_session_id = None  # Clean up invalid session
            return MCPStatusResponse(
                enabled=False,
                session_active=False,
                available_tools=[],
                filesystem_access=False,
                git_access=False,
                sqlite_access=False,
                session_id=None,
                last_activity=None
            )
        
        servers = session_status.get("servers", {})
        available_tools = []
        
        # Build available tools list from active servers
        if "filesystem" in servers and servers["filesystem"]["running"]:
            available_tools.extend(["filesystem_read", "filesystem_write", "filesystem_list"])
        if "git" in servers and servers["git"]["running"]:
            available_tools.extend(["git_status", "git_log", "git_diff", "git_show"])
        if "sqlite" in servers and servers["sqlite"]["running"]:
            available_tools.extend(["sqlite_query", "sqlite_schema", "sqlite_list_tables"])
        
        return MCPStatusResponse(
            enabled=session_status["active"],
            session_active=session_status["active"],
            available_tools=available_tools,
            filesystem_access="filesystem" in servers and servers["filesystem"]["running"],
            git_access="git" in servers and servers["git"]["running"],
            sqlite_access="sqlite" in servers and servers["sqlite"]["running"],
            session_id=current_mcp_session_id,
            last_activity=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"MCP status error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get MCP status: {str(e)}"
        )

@app.post("/mcp/toggle")
async def toggle_mcp(request: MCPToggleRequest):
    """Enable or disable MCP for current session."""
    try:
        global current_mcp_session_id
        
        if request.enabled:
            # Enable MCP - create new session if needed
            if not current_mcp_session_id or not request.preserve_session:
                current_mcp_session_id = mcp_manager.create_session()
            
            logger.info(f"MCP enabled for session {current_mcp_session_id}")
        else:
            # Disable MCP
            if current_mcp_session_id:
                await mcp_manager.stop_session(current_mcp_session_id)
                current_mcp_session_id = None
            
            logger.info("MCP disabled")
        
        return {
            "success": True,
            "enabled": request.enabled,
            "session_id": current_mcp_session_id,
            "message": f"MCP {'enabled' if request.enabled else 'disabled'} successfully"
        }
        
    except Exception as e:
        logger.error(f"MCP toggle error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle MCP: {str(e)}"
        )

@app.post("/mcp/config")
async def configure_mcp(config: MCPConfig):
    """Set MCP configuration for filesystem, git, and sqlite access."""
    try:
        global current_mcp_session_id
        
        if not current_mcp_session_id:
            raise HTTPException(
                status_code=400,
                detail="MCP session not active. Enable MCP first via /mcp/toggle"
            )
        
        # Start servers based on configuration
        started_servers = []
        
        if config.filesystem_root:
            success = await mcp_manager.start_server(
                current_mcp_session_id, 
                "filesystem", 
                {"root_path": config.filesystem_root}
            )
            if success:
                started_servers.append("filesystem")
        
        if config.git_repositories:
            success = await mcp_manager.start_server(
                current_mcp_session_id,
                "git",
                {"allowed_repos": config.git_repositories}
            )
            if success:
                started_servers.append("git")
        
        if config.sqlite_connections:
            success = await mcp_manager.start_server(
                current_mcp_session_id,
                "sqlite",
                {"allowed_dbs": config.sqlite_connections}
            )
            if success:
                started_servers.append("sqlite")
        
        logger.info(f"MCP configured with {len(started_servers)} servers: {started_servers}")
        
        return {
            "success": True,
            "started_servers": started_servers,
            "message": f"MCP configured successfully with {len(started_servers)} servers"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP config error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure MCP: {str(e)}"
        )

@app.get("/mcp/tools")
async def get_available_mcp_tools():
    """Get list of available MCP tools based on current configuration."""
    try:
        global current_mcp_session_id
        
        if not current_mcp_session_id:
            return {
                "available": False,
                "tools": [],
                "message": "MCP session not active"
            }
        
        # Get session status
        session_status = mcp_manager.get_session_status(current_mcp_session_id)
        if not session_status or not session_status["active"]:
            return {
                "available": False,
                "tools": [],
                "message": "MCP session not active"
            }
        
        # Get tool details with descriptions
        tool_details = {
            # Filesystem tools
            "filesystem_read": {
                "name": "filesystem_read",
                "category": "filesystem",
                "description": "Read file contents from configured root directory",
                "parameters": ["file_path"]
            },
            "filesystem_write": {
                "name": "filesystem_write", 
                "category": "filesystem",
                "description": "Write content to files in configured root directory",
                "parameters": ["file_path", "content"]
            },
            "filesystem_list": {
                "name": "filesystem_list",
                "category": "filesystem", 
                "description": "List directory contents within configured root",
                "parameters": ["directory_path"]
            },
            # Git tools
            "git_status": {
                "name": "git_status",
                "category": "git",
                "description": "Get git repository status",
                "parameters": ["repository_path"]
            },
            "git_log": {
                "name": "git_log",
                "category": "git",
                "description": "Get git commit history",
                "parameters": ["repository_path", "limit"]
            },
            "git_diff": {
                "name": "git_diff",
                "category": "git",
                "description": "Get git diff information",
                "parameters": ["repository_path", "commit_ref"]
            },
            "git_show": {
                "name": "git_show",
                "category": "git", 
                "description": "Show specific git commit or file",
                "parameters": ["repository_path", "object_ref"]
            },
            # SQLite tools
            "sqlite_query": {
                "name": "sqlite_query",
                "category": "sqlite",
                "description": "Execute SQL query on configured database",
                "parameters": ["database", "query"]
            },
            "sqlite_schema": {
                "name": "sqlite_schema",
                "category": "sqlite",
                "description": "Get database schema information",
                "parameters": ["database"]
            },
            "sqlite_list_tables": {
                "name": "sqlite_list_tables",
                "category": "sqlite",
                "description": "List tables in configured database",
                "parameters": ["database"]
            }
        }
        
        # Build available tools from active servers
        servers = session_status.get("servers", {})
        available_tools = []
        
        if "filesystem" in servers and servers["filesystem"]["running"]:
            available_tools.extend(["filesystem_read", "filesystem_write", "filesystem_list"])
        if "git" in servers and servers["git"]["running"]:
            available_tools.extend(["git_status", "git_log", "git_diff", "git_show"])
        if "sqlite" in servers and servers["sqlite"]["running"]:
            available_tools.extend(["sqlite_query", "sqlite_schema", "sqlite_list_tables"])
        
        # Get details for available tools
        available_tool_details = [
            tool_details[tool_name] 
            for tool_name in available_tools 
            if tool_name in tool_details
        ]
        
        return {
            "available": True,
            "session_id": current_mcp_session_id,
            "tools": available_tool_details,
            "tool_count": len(available_tool_details),
            "categories": list(set(tool["category"] for tool in available_tool_details)),
            "active_servers": list(servers.keys())
        }
        
    except Exception as e:
        logger.error(f"Get MCP tools error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get MCP tools: {str(e)}"
        )

@app.get("/chat/debug")
async def debug_retrieval(query: str, include_mcp: bool = False):
    """Debug endpoint for hybrid retrieval analysis with optional MCP info."""
    try:
        debug_info = {}
        
        # Standard hybrid retrieval debug info
        if hybrid_retriever:
            # Get candidates using hybrid retrieval
            candidates = hybrid_retriever.get_candidates(query)
            
            # Load env config for budget analysis
            frac_ans = float(os.getenv("PROMPT_FRACTION_ANSWER", 0.30))
            frac_instr = float(os.getenv("PROMPT_FRACTION_INSTRUCTIONS", 0.10))
            max_ctx = int(os.getenv("MODEL_MAX_TOKENS", 8192))
            
            # Analyze prompt budget
            system_instructions = "You are a helpful research assistant."
            budget = plan_budget(query, system_instructions, max_ctx, frac_ans, frac_instr)
            
            # Pack context
            context_blob, used_tokens = pack_context(candidates, budget["context_budget"])
            
            debug_info.update({
                "query": query,
                "candidates_found": len(candidates),
                "budget_analysis": budget,
                "context_tokens_used": used_tokens,
                "context_preview": context_blob[:500] + "..." if len(context_blob) > 500 else context_blob,
                "candidates_detail": [
                    {
                        "id": c["id"],
                        "doc_id": c["meta"].get("doc_id", "unknown"),
                        "vec_score": c.get("vec_score", 0),
                        "rerank_score": c.get("rerank_score", 0),
                        "text_preview": c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"]
                    }
                    for c in candidates[:10]  # Show top 10
                ]
            })
        else:
            debug_info["error"] = "Hybrid retriever not initialized"
        
        # Add MCP debug info if requested
        if include_mcp:
            global current_mcp_session_id
            if current_mcp_session_id:
                session_status = mcp_manager.get_session_status(current_mcp_session_id)
                debug_info["mcp_info"] = {
                    "session_active": bool(session_status and session_status["active"]),
                    "session_id": current_mcp_session_id,
                    "active_servers": list(session_status["servers"].keys()) if session_status else [],
                    "server_count": len(session_status["servers"]) if session_status else 0
                }
            else:
                debug_info["mcp_info"] = {
                    "session_active": False,
                    "session_id": None,
                    "active_servers": [],
                    "server_count": 0
                }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug retrieval error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)