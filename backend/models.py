"""Data models for Able with GraphRAG integration."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl

class DocumentChunk(BaseModel):
    """Document text chunk."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int

class Document(BaseModel):
    """Document model."""
    id: str
    name: str
    file_path: str
    summary: str
    chunks: List[DocumentChunk]
    created_at: datetime
    file_size: int
    source_type: str = "pdf"  # "pdf" or "web"
    original_url: Optional[str] = None

class DocumentMetadata(BaseModel):
    """Optimized document metadata for storage (no chunk content)."""
    id: str
    name: str
    file_path: str
    summary: str
    created_at: datetime
    file_size: int
    chunk_count: int
    chunk_ids: List[str]
    source_type: str = "pdf"  # "pdf" or "web"
    original_url: Optional[str] = None

class DocumentSummary(BaseModel):
    """Document summary for UI."""
    id: str
    name: str
    summary: str
    created_at: datetime
    file_size: int
    chunk_count: int
    source_type: str = "pdf"
    original_url: Optional[str] = None

class ChatRequest(BaseModel):
    """Chat request model."""
    question: str
    document_ids: Optional[List[str]] = None

class SourceInfo(BaseModel):
    """Source information for chat responses."""
    document_id: str
    document_name: str
    chunk_content: str
    relevance_score: float

class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: List[SourceInfo]
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    documents_count: int
    vector_db_status: str
    timestamp: datetime

class UploadResponse(BaseModel):
    """Upload response model."""
    success: bool
    document: Optional[DocumentSummary] = None
    message: str

class WebScrapingRequest(BaseModel):
    """Web scraping request model."""
    url: HttpUrl
    title: Optional[str] = None

class WebScrapingResponse(BaseModel):
    """Web scraping response model."""
    success: bool
    document: Optional[DocumentSummary] = None
    message: str

# GraphRAG-specific models

class EntityInfo(BaseModel):
    """Knowledge graph entity information."""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    source_document_id: str
    source_document_name: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class RelationshipInfo(BaseModel):
    """Knowledge graph relationship information."""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    description: Optional[str] = None
    weight: float = 1.0
    source_document_id: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class CommunityInfo(BaseModel):
    """Knowledge graph community information."""
    id: str
    title: str
    summary: str
    entities: List[str]
    relationships: List[str]
    level: int
    rank: float

class GraphSearchResult(BaseModel):
    """Result from GraphRAG search."""
    answer: str
    search_type: str  # "global", "local", or "vector"
    context_data: Dict[str, Any] = Field(default_factory=dict)
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    communities: List[CommunityInfo] = Field(default_factory=list)
    completion_time: Optional[float] = None
    llm_calls: Optional[int] = None
    prompt_tokens: Optional[int] = None

class EnhancedChatRequest(BaseModel):
    """Enhanced chat request with GraphRAG support."""
    question: str
    document_ids: Optional[List[str]] = None
    search_type: Optional[str] = None  # "auto", "global", "local", "vector"
    include_entities: bool = True
    include_relationships: bool = True

class EnhancedSourceInfo(BaseModel):
    """Enhanced source information with GraphRAG context."""
    document_id: str
    document_name: str
    chunk_content: str
    relevance_score: float
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)

class EnhancedChatResponse(BaseModel):
    """Enhanced chat response with GraphRAG insights."""
    answer: str
    sources: List[EnhancedSourceInfo]
    timestamp: datetime
    search_type: str
    graph_insights: Optional[GraphSearchResult] = None

class GraphProcessingResult(BaseModel):
    """Result of GraphRAG processing for a document."""
    success: bool
    message: str
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0
    processing_time: Optional[float] = None

class GraphStatistics(BaseModel):
    """Knowledge graph statistics."""
    total_entities: int
    total_relationships: int
    total_communities: int
    graph_nodes: int
    graph_edges: int
    graph_connected_components: int
    graphrag_available: bool
    entity_types: Dict[str, int] = Field(default_factory=dict)

class DocumentWithGraph(BaseModel):
    """Document model with GraphRAG information."""
    document: DocumentSummary
    graph_processing: Optional[GraphProcessingResult] = None
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)

# Model Management Models

class ModelInfo(BaseModel):
    """AI model information."""
    name: str
    display_name: str
    provider: str  # "anthropic" or "ollama"
    available: bool
    type: str  # "remote" or "local"
    size: Optional[int] = None
    modified: Optional[str] = None

class ModelSwitchRequest(BaseModel):
    """Request to switch AI provider/model."""
    provider: str
    model: Optional[str] = None

class ModelDownloadRequest(BaseModel):
    """Request to download an Ollama model."""
    model_name: str

class ModelStatusResponse(BaseModel):
    """AI model status response."""
    current_provider: str
    current_model: str
    available_providers: List[str]
    fallback_enabled: bool
    provider_status: Dict[str, bool]

class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str
    provider: str
    total_requests: int
    avg_response_time: float
    avg_memory_usage: float
    last_used: Optional[str]

class SystemStatusResponse(BaseModel):
    """System status and resource usage."""
    cpu_percent: float
    memory: Dict[str, float]
    disk: Dict[str, float]
    ollama_connection: bool
    anthropic_connection: bool

# MCP (Model Context Protocol) Models

class MCPConfig(BaseModel):
    """MCP configuration for filesystem, git, and sqlite access."""
    filesystem_root: Optional[str] = None
    git_repositories: List[str] = Field(default_factory=list)
    sqlite_connections: List[str] = Field(default_factory=list)
    enabled_tools: List[str] = Field(default_factory=list)
    session_timeout: int = 3600  # 1 hour default

class MCPToolResult(BaseModel):
    """Result from MCP tool execution."""
    tool_name: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None

class MCPSession(BaseModel):
    """MCP session state management."""
    session_id: str
    enabled: bool
    config: Optional[MCPConfig] = None
    available_tools: List[str] = Field(default_factory=list)
    created_at: datetime
    last_activity: datetime
    active_connections: Dict[str, bool] = Field(default_factory=dict)

class MCPStatusResponse(BaseModel):
    """MCP status response."""
    enabled: bool
    session_active: bool
    available_tools: List[str] = Field(default_factory=list)
    filesystem_access: bool = False
    git_access: bool = False
    sqlite_access: bool = False
    session_id: Optional[str] = None
    last_activity: Optional[datetime] = None

class MCPToggleRequest(BaseModel):
    """Request to toggle MCP functionality."""
    enabled: bool
    preserve_session: bool = True

class MCPEnhancedSourceInfo(BaseModel):
    """Enhanced source info with MCP tool results."""
    document_id: str
    document_name: str
    chunk_content: str
    relevance_score: float
    mcp_results: List[MCPToolResult] = Field(default_factory=list)
    entities: List[EntityInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)

class MCPEnhancedChatResponse(BaseModel):
    """Enhanced chat response with MCP integration."""
    answer: str
    sources: List[MCPEnhancedSourceInfo]
    timestamp: datetime
    search_type: str
    mcp_tools_used: List[str] = Field(default_factory=list)
    mcp_results: List[MCPToolResult] = Field(default_factory=list)
    graph_insights: Optional[GraphSearchResult] = None