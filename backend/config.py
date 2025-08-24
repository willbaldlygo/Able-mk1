"""Configuration management for Able."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class Config:
    """Clean configuration management."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        # Load .env file from project root
        env_path = self.base_dir / '.env'
        load_dotenv(env_path)
        self.load_environment()
    
    def load_environment(self):
        """Load configuration from environment variables."""
        # Core settings
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        # Paths
        self.project_root = self.base_dir
        self.sources_dir = self.base_dir / "data" / "sources"
        self.vectordb_dir = self.base_dir / "data" / "vectordb"
        self.graphrag_dir = self.base_dir / "data" / "graphrag"
        self.metadata_file = self.base_dir / "data" / "metadata.json"
        
        # Create directories
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.vectordb_dir.mkdir(parents=True, exist_ok=True)
        self.graphrag_dir.mkdir(parents=True, exist_ok=True)
        
        # AI settings
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        # Ollama settings
        self.ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.default_provider = os.getenv("DEFAULT_AI_PROVIDER", "anthropic")  # "anthropic" or "ollama"
        self.fallback_enabled = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
        
        # Search settings
        self.search_results = int(os.getenv("SEARCH_RESULTS", "8"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "600"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        
        # GraphRAG settings
        self.graphrag_enabled = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"
        self.graphrag_entity_types = os.getenv("GRAPHRAG_ENTITY_TYPES", "PERSON,ORGANIZATION,LOCATION,EVENT,CONCEPT,TECHNOLOGY,METHOD").split(",")
        self.graphrag_max_gleanings = int(os.getenv("GRAPHRAG_MAX_GLEANINGS", "1"))
        self.graphrag_community_max_length = int(os.getenv("GRAPHRAG_COMMUNITY_MAX_LENGTH", "1500"))
        self.graphrag_chunk_size = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "1200"))
        self.graphrag_chunk_overlap = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "100"))
        
        # Server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

config = Config()