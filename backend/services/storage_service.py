"""File storage service for Able."""
import json
import shutil
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from config import config
from models import Document, DocumentSummary, DocumentMetadata

class StorageService:
    """Clean file and metadata storage."""
    
    def __init__(self):
        self.sources_dir = config.sources_dir
        self.metadata_file = config.metadata_file
        self._ensure_metadata_exists()
    
    def _ensure_metadata_exists(self):
        """Ensure metadata file exists."""
        if not self.metadata_file.exists():
            self.metadata_file.write_text("{}")
    
    def save_file(self, file_content: bytes, filename: str) -> Path:
        """Save uploaded file with clean filename."""
        clean_filename = self._sanitize_filename(filename)
        file_path = self.sources_dir / clean_filename
        
        # Handle duplicates
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = self.sources_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        file_path.write_bytes(file_content)
        return file_path
    
    def delete_file(self, file_path: Path) -> bool:
        """Delete file from storage."""
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def save_metadata(self, document: Document) -> bool:
        """Save optimized document metadata (no chunk content)."""
        try:
            metadata = self._load_metadata()
            
            # Create optimized metadata without chunk content
            optimized_metadata = DocumentMetadata(
                id=document.id,
                name=document.name,
                file_path=document.file_path,
                summary=document.summary,
                created_at=document.created_at,
                file_size=document.file_size,
                chunk_count=len(document.chunks),
                chunk_ids=[chunk.id for chunk in document.chunks],
                source_type=document.source_type,
                original_url=document.original_url
            )
            
            metadata[document.id] = optimized_metadata.dict()
            self.metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
            return True
        except Exception:
            return False
    
    def load_metadata(self) -> Dict[str, DocumentMetadata]:
        """Load optimized document metadata."""
        try:
            metadata = self._load_metadata()
            documents = {}
            for doc_id, doc_data in metadata.items():
                try:
                    # Handle datetime strings
                    if isinstance(doc_data.get('created_at'), str):
                        doc_data['created_at'] = datetime.fromisoformat(doc_data['created_at'])
                    
                    # Handle legacy format (with chunks) vs new format (without chunks)
                    if 'chunks' in doc_data and 'chunk_ids' not in doc_data:
                        # Legacy format - extract chunk IDs from chunks
                        chunk_ids = [chunk['id'] for chunk in doc_data['chunks']]
                        chunk_count = len(doc_data['chunks'])
                        doc_metadata = DocumentMetadata(
                            id=doc_data['id'],
                            name=doc_data['name'],
                            file_path=doc_data['file_path'],
                            summary=doc_data['summary'],
                            created_at=doc_data['created_at'],
                            file_size=doc_data['file_size'],
                            chunk_count=chunk_count,
                            chunk_ids=chunk_ids,
                            source_type=doc_data.get('source_type', 'pdf'),
                            original_url=doc_data.get('original_url')
                        )
                    else:
                        # New optimized format
                        doc_metadata = DocumentMetadata(**doc_data)
                    
                    documents[doc_id] = doc_metadata
                except Exception:
                    continue  # Skip invalid entries
            return documents
        except Exception:
            return {}
    
    def delete_metadata(self, document_id: str) -> bool:
        """Delete document metadata."""
        try:
            metadata = self._load_metadata()
            if document_id in metadata:
                del metadata[document_id]
                self.metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
                return True
            return False
        except Exception:
            return False
    
    def get_document_summaries(self) -> List[DocumentSummary]:
        """Get all document summaries."""
        documents = self.load_metadata()
        summaries = []
        for doc in documents.values():
            if Path(doc.file_path).exists():  # Only include existing files
                summaries.append(DocumentSummary(
                    id=doc.id,
                    name=doc.name,
                    summary=doc.summary,
                    created_at=doc.created_at,
                    file_size=doc.file_size,
                    chunk_count=doc.chunk_count,
                    source_type=doc.source_type,
                    original_url=doc.original_url
                ))
        return sorted(summaries, key=lambda x: x.created_at, reverse=True)
    
    def _load_metadata(self) -> dict:
        """Load raw metadata from file."""
        try:
            return json.loads(self.metadata_file.read_text())
        except Exception:
            return {}
    
    def _sanitize_filename(self, filename: str) -> str:
        """Create clean, readable filename."""
        # Remove path components
        clean_name = Path(filename).name
        stem = Path(clean_name).stem
        suffix = Path(clean_name).suffix
        
        # Replace problematic characters
        clean_stem = re.sub(r'[<>:"/\\|?*]', '_', stem)
        clean_stem = re.sub(r'[^\w\s\-_\.]', '_', clean_stem)
        clean_stem = re.sub(r'_+', '_', clean_stem)
        clean_stem = re.sub(r'\s+', '_', clean_stem)
        clean_stem = clean_stem.strip('_')
        
        # Limit length
        if len(clean_stem) > 200:
            clean_stem = clean_stem[:200]
        
        return f"{clean_stem}{suffix}"