"""Document processing service for Able with GraphRAG integration."""
import asyncio
import uuid
import fitz  # PyMuPDF
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import config
from models import Document, DocumentChunk
from services.graphrag_service import GraphRAGService

logger = logging.getLogger(__name__)

def _tokens(s: str) -> List[str]:
    return s.split()  # simple proxy; swap to a proper tokenizer later

def _windows(n: int, size: int, overlap: int):
    i = 0
    while i < n:
        yield i, min(n, i + size)
        i += max(1, size - overlap)

def make_child_parent_chunks(text: str, child_size: int = 700, child_overlap: int = 100, parent_size: int = 2000) -> List[Dict[str, Any]]:
    toks = _tokens(text)
    N = len(toks)
    out = []
    for s, e in _windows(N, child_size, child_overlap):
        child = " ".join(toks[s:e])
        c = (s + e) // 2
        ph = parent_size // 2
        ps = max(0, c - ph)
        pe = min(N, c + ph)
        out.append({
            "text": child,
            "parent_text": " ".join(toks[ps:pe]),
            "parent_span": (ps, pe)
        })
    return out

class DocumentService:
    """Document processing with GraphRAG integration for advanced knowledge synthesis."""
    
    def __init__(self):
        self.graphrag_service = GraphRAGService()
    
    def process_pdf(self, file_path: Path, original_filename: str) -> Document:
        """Process PDF into document with chunks."""
        doc_id = str(uuid.uuid4())
        
        # Extract text
        text_content = self._extract_text(file_path)
        if not text_content.strip():
            raise ValueError("PDF contains no extractable text")
        
        # Create chunks
        chunks = self._create_chunks(text_content, doc_id)
        
        # Generate summary
        summary = self._generate_summary(chunks[:3])
        
        return Document(
            id=doc_id,
            name=original_filename,
            file_path=str(file_path),
            summary=summary,
            chunks=chunks,
            created_at=datetime.now(),
            file_size=file_path.stat().st_size,
            source_type="pdf"
        )
    
    def process_web_content(self, file_path: Path, title: str, content: str, original_url: str) -> Document:
        """Process web content into document with chunks."""
        doc_id = str(uuid.uuid4())
        
        # Validate content
        if not content.strip():
            raise ValueError("Web content is empty")
        
        # Create chunks from content
        chunks = self._create_chunks(content, doc_id)
        
        # Generate summary
        summary = self._generate_summary(chunks[:3])
        
        return Document(
            id=doc_id,
            name=title,
            file_path=str(file_path),
            summary=summary,
            chunks=chunks,
            created_at=datetime.now(),
            file_size=file_path.stat().st_size,
            source_type="web",
            original_url=original_url
        )
    
    def process_text_content(self, file_path: Path, title: str, text_content: str, original_url: str) -> Document:
        """Process text file content into document with chunks."""
        doc_id = str(uuid.uuid4())
        
        # Extract clean content from the text file (skip header)
        content = self._extract_content_from_text_file(text_content)
        
        # Validate content
        if not content.strip():
            raise ValueError("Text content is empty")
        
        # Create chunks from content
        chunks = self._create_chunks(content, doc_id)
        
        # Generate summary
        summary = self._generate_summary(chunks[:3])
        
        return Document(
            id=doc_id,
            name=title,
            file_path=str(file_path),
            summary=summary,
            chunks=chunks,
            created_at=datetime.now(),
            file_size=file_path.stat().st_size,
            source_type="web",
            original_url=original_url
        )
    
    def validate_pdf(self, file_path: Path) -> bool:
        """Validate PDF file."""
        try:
            doc = fitz.open(file_path)
            is_valid = doc.page_count > 0
            doc.close()
            return is_valid
        except Exception:
            return False
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF."""
        try:
            doc = fitz.open(file_path)
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text()
                text_content += "\n\n"
            
            doc.close()
            return text_content.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _create_chunks(self, text: str, doc_id: str) -> List[DocumentChunk]:
        """Create child/parent chunks with enhanced context."""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Use child/parent chunking
        child_parent_chunks = make_child_parent_chunks(
            text,
            child_size=700,
            child_overlap=100,
            parent_size=2000
        )
        
        for chunk_index, cp_chunk in enumerate(child_parent_chunks):
            chunk_content = cp_chunk["text"]
            parent_text = cp_chunk["parent_text"]
            
            # Calculate character positions (approximate)
            start_char = text.find(chunk_content[:50]) if len(chunk_content) > 50 else text.find(chunk_content)
            if start_char == -1:
                start_char = chunk_index * 500  # fallback
            end_char = start_char + len(chunk_content)
            
            chunks.append(DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata={"parent_text": parent_text}
            ))
        
        return chunks
    
    def _generate_summary(self, chunks: List[DocumentChunk]) -> str:
        """Generate simple summary from first chunks."""
        if not chunks:
            return "No content available"
        
        # Combine first chunks
        combined_text = " ".join([chunk.content for chunk in chunks])
        
        # Extract first sentences
        sentences = combined_text.split('. ')
        summary_sentences = sentences[:3]
        
        summary = '. '.join(summary_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        # Limit length
        if len(summary) > 300:
            summary = summary[:297] + "..."
        
        return summary or "Document content extracted successfully"
    
    def _extract_content_from_text_file(self, text_content: str) -> str:
        """Extract main content from text file, skipping header metadata."""
        lines = text_content.split('\n')
        
        # Find the end of the header (marked by === or empty line after header)
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '===================' or (
                i > 0 and line.strip() == '' and 
                any('===' in prev_line for prev_line in lines[:i])
            ):
                content_start = i + 1
                break
        
        # Extract content after header
        content_lines = lines[content_start:]
        return '\n'.join(content_lines).strip()
    
    async def process_document_with_graph(self, document: Document) -> Dict[str, Any]:
        """Process document for GraphRAG knowledge graph extraction."""
        try:
            # Combine all chunks into full text content
            full_content = "\n\n".join([chunk.content for chunk in document.chunks])
            
            # Process with GraphRAG
            graph_result = await self.graphrag_service.process_document_for_graph(
                document_id=document.id,
                content=full_content,
                document_name=document.name
            )
            
            if graph_result["success"]:
                logger.info(f"GraphRAG processing completed for {document.name}: "
                           f"{graph_result.get('entity_count', 0)} entities, "
                           f"{graph_result.get('relationship_count', 0)} relationships")
            else:
                logger.warning(f"GraphRAG processing failed for {document.name}: {graph_result.get('message', 'Unknown error')}")
            
            return graph_result
        
        except Exception as e:
            logger.error(f"Error in GraphRAG processing for {document.name}: {e}")
            return {"success": False, "message": str(e)}
    
    def get_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        """Get entities extracted from a specific document."""
        return self.graphrag_service.get_document_entities(document_id)
    
    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity."""
        return self.graphrag_service.get_entity_relationships(entity_name)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return self.graphrag_service.get_graph_statistics()
    
    async def delete_document_from_graph(self, document_id: str) -> Dict[str, Any]:
        """Remove document from knowledge graph."""
        return await self.graphrag_service.delete_document_from_graph(document_id)
    
    def is_graphrag_available(self) -> bool:
        """Check if GraphRAG is available."""
        return self.graphrag_service.is_available()