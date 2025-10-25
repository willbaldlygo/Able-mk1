"""Document processing service for Able with GraphRAG and multimodal integration."""
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
from services.image_extractor import ImageExtractor
from services.multimodal_service import MultimodalService

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
    """Document processing with GraphRAG and multimodal integration for advanced knowledge synthesis."""

    def __init__(self):
        self.graphrag_service = GraphRAGService()
        self.image_extractor = ImageExtractor()
        self.multimodal_service = MultimodalService()
    
    def process_pdf(self, file_path: Path, original_filename: str, enable_multimodal: bool = True) -> Document:
        """Process PDF into document with chunks and optional multimodal processing."""
        doc_id = str(uuid.uuid4())

        # Extract text
        text_content = self._extract_text(file_path)
        if not text_content.strip():
            raise ValueError("PDF contains no extractable text")

        # Create chunks
        chunks = self._create_chunks(text_content, doc_id)

        # Extract images if multimodal is enabled
        if enable_multimodal:
            try:
                image_results = self.image_extractor.extract_all_images(str(file_path), doc_id)
                if image_results["summary"]["success"]:
                    # Add image information to chunk metadata
                    self._enhance_chunks_with_image_context(chunks, image_results)
                    logger.info(f"Extracted {image_results['summary']['total_images']} images from {original_filename}")
            except Exception as e:
                logger.warning(f"Image extraction failed for {original_filename}: {e}")

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
            # Handle both Document and DocumentMetadata objects
            if hasattr(document, 'chunks') and document.chunks:
                # Full Document object with chunks
                full_content = "\n\n".join([chunk.content for chunk in document.chunks])
            else:
                # DocumentMetadata object - need to extract text from file
                file_path = Path(document.file_path)
                if file_path.exists() and file_path.suffix.lower() == '.pdf':
                    full_content = self._extract_text(file_path)
                else:
                    raise ValueError(f"Cannot process document: file not found or not PDF: {file_path}")
            
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

    def _enhance_chunks_with_image_context(self, chunks: List[DocumentChunk], image_results: Dict[str, Any]) -> None:
        """Enhance document chunks with image context information."""
        try:
            images = image_results.get("images", [])
            if not images:
                return

            # Group images by page number for page-based association
            page_images = {}
            for image in images:
                page_num = image.get("page_number", 1)
                if page_num not in page_images:
                    page_images[page_num] = []
                page_images[page_num].append(image)

            # Add image references to chunks
            # Simple approach: distribute images across chunks based on page distribution
            total_chunks = len(chunks)
            if total_chunks > 0:
                for chunk_idx, chunk in enumerate(chunks):
                    # Estimate which page this chunk belongs to
                    estimated_page = max(1, (chunk_idx * len(page_images)) // total_chunks + 1)

                    # Find images for this estimated page or nearby pages
                    chunk_images = []
                    for page_num in range(max(1, estimated_page - 1), min(len(page_images) + 1, estimated_page + 2)):
                        if page_num in page_images:
                            chunk_images.extend(page_images[page_num])

                    if chunk_images:
                        # Add image metadata to chunk
                        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                            chunk.metadata = {}

                        chunk.metadata["images"] = [
                            {
                                "image_id": img["image_id"],
                                "file_path": img["file_path"],
                                "image_type": img["image_type"],
                                "page_number": img.get("page_number"),
                                "width": img.get("width"),
                                "height": img.get("height")
                            }
                            for img in chunk_images[:2]  # Limit to 2 images per chunk
                        ]

        except Exception as e:
            logger.error(f"Error enhancing chunks with image context: {e}")

    async def process_pdf_with_multimodal_analysis(self, file_path: Path, original_filename: str) -> Tuple[Document, Dict[str, Any]]:
        """
        Process PDF with chunked multimodal analysis.
        """
        doc_id = str(uuid.uuid4())

        # Extract text
        text_content = self._extract_text(file_path)
        if not text_content.strip():
            raise ValueError("PDF contains no extractable text")

        # Create chunks
        chunks = self._create_chunks(text_content, doc_id)

        # Chunked image processing
        multimodal_results = await self._process_images_chunked(file_path, doc_id, original_filename)

        # Add visual context to chunks if available
        if multimodal_results.get("images"):
            visual_context = self.multimodal_service.create_visual_context_summary(multimodal_results["images"])
            if visual_context:
                for chunk in chunks[:3]:
                    if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata["visual_context"] = visual_context

        # Generate summary
        summary = self._generate_summary(chunks[:3])

        document = Document(
            id=doc_id,
            name=original_filename,
            file_path=str(file_path),
            summary=summary,
            chunks=chunks,
            created_at=datetime.now(),
            file_size=file_path.stat().st_size,
            source_type="pdf"
        )

        return document, multimodal_results

    async def _process_images_chunked(self, file_path: Path, doc_id: str, filename: str) -> Dict[str, Any]:
        """Process images in chunks to prevent memory issues."""
        multimodal_results = {"images": [], "summary": {}}
        
        try:
            # Extract images in batches of 5 pages
            page_batch_size = 5
            pdf_doc = fitz.open(file_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            all_images = []
            processed_count = 0
            failed_count = 0
            
            for start_page in range(0, total_pages, page_batch_size):
                end_page = min(start_page + page_batch_size, total_pages)
                logger.info(f"Processing pages {start_page+1}-{end_page} of {filename}")
                
                # Extract images for this batch
                batch_images = self.image_extractor.extract_images_from_pdf_pages_range(
                    str(file_path), doc_id, start_page, end_page
                )
                
                if batch_images:
                    # Process with LLava in smaller batches
                    image_paths = [img["file_path"] for img in batch_images]
                    batch_analysis = await self.multimodal_service.process_images_batch(
                        image_paths[:3], doc_id  # Limit to 3 images per batch
                    )
                    
                    all_images.extend(batch_analysis)
                    processed_count += len([img for img in batch_analysis if img.get("visual_description")])
                    failed_count += len([img for img in batch_analysis if not img.get("visual_description")])
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            multimodal_results["images"] = all_images
            multimodal_results["summary"] = {
                "total_images": len(all_images),
                "completed_analysis": processed_count,
                "failed": failed_count,
                "success": len(all_images) > 0
            }
            
            logger.info(f"Chunked processing complete for {filename}: {processed_count} images analyzed")
            
        except Exception as e:
            logger.error(f"Chunked image processing failed for {filename}: {e}")
            multimodal_results["error"] = str(e)
        
        return multimodal_results

    async def process_image_file(self, image_data: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process uploaded image file.

        Args:
            image_data: Raw image file data
            filename: Original filename

        Returns:
            Tuple of (document_id, processing_results)
        """
        doc_id = str(uuid.uuid4())

        try:
            # Extract and save image
            image_info = self.image_extractor.extract_images_from_direct_upload(image_data, filename, doc_id)

            # Analyze image with multimodal service
            analysis_result = await self.multimodal_service.process_image(
                image_info["file_path"],
                doc_id,
                chunk_index=0
            )

            return doc_id, {
                "image_info": image_info,
                "analysis": analysis_result,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error processing image file {filename}: {e}")
            return doc_id, {
                "error": str(e),
                "success": False
            }

    def get_document_images(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all images associated with a document.

        Args:
            document_id: Document ID

        Returns:
            List of image information dictionaries
        """
        # This would typically query a database in a production system
        # For now, we'll scan the images directory
        images = []
        try:
            pattern = f"{document_id}_*"
            for image_file in self.image_extractor.images_dir.glob(pattern):
                image_info = self.image_extractor.get_image_info(str(image_file))
                if image_info:
                    images.append(image_info)
        except Exception as e:
            logger.error(f"Error getting images for document {document_id}: {e}")

        return images

    def cleanup_document_images(self, document_id: str) -> int:
        """
        Clean up images associated with a document.

        Args:
            document_id: Document ID

        Returns:
            Number of images deleted
        """
        return self.image_extractor.cleanup_document_images(document_id)