"""
PDF image extraction service for Able mk I.
Extracts images from PDF documents using pdf2image and PyMuPDF.
"""
import io
import logging
import os
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Service for extracting images from PDF documents."""

    def __init__(self, images_dir: Optional[str] = None):
        if images_dir is None:
            from config import config
            self.images_dir = config.sources_dir / "images"
        else:
            self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg'}

        # PDF image extraction settings
        self.dpi = 150  # Balance between quality and file size
        self.min_image_size = 50  # Minimum width/height in pixels
        self.max_image_size = 2000  # Maximum width/height to prevent memory issues

    def extract_images_from_pdf_pages_range(self, pdf_path: str, document_id: str, start_page: int = 0, end_page: int = None) -> List[Dict[str, Any]]:
        """Extract images from a specific range of PDF pages."""
        extracted_images = []

        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='PNG',
                first_page=start_page + 1,
                last_page=end_page
            )

            for page_offset, page_image in enumerate(images):
                page_num = start_page + page_offset + 1
                try:
                    image_filename = f"{document_id}_page_{page_num}_{uuid.uuid4().hex[:8]}.png"
                    image_path = self.images_dir / image_filename

                    if (page_image.width > self.max_image_size or
                        page_image.height > self.max_image_size):
                        ratio = min(
                            self.max_image_size / page_image.width,
                            self.max_image_size / page_image.height
                        )
                        new_size = (
                            int(page_image.width * ratio),
                            int(page_image.height * ratio)
                        )
                        page_image = page_image.resize(new_size, Image.Resampling.LANCZOS)

                    page_image.save(image_path, "PNG", optimize=True)

                    image_info = {
                        "image_id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "page_number": page_num,
                        "image_type": "pdf_page",
                        "file_path": str(image_path),
                        "filename": image_filename,
                        "width": page_image.width,
                        "height": page_image.height,
                        "file_size": os.path.getsize(image_path),
                        "extraction_method": "pdf2image"
                    }

                    extracted_images.append(image_info)

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error converting PDF pages {start_page}-{end_page}: {e}")

        return extracted_images

    def extract_images_from_pdf_pages(self, pdf_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Extract images by converting PDF pages to images using pdf2image."""
        extracted_images = []

        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='PNG',
                first_page=None,
                last_page=None
            )

            for page_num, page_image in enumerate(images, 1):
                try:
                    # Generate unique filename
                    image_filename = f"{document_id}_page_{page_num}_{uuid.uuid4().hex[:8]}.png"
                    image_path = self.images_dir / image_filename

                    # Resize if too large
                    if (page_image.width > self.max_image_size or
                        page_image.height > self.max_image_size):

                        # Calculate proportional resize
                        ratio = min(
                            self.max_image_size / page_image.width,
                            self.max_image_size / page_image.height
                        )
                        new_size = (
                            int(page_image.width * ratio),
                            int(page_image.height * ratio)
                        )
                        page_image = page_image.resize(new_size, Image.Resampling.LANCZOS)

                    # Save the page image
                    page_image.save(image_path, "PNG", optimize=True)

                    image_info = {
                        "image_id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "page_number": page_num,
                        "image_type": "pdf_page",
                        "file_path": str(image_path),
                        "filename": image_filename,
                        "width": page_image.width,
                        "height": page_image.height,
                        "file_size": os.path.getsize(image_path),
                        "extraction_method": "pdf2image"
                    }

                    extracted_images.append(image_info)
                    logger.info(f"Extracted page {page_num} as image: {image_filename}")

                except Exception as e:
                    logger.error(f"Error processing page {page_num} of {pdf_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error converting PDF pages to images for {pdf_path}: {e}")

        return extracted_images

    def extract_embedded_images_from_pdf(self, pdf_path: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Extract embedded images from PDF using PyMuPDF.
        This extracts actual image objects embedded in the PDF.

        Args:
            pdf_path: Path to the PDF file
            document_id: ID of the document

        Returns:
            List of extracted embedded image information
        """
        extracted_images = []

        try:
            pdf_document = fitz.open(pdf_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        # Skip if image is too small
                        if (pix.width < self.min_image_size or
                            pix.height < self.min_image_size):
                            pix = None
                            continue

                        # Convert to PNG if not already
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None

                        # Generate unique filename
                        image_filename = f"{document_id}_embedded_{page_num+1}_{img_index}_{uuid.uuid4().hex[:8]}.png"
                        image_path = self.images_dir / image_filename

                        # Save the image
                        with open(image_path, "wb") as img_file:
                            img_file.write(img_data)

                        image_info = {
                            "image_id": str(uuid.uuid4()),
                            "document_id": document_id,
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "image_type": "embedded",
                            "file_path": str(image_path),
                            "filename": image_filename,
                            "width": pix.width,
                            "height": pix.height,
                            "file_size": os.path.getsize(image_path),
                            "extraction_method": "pymupdf"
                        }

                        extracted_images.append(image_info)
                        logger.info(f"Extracted embedded image from page {page_num+1}: {image_filename}")

                        pix = None  # Free memory

                    except Exception as e:
                        logger.error(f"Error extracting embedded image {img_index} from page {page_num+1}: {e}")
                        continue

            pdf_document.close()

        except Exception as e:
            logger.error(f"Error extracting embedded images from {pdf_path}: {e}")

        return extracted_images

    def extract_images_from_direct_upload(self, image_file_data: bytes, filename: str, document_id: str) -> Dict[str, Any]:
        """
        Process directly uploaded image files.

        Args:
            image_file_data: Raw image file data
            filename: Original filename
            document_id: ID of the document

        Returns:
            Image information dictionary
        """
        try:
            # Validate file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {file_ext}")

            # Open and validate image
            image = Image.open(io.BytesIO(image_file_data))

            # Skip if image is too small
            if (image.width < self.min_image_size or
                image.height < self.min_image_size):
                raise ValueError(f"Image too small: {image.width}x{image.height}")

            # Resize if too large
            if (image.width > self.max_image_size or
                image.height > self.max_image_size):

                ratio = min(
                    self.max_image_size / image.width,
                    self.max_image_size / image.height
                )
                new_size = (
                    int(image.width * ratio),
                    int(image.height * ratio)
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Generate unique filename
            image_filename = f"{document_id}_upload_{uuid.uuid4().hex[:8]}{file_ext}"
            image_path = self.images_dir / image_filename

            # Save the processed image
            image.save(image_path, format=image.format or 'PNG', optimize=True)

            image_info = {
                "image_id": str(uuid.uuid4()),
                "document_id": document_id,
                "image_type": "direct_upload",
                "file_path": str(image_path),
                "filename": image_filename,
                "original_filename": filename,
                "width": image.width,
                "height": image.height,
                "file_size": os.path.getsize(image_path),
                "extraction_method": "direct_upload"
            }

            logger.info(f"Processed uploaded image: {image_filename}")
            return image_info

        except Exception as e:
            logger.error(f"Error processing uploaded image {filename}: {e}")
            raise

    def extract_all_images(self, pdf_path: str, document_id: str, extract_pages: bool = True, extract_embedded: bool = True) -> Dict[str, Any]:
        """
        Extract all images from a PDF using both page conversion and embedded extraction.

        Args:
            pdf_path: Path to the PDF file
            document_id: ID of the document
            extract_pages: Whether to extract page images
            extract_embedded: Whether to extract embedded images

        Returns:
            Dictionary containing all extracted image information
        """
        all_images = []
        extraction_summary = {
            "total_images": 0,
            "page_images": 0,
            "embedded_images": 0,
            "extraction_methods": [],
            "success": False,
            "errors": []
        }

        try:
            # Extract page images
            if extract_pages:
                try:
                    page_images = self.extract_images_from_pdf_pages(pdf_path, document_id)
                    all_images.extend(page_images)
                    extraction_summary["page_images"] = len(page_images)
                    if page_images:
                        extraction_summary["extraction_methods"].append("pdf2image")
                except Exception as e:
                    error_msg = f"Page extraction failed: {e}"
                    logger.error(error_msg)
                    extraction_summary["errors"].append(error_msg)

            # Extract embedded images
            if extract_embedded:
                try:
                    embedded_images = self.extract_embedded_images_from_pdf(pdf_path, document_id)
                    all_images.extend(embedded_images)
                    extraction_summary["embedded_images"] = len(embedded_images)
                    if embedded_images:
                        extraction_summary["extraction_methods"].append("pymupdf")
                except Exception as e:
                    error_msg = f"Embedded extraction failed: {e}"
                    logger.error(error_msg)
                    extraction_summary["errors"].append(error_msg)

            extraction_summary["total_images"] = len(all_images)
            extraction_summary["success"] = len(all_images) > 0

        except Exception as e:
            error_msg = f"Image extraction failed: {e}"
            logger.error(error_msg)
            extraction_summary["errors"].append(error_msg)

        return {
            "images": all_images,
            "summary": extraction_summary
        }

    def cleanup_document_images(self, document_id: str) -> int:
        """
        Remove all images associated with a document.

        Args:
            document_id: ID of the document

        Returns:
            Number of images deleted
        """
        deleted_count = 0

        try:
            # Find all images for this document
            pattern = f"{document_id}_*"
            for image_file in self.images_dir.glob(pattern):
                try:
                    image_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted image: {image_file.name}")
                except Exception as e:
                    logger.error(f"Error deleting image {image_file}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up images for document {document_id}: {e}")

        return deleted_count

    def get_image_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an existing image file.

        Args:
            image_path: Path to the image file

        Returns:
            Image information dictionary or None if file not found
        """
        try:
            if not os.path.exists(image_path):
                return None

            with Image.open(image_path) as img:
                return {
                    "file_path": image_path,
                    "filename": os.path.basename(image_path),
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": os.path.getsize(image_path)
                }

        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {e}")
            return None