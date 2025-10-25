"""
Multimodal processing service for Able mk I.
Handles image analysis using ollama/llava integration.
"""
import asyncio
import base64
import io
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class MultimodalService:
    """Service for processing images and visual content using llava."""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model_name = "llava:latest"
        from config import config
        self.images_dir = config.sources_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    async def _check_llava_availability(self) -> bool:
        """Check if llava model is available in ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if "llava" in model.get("name", "").lower():
                        return True
            return False
        except Exception as e:
            logger.warning(f"Could not check llava availability: {e}")
            return False

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for llava processing."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def _process_image_with_pil(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract basic image metadata using PIL."""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_bytes": os.path.getsize(image_path)
                }
        except Exception as e:
            logger.error(f"Error processing image with PIL {image_path}: {e}")
            return None

    async def _analyze_image_with_llava(self, image_path: str, prompt: str = None) -> Optional[str]:
        """Analyze image using llava model."""
        if not await self._check_llava_availability():
            logger.warning("llava model not available, skipping image analysis")
            return None

        try:
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return None

            if not prompt:
                prompt = """Describe this image in detail. Focus on:
1. Main content and subjects
2. Text if any (OCR)
3. Visual elements like charts, diagrams, tables
4. Context and setting
5. Important details that would help with document understanding"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"llava analysis failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error analyzing image with llava: {e}")
            return None

    async def process_image(self, image_path: str, document_id: str, chunk_index: int = 0) -> Dict[str, Any]:
        """
        Process a single image with both PIL metadata and llava analysis.

        Args:
            image_path: Path to the image file
            document_id: ID of the parent document
            chunk_index: Index of the chunk this image belongs to

        Returns:
            Dictionary with image processing results
        """
        result = {
            "image_id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "file_path": image_path,
            "processed_at": datetime.now(),
            "metadata": {},
            "visual_description": None,
            "processing_status": "pending"
        }

        try:
            # Get basic image metadata
            metadata = self._process_image_with_pil(image_path)
            if metadata:
                result["metadata"] = metadata

            # Attempt llava analysis
            visual_description = await self._analyze_image_with_llava(image_path)
            if visual_description:
                result["visual_description"] = visual_description
                result["processing_status"] = "completed"
            else:
                result["processing_status"] = "metadata_only"

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            result["processing_status"] = "failed"
            result["error"] = str(e)

        return result

    async def process_images_batch(self, image_paths: List[str], document_id: str) -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel.

        Args:
            image_paths: List of image file paths
            document_id: ID of the parent document

        Returns:
            List of image processing results
        """
        tasks = []
        for i, image_path in enumerate(image_paths):
            task = self.process_image(image_path, document_id, chunk_index=i)
            tasks.append(task)

        # Process up to 3 images concurrently to avoid overwhelming the system
        semaphore = asyncio.Semaphore(3)

        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        limited_tasks = [process_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing image {image_paths[i]}: {result}")
                processed_results.append({
                    "image_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": i,
                    "file_path": image_paths[i],
                    "processed_at": datetime.now(),
                    "processing_status": "failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    def save_image(self, image_data: bytes, filename: str) -> str:
        """
        Save image data to the images directory.

        Args:
            image_data: Raw image bytes
            filename: Original filename

        Returns:
            Path to saved image file
        """
        # Generate unique filename
        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.png', '.jpg', '.jpeg']:
            file_ext = '.png'  # Default to PNG

        unique_filename = f"{uuid.uuid4()}{file_ext}"
        image_path = self.images_dir / unique_filename

        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Saved image to {image_path}")
            return str(image_path)
        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")
            raise

    def get_image_processing_summary(self, image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of image processing results.

        Args:
            image_results: List of image processing results

        Returns:
            Summary dictionary
        """
        total_images = len(image_results)
        completed = sum(1 for r in image_results if r.get("processing_status") == "completed")
        metadata_only = sum(1 for r in image_results if r.get("processing_status") == "metadata_only")
        failed = sum(1 for r in image_results if r.get("processing_status") == "failed")

        return {
            "total_images": total_images,
            "completed_analysis": completed,
            "metadata_only": metadata_only,
            "failed": failed,
            "success_rate": (completed + metadata_only) / total_images if total_images > 0 else 0,
            "has_visual_descriptions": completed > 0
        }

    async def reprocess_image(self, image_id: str, image_path: str, document_id: str) -> Dict[str, Any]:
        """
        Reprocess a single image (useful for retry logic).

        Args:
            image_id: Existing image ID
            image_path: Path to the image file
            document_id: ID of the parent document

        Returns:
            Updated image processing result
        """
        result = await self.process_image(image_path, document_id)
        result["image_id"] = image_id  # Preserve existing ID
        return result

    def create_visual_context_summary(self, image_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Create a summary of visual context from processed images.

        Args:
            image_results: List of processed image results

        Returns:
            Combined visual context summary
        """
        descriptions = []
        for result in image_results:
            if result.get("visual_description"):
                descriptions.append(result["visual_description"])

        if not descriptions:
            return None

        if len(descriptions) == 1:
            return descriptions[0]

        # Combine multiple descriptions
        combined = f"This document contains {len(descriptions)} images:\n\n"
        for i, desc in enumerate(descriptions, 1):
            combined += f"Image {i}: {desc}\n\n"

        return combined.strip()