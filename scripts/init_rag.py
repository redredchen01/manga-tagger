"""Initialize RAG dataset from reference images.

This script builds the initial ChromaDB index from a directory of 
reference manga covers with their associated tags.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tag_library(path: Path) -> Dict[str, Any]:
    """Load tag library from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_reference_images(dataset_path: Path) -> List[Path]:
    """Find all reference images in dataset directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []
    
    for ext in image_extensions:
        images.extend(dataset_path.rglob(f"*{ext}"))
        images.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def load_image_metadata(image_path: Path) -> Dict[str, Any]:
    """Load metadata for an image from accompanying JSON file."""
    json_path = image_path.with_suffix(".json")
    
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Try to infer from directory structure
    # e.g., data/rag_dataset/character_type/tag_name/image.jpg
    parts = image_path.relative_to(image_path.parent.parent.parent).parts
    
    metadata = {
        "source_path": str(image_path),
        "inferred_category": parts[0] if parts else "unknown"
    }
    
    return metadata


async def build_rag_index(dataset_path: Path, rag_service: RAGService):
    """Build RAG index from dataset."""
    logger.info(f"Scanning dataset: {dataset_path}")
    
    image_paths = find_reference_images(dataset_path)
    logger.info(f"Found {len(image_paths)} reference images")
    
    if not image_paths:
        logger.warning("No images found in dataset!")
        return
    
    # Process each image
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            logger.info(f"Processing [{i}/{len(image_paths)}]: {image_path.name}")
            
            # Load image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Load metadata
            metadata = load_image_metadata(image_path)
            
            # Extract tags from metadata or filename
            tags = metadata.get("tags", [])
            
            # If no tags in metadata, try to infer from path
            if not tags:
                # Use parent directory name as tag
                parent_name = image_path.parent.name
                if parent_name and parent_name != dataset_path.name:
                    tags = [parent_name]
            
            if not tags:
                logger.warning(f"  No tags found for {image_path.name}, skipping")
                continue
            
            # Add to RAG
            doc_id = await rag_service.add_image(
                image_bytes=image_bytes,
                tags=tags,
                metadata={
                    "source_path": str(image_path),
                    **metadata
                }
            )
            
            logger.info(f"  Added with ID: {doc_id}, Tags: {tags}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"  Error processing {image_path}: {e}")
            error_count += 1
    
    logger.info(f"\nIndexing complete!")
    logger.info(f"  Successfully indexed: {success_count}")
    logger.info(f"  Errors: {error_count}")
    
    # Show stats
    stats = await rag_service.get_stats()
    logger.info(f"  Total documents in collection: {stats['total_documents']}")


async def add_single_image(
    image_path: Path, 
    tags: List[str], 
    rag_service: RAGService
):
    """Add a single image to RAG."""
    logger.info(f"Adding single image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    doc_id = await rag_service.add_image(
        image_bytes=image_bytes,
        tags=tags,
        metadata={"source_path": str(image_path)}
    )
    
    logger.info(f"Added with ID: {doc_id}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize RAG dataset for Manga Tagger"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/rag_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--single-image",
        type=str,
        help="Add a single image instead of building full index"
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="JSON array of tags for single image (e.g., '[\"tag1\", \"tag2\"]')"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before indexing"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG service
    logger.info("Initializing RAG service...")
    rag_service = RAGService()
    
    # Reset if requested
    if args.reset:
        logger.warning("Resetting collection...")
        rag_service.chroma_client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        rag_service.collection = rag_service.chroma_client.create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection reset complete")
    
    # Run in async context
    import asyncio
    
    if args.single_image:
        # Add single image
        if not args.tags:
            logger.error("--tags required when using --single-image")
            return
        
        tags = json.loads(args.tags)
        asyncio.run(add_single_image(
            Path(args.single_image),
            tags,
            rag_service
        ))
    else:
        # Build full index
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        asyncio.run(build_rag_index(dataset_path, rag_service))


if __name__ == "__main__":
    main()
