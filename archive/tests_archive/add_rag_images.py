#!/usr/bin/env python3
"""
RAG Image Enhancement Script

Add reference images to the RAG database to improve tag accuracy.

Usage:
    python scripts/add_rag_images.py --dir ./data/rag_dataset --recursive
    python scripts/add_rag_images.py --file ./reference.jpg --tags "loli,catgirl"
    python scripts/add_rag_images.py --demo
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.clip_image_embedding_service import (
    get_clip_image_service,
    reset_clip_service,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_argparse():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description="Add reference images to RAG database")

    parser.add_argument("--dir", "-d", help="Directory containing images")
    parser.add_argument("--file", "-f", help="Single image file to add")
    parser.add_argument("--tags", "-t", help="Tags for the image (comma-separated)")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search recursively in directories",
    )
    parser.add_argument("--demo", "-m", action="store_true", help="Add demo images")
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear existing collection before adding",
    )
    parser.add_argument(
        "--model", default="openai/clip-vit-large-patch14", help="CLIP model name"
    )
    parser.add_argument(
        "--persist-dir", default="./data/rag_db", help="Directory for RAG database"
    )

    return parser


def add_demo_images(service: CLIPImageEmbeddingService):
    """Add demo reference images."""
    import io
    from PIL import Image

    demo_images = [
        {
            "name": "loli_reference",
            "tags": ["蘿莉", "loli", "少女"],
            "description": "Demo loli reference image",
        },
        {
            "name": "catgirl_reference",
            "tags": ["貓娘", "catgirl", "cat girl"],
            "description": "Demo catgirl reference image",
        },
        {
            "name": "big_breasts_reference",
            "tags": ["巨乳", "big breasts", "large breasts"],
            "description": "Demo big breasts reference image",
        },
    ]

    for demo in demo_images:
        # Create a simple colored image as demo
        # In real usage, you'd use actual reference images
        img = Image.new("RGB", (224, 224), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        try:
            doc_id = service.add_image(
                img_bytes.read(),
                tags=demo["tags"],
                metadata={"description": demo["description"], "is_demo": True},
            )
            logger.info(f"Added demo image: {demo['name']} -> {doc_id}")
        except Exception as e:
            logger.error(f"Failed to add demo image {demo['name']}: {e}")


def main():
    """Main function."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Validate arguments
    if not any([args.dir, args.file, args.demo]):
        parser.error("Must specify --dir, --file, or --demo")

    # Get service
    if args.clear:
        reset_clip_service()

    service = get_clip_image_service(
        model_name=args.model, persist_directory=args.persist_dir
    )

    # Show current stats
    stats = service.get_stats()
    logger.info(f"Current RAG stats: {stats}")

    # Process based on arguments
    if args.clear:
        logger.info("Clearing existing collection...")
        service.delete_collection()

    if args.demo:
        logger.info("Adding demo images...")
        add_demo_images(service)

    if args.file and args.tags:
        logger.info(f"Adding single image: {args.file}")
        with open(args.file, "rb") as f:
            image_bytes = f.read()

        tags = [t.strip() for t in args.tags.split(",")]
        doc_id = service.add_image(image_bytes, tags)
        logger.info(f"Added image: {args.file} -> {doc_id}")

    if args.dir:
        logger.info(f"Adding images from directory: {args.dir}")
        results = service.add_images_from_directory(args.dir, recursive=args.recursive)
        logger.info(f"Added {results['added']} images, {results['failed']} failed")

    # Show final stats
    final_stats = service.get_stats()
    logger.info(f"Final RAG stats: {final_stats}")


if __name__ == "__main__":
    main()
