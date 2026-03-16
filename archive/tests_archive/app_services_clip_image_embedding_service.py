"""CLIP Image Embedding Service for RAG.

Uses OpenAI CLIP model for generating image embeddings.
Provides better image similarity search than hash-based fallbacks.
"""

import io
import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)


class CLIPImageEmbeddingService:
    """CLIP-based image embedding service for RAG similarity search."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        persist_directory: Optional[str] = None,
        collection_name: str = "rag_images",
    ):
        """
        Initialize CLIP embedding service.

        Args:
            model_name: HuggingFace CLIP model name
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name
        self.collection_name = collection_name

        # Determine storage path
        self.persist_directory = persist_directory or getattr(
            settings, "RAG_PERSIST_DIR", "./data/rag_db"
        )

        # Initialize CLIP model
        self._init_clip_model()

        # Initialize ChromaDB
        self._init_chromadb()

        logger.info(f"CLIP Image Embedding Service initialized: {model_name}")

    def _init_clip_model(self):
        """Initialize CLIP model for image embeddings."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            logger.info(f"Loading CLIP model: {self.model_name}")

            # Load CLIP model
            self.clip_model = CLIPModel.from_pretrained(self.model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)
            self.clip_model.eval()

            logger.info("CLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            # Initialize client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

            count = self.collection.count()
            logger.info(f"ChromaDB collection '{self.collection_name}': {count} images")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _prepare_image(self, image_bytes: bytes) -> Image.Image:
        """Prepare image for embedding."""
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def generate_embedding(self, image_bytes: bytes) -> List[float]:
        """
        Generate CLIP embedding for image.

        Args:
            image_bytes: Raw image bytes

        Returns:
            List of floats representing the embedding vector
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            from PIL import Image

            # Prepare image
            image = self._prepare_image(image_bytes)

            # Process with CLIP
            inputs = self.clip_processor(
                images=image, return_tensors="pt", do_rescale=False
            )

            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            # Normalize and convert to list
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding_list = embedding.squeeze().tolist()

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def add_image(
        self, image_bytes: bytes, tags: List[str], metadata: Optional[dict] = None
    ) -> str:
        """
        Add image to RAG dataset.

        Args:
            image_bytes: Image bytes
            tags: List of tags for this image
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        import uuid

        try:
            # Generate embedding
            embedding = self.generate_embedding(image_bytes)

            # Generate unique ID
            doc_id = f"img_{uuid.uuid4().hex[:12]}"

            # Prepare metadata
            doc_metadata = {
                "tags": ",".join(tags),
                "tag_count": len(tags),
                **(metadata or {}),
            }

            # Add to collection
            self.collection.add(
                ids=[doc_id], embeddings=[embedding], metadatas=[doc_metadata]
            )

            logger.info(f"Added image to RAG: {doc_id} with tags: {tags}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            raise

    def add_image_from_path(
        self, image_path: str, tags: List[str], metadata: Optional[dict] = None
    ) -> str:
        """
        Add image from file path.

        Args:
            image_path: Path to image file
            tags: List of tags for this image
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Get filename for metadata
        filename = Path(image_path).stem

        if metadata is None:
            metadata = {}

        metadata["source_file"] = image_path
        metadata["filename"] = filename

        return self.add_image(image_bytes, tags, metadata)

    def add_images_from_directory(
        self,
        directory: str,
        recursive: bool = False,
        allowed_extensions: tuple = (".jpg", ".jpeg", ".png"),
    ) -> dict:
        """
        Add all images from a directory.

        Args:
            directory: Directory path containing images
            recursive: Whether to search subdirectories
            allowed_extensions: Tuple of allowed file extensions

        Returns:
            Dict with 'added' and 'failed' counts
        """
        import os
        from pathlib import Path

        dir_path = Path(directory)
        results = {"added": 0, "failed": 0, "images": []}

        # Find image files
        if recursive:
            image_files = list(dir_path.rglob(f"*{ext}") for ext in allowed_extensions)
            image_files = [f for files in image_files for f in files if f.is_file()]
        else:
            image_files = [
                f
                for f in dir_path.iterdir()
                if f.is_file() and f.suffix.lower() in allowed_extensions
            ]

        logger.info(f"Found {len(image_files)} images in {directory}")

        for image_path in image_files:
            try:
                # Try to infer tags from directory structure or filename
                # Use parent directory name as tag
                parent_tag = image_path.parent.name.replace("_", " ").replace("-", " ")
                tags = [parent_tag] if parent_tag not in ["", "rag_dataset"] else []

                # Add filename as tag too
                filename_tag = image_path.stem.replace("_", " ").replace("-", " ")
                if filename_tag and filename_tag not in tags:
                    tags.append(filename_tag)

                doc_id = self.add_image_from_path(str(image_path), tags)
                results["added"] += 1
                results["images"].append({"id": doc_id, "path": str(image_path)})

            except Exception as e:
                logger.error(f"Failed to add {image_path}: {e}")
                results["failed"] += 1

        logger.info(f"Added {results['added']} images, {results['failed']} failed")
        return results

    def search(
        self, image_bytes: bytes, top_k: int = 5, similarity_threshold: float = 0.2
    ) -> List[dict]:
        """
        Search for similar images.

        Args:
            image_bytes: Query image bytes
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of match dictionaries with id, score, and tags
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(image_bytes)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )

            # Format results
            matches = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 1.0
                    )
                    similarity = 1 - distance

                    if similarity >= similarity_threshold:
                        metadata = (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        )
                        tags_str = metadata.get("tags", "")
                        tags_list = [
                            t.strip() for t in tags_str.split(",") if t.strip()
                        ]

                        matches.append(
                            {
                                "id": doc_id,
                                "score": round(similarity, 4),
                                "tags": tags_list,
                                "metadata": metadata,
                            }
                        )

            logger.info(f"RAG search found {len(matches)} matches")
            return matches

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def search_by_text(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.2
    ) -> List[dict]:
        """
        Search for images by text query using CLIP text encoder.

        Args:
            query_text: Text query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of match dictionaries
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            # Encode text with CLIP
            inputs = self.clip_processor(
                text=[query_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)

            # Normalize
            embedding = text_features / text_features.norm(dim=-1, keepdim=True)
            query_embedding = embedding.squeeze().tolist()

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )

            matches = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 1.0
                    )
                    similarity = 1 - distance

                    if similarity >= similarity_threshold:
                        metadata = (
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        )
                        tags_str = metadata.get("tags", "")
                        tags_list = [
                            t.strip() for t in tags_str.split(",") if t.strip()
                        ]

                        matches.append(
                            {
                                "id": doc_id,
                                "score": round(similarity, 4),
                                "tags": tags_list,
                                "query": query_text,
                                "metadata": metadata,
                            }
                        )

            return matches

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Get RAG dataset statistics."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "model": self.model_name,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            return {"error": str(e)}

    def delete_collection(self):
        """Delete and recreate collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' recreated")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")


# Singleton instance
_clip_service: Optional[CLIPImageEmbeddingService] = None


def get_clip_image_service(
    model_name: str = "openai/clip-vit-large-patch14",
    persist_directory: Optional[str] = None,
) -> CLIPImageEmbeddingService:
    """Get or create CLIP image embedding service singleton."""
    global _clip_service
    if _clip_service is None:
        _clip_service = CLIPImageEmbeddingService(
            model_name=model_name, persist_directory=persist_directory
        )
    return _clip_service


def reset_clip_service():
    """Reset the singleton instance."""
    global _clip_service
    _clip_service = None
