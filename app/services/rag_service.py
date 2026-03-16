"""RAG Service for Stage 2: Image Similarity Search.

Uses LM Studio for embeddings and ChromaDB for storage.
"""

import io
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.lm_studio_embedding_service import (
    LMStudioEmbeddingService,
    get_embedding_service,
)
from app.services.chinese_embedding_service import (
    ChineseEmbeddingService,
    get_chinese_embedding_service,
)

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service for image similarity search."""

    def __init__(self):
        """Initialize ChromaDB and embedding service."""
        self.chroma_client = None
        self.collection = None
        self.device = settings.DEVICE
        self.embedding_service: Optional[LMStudioEmbeddingService] = None
        self.chinese_embedding_service: Optional[ChineseEmbeddingService] = None

        # Initialize ChromaDB
        self._init_chromadb()

        # Initialize embedding service if LM Studio is enabled
        if settings.USE_LM_STUDIO:
            self.embedding_service = get_embedding_service()
            logger.info("RAG Service: Using LM Studio for embeddings")
        else:
            logger.info("RAG Service: Using deterministic embeddings")

        # Initialize Chinese embedding service if enabled
        if getattr(settings, "USE_CHINESE_EMBEDDINGS", True):
            self.chinese_embedding_service = get_chinese_embedding_service()
            if self.chinese_embedding_service.is_available():
                logger.info("RAG Service: Chinese embeddings enabled")
            else:
                logger.warning(
                    "RAG Service: Chinese embeddings not available, will use fallback"
                )
        else:
            logger.info("RAG Service: Chinese embeddings disabled")

    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            logger.info("Initializing ChromaDB")

            # Ensure directory exists
            chroma_path = getattr(settings, "CHROMA_DB_PATH", "./data/chroma_db")
            Path(chroma_path).mkdir(parents=True, exist_ok=True)

            # Initialize client with new ChromaDB API
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)

            # Get or create collection
            collection_name = getattr(
                settings, "CHROMA_COLLECTION_NAME", "manga_covers"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB initialized: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            # Create fallback in-memory storage
            self.chroma_client = chromadb.Client()
            collection_name = getattr(
                settings, "CHROMA_COLLECTION_NAME", "manga_covers"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

    def _prepare_image(self, image_bytes: bytes) -> Image.Image:
        """Prepare image for embedding."""
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    async def _generate_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Generate embedding for image using LM Studio or fallback."""
        try:
            if self.embedding_service:
                # Use LM Studio embedding service with 512 dimensions for ChromaDB
                embedding_list = await self.embedding_service.generate_embedding(
                    image_bytes, target_dim=512
                )
                return np.array(embedding_list, dtype=np.float32)
            else:
                # Fallback to deterministic embedding
                return self._generate_deterministic_embedding(image_bytes)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._generate_deterministic_embedding(image_bytes)

    def _generate_deterministic_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Generate deterministic embedding for image as fallback."""
        import hashlib

        hash_obj = hashlib.md5(image_bytes)
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.randn(512).astype(np.float32)

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def search_similar(
        self, image_bytes: bytes, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images in RAG dataset.

        Args:
            image_bytes: Query image bytes
            top_k: Number of results to return

        Returns:
            List of match dictionaries with id, score, and tags
        """
        import asyncio
        
        try:
            # Generate embedding for query image with timeout
            try:
                query_embedding = await asyncio.wait_for(
                    self._generate_embedding(image_bytes), 
                    timeout=5.0  # 5 second timeout for embedding
                )
            except asyncio.TimeoutError:
                logger.warning("Embedding generation timed out, skipping RAG search")
                return []

            # Search in ChromaDB
            if self.collection is not None and hasattr(self.collection, "query"):
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=["metadatas", "distances"],
                )
            else:
                logger.warning("ChromaDB collection not properly initialized")
                return []

            # Format results
            matches = []
            if "ids" in results and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = (
                        results["distances"][0][i] if results["distances"] else 1.0
                    )
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )

                    # Convert distance to similarity score (cosine similarity)
                    similarity = 1.0 - distance

                    # Clamp to valid range with proper rounding
                    similarity = max(0.0, min(0.99, similarity))
                    similarity = round(float(similarity), 4)

                    # Get threshold
                    threshold = getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.7)

                    # Filter by threshold
                    if similarity >= threshold:
                        tags_str = metadata.get("tags", "")
                        tags_list = (
                            tags_str.split(",") if isinstance(tags_str, str) else []
                        )
                        tags_list = [tag.strip() for tag in tags_list if tag.strip()]

                        matches.append(
                            {
                                "id": doc_id,
                                "score": similarity,
                                "tags": tags_list,
                                "metadata": metadata,
                            }
                        )

            logger.info(f"RAG search found {len(matches)} matches above threshold")
            return matches

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            # Return empty results rather than raising
            return []

    async def add_image(
        self,
        image_bytes: bytes,
        tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an image to the RAG dataset.

        Args:
            image_bytes: Image bytes
            tags: List of tags for this image
            metadata: Optional additional metadata

        Returns:
            Document ID
        """
        try:
            # Generate embedding
            embedding = await self._generate_embedding(image_bytes)

            # Generate unique ID
            doc_id = str(uuid.uuid4())

            # Prepare metadata - convert tags list to string
            tags_string = ",".join(tags) if isinstance(tags, list) else str(tags)
            doc_metadata = {"tags": tags_string, **(metadata or {})}

            # Add to collection
            if self.collection is not None and hasattr(self.collection, "add"):
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[doc_metadata],
                )

                # Persist if possible (newer ChromaDB versions auto-persist)
                pass  # ChromaDB PersistentClient auto-persists
            else:
                logger.warning("ChromaDB collection not properly initialized")

            logger.info(f"Added image to RAG: {doc_id} with {len(tags)} tags")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add image to RAG: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG dataset statistics."""
        try:
            if self.collection is not None and hasattr(self.collection, "count"):
                count = self.collection.count()
            else:
                count = 0

            return {
                "total_documents": count,
                "collection_name": getattr(
                    settings, "CHROMA_COLLECTION_NAME", "manga_covers"
                ),
                "embedding_mode": "lm_studio"
                if self.embedding_service
                else "deterministic",
                "use_lm_studio": settings.USE_LM_STUDIO,
                "chinese_embeddings_enabled": getattr(
                    settings, "USE_CHINESE_EMBEDDINGS", True
                ),
                "chinese_embeddings_available": (
                    self.chinese_embedding_service.is_available()
                    if self.chinese_embedding_service
                    else False
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_documents": 0, "error": str(e)}

    async def search_tags_by_text(
        self,
        query_text: str,
        tag_list: List[str],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search for tags by text similarity using Chinese embeddings.

        Args:
            query_text: Query text (e.g., image description)
            tag_list: List of available tags
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of dictionaries with tag and similarity score
        """
        if (
            not self.chinese_embedding_service
            or not self.chinese_embedding_service.is_available()
        ):
            logger.warning("Chinese embedding service not available for text search")
            return []

        try:
            results = await self.chinese_embedding_service.search_tags_by_text(
                query_text=query_text,
                tag_list=tag_list,
                top_k=top_k,
                threshold=threshold,
            )

            logger.info(f"Text-based tag search found {len(results)} matches")
            return results

        except Exception as e:
            logger.error(f"Text-based tag search failed: {e}")
            return []

    async def hybrid_search(
        self,
        image_bytes: bytes,
        query_text: Optional[str] = None,
        tag_list: Optional[List[str]] = None,
        image_top_k: int = 5,
        text_top_k: int = 10,
        text_threshold: float = 0.3,
        image_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining image similarity and text similarity.

        Args:
            image_bytes: Query image bytes
            query_text: Optional query text for text-based search
            tag_list: Optional list of tags for text-based search
            image_top_k: Number of image results to return
            text_top_k: Number of text results to return
            text_threshold: Minimum similarity threshold for text search
            image_weight: Weight for image similarity scores
            text_weight: Weight for text similarity scores

        Returns:
            Dictionary with image_results, text_results, and combined_results
        """
        results = {"image_results": [], "text_results": [], "combined_results": []}

        # Image-based search (always performed)
        try:
            image_results = await self.search_similar(image_bytes, top_k=image_top_k)
            results["image_results"] = image_results
            logger.info(f"Image search found {len(image_results)} matches")
        except Exception as e:
            logger.error(f"Image search failed: {e}")

        # Text-based search (if query text and tag list provided)
        if (
            query_text
            and tag_list
            and self.chinese_embedding_service
            and self.chinese_embedding_service.is_available()
        ):
            try:
                text_results = await self.search_tags_by_text(
                    query_text=query_text,
                    tag_list=tag_list,
                    top_k=text_top_k,
                    threshold=text_threshold,
                )
                results["text_results"] = text_results
                logger.info(f"Text search found {len(text_results)} matches")
            except Exception as e:
                logger.error(f"Text search failed: {e}")

        # Combine results (if both searches were performed)
        if results["image_results"] and results["text_results"]:
            combined_results = self._combine_search_results(
                image_results=results["image_results"],
                text_results=results["text_results"],
                image_weight=image_weight,
                text_weight=text_weight,
            )
            results["combined_results"] = combined_results
            logger.info(f"Combined search found {len(combined_results)} results")

        return results

    def _combine_search_results(
        self,
        image_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        image_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Combine image and text search results.

        Args:
            image_results: Results from image search
            text_results: Results from text search
            image_weight: Weight for image scores
            text_weight: Weight for text scores

        Returns:
            Combined and sorted list of results
        """
        combined = {}

        # Process image results
        for result in image_results:
            doc_id = result.get("id", "")
            if doc_id:
                combined[doc_id] = {
                    "id": doc_id,
                    "tags": result.get("tags", []),
                    "metadata": result.get("metadata", {}),
                    "image_score": result.get("score", 0.0),
                    "text_score": 0.0,
                    "combined_score": result.get("score", 0.0) * image_weight,
                }

        # Process text results and merge with image results
        for result in text_results:
            tag = result.get("tag", "")
            text_score = result.get("similarity", 0.0)

            # Find matching image results by tag
            for doc_id, doc_data in combined.items():
                if tag in doc_data.get("tags", []):
                    doc_data["text_score"] = text_score
                    doc_data["combined_score"] = (
                        doc_data["image_score"] * image_weight
                        + text_score * text_weight
                    )
                    break
            else:
                # Create new entry for text-only result
                combined[f"text_{tag}"] = {
                    "id": f"text_{tag}",
                    "tags": [tag],
                    "metadata": {"source": "text_only"},
                    "image_score": 0.0,
                    "text_score": text_score,
                    "combined_score": text_score * text_weight,
                }

        # Sort by combined score and return
        sorted_results = sorted(
            combined.values(), key=lambda x: x["combined_score"], reverse=True
        )

        return sorted_results
