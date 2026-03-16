"""
Tag Vector Store Module
Handles loading tags from JSON and storing embeddings in ChromaDB
"""

import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class TagVectorStore:
    """Manages tag embeddings using ChromaDB for RAG retrieval"""

    # BGE models require special instruction for optimal performance
    BGE_INSTRUCTION = "Represent this sentence for searching relevant tags: "

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = "./chroma_db",
        embedding_model: str = "BAAI/bge-m3",
        use_instruction: bool = True,
    ):
        """
        Initialize the tag vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Sentence transformer model for embeddings
            use_instruction: Whether to use BGE-style instruction prefix
        """
        # Use configured collection name if not specified
        if collection_name is None:
            try:
                from app.config import settings

                collection_name = getattr(
                    settings, "CHROMA_TAG_COLLECTION", "tag_library"
                )
            except ImportError:
                collection_name = "tag_library"

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_instruction = use_instruction and "bge" in embedding_model.lower()
        self.embedding_model_name = embedding_model

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name or "tag_library",
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Model loaded successfully!")

    def load_tags_from_json(self, json_path: str) -> List[Dict[str, str]]:
        """
        Load tags from JSON file

        Args:
            json_path: Path to JSON file containing tags

        Returns:
            List of tag dictionaries with 'tag_name' and 'description'
        """
        with open(json_path, "r", encoding="utf-8") as f:
            tags = json.load(f)
        return tags

    def _generate_id(self, tag_name: str) -> str:
        """Generate unique ID for a tag"""
        return hashlib.md5(tag_name.encode()).hexdigest()

    def _create_embedding_text(self, tag: Dict[str, str]) -> str:
        """
        Create embedding text from tag data
        Combines tag name and description for better semantic matching
        """
        tag_name = tag.get("tag_name", "")
        description = tag.get("description", "")
        return f"{tag_name}: {description}"

    def _encode_text(self, text: str, is_query: bool = True) -> List[float]:
        """
        Encode text to embedding vector with BGE instruction if applicable

        Args:
            text: Text to encode
            is_query: If True, add BGE instruction prefix (for queries)
                     If False, encode as-is (for documents)
        """
        if self.use_instruction and is_query:
            # BGE models benefit from instruction prefix for queries only
            text = self.BGE_INSTRUCTION + text
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def add_tags(self, tags: List[Dict[str, str]], batch_size: int = 100):
        """
        Add tags to vector store

        Args:
            tags: List of tag dictionaries
            batch_size: Number of tags to process in each batch
        """
        total = len(tags)
        print(f"Adding {total} tags to vector store...")

        for i in range(0, total, batch_size):
            batch = tags[i : i + batch_size]

            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for tag in batch:
                tag_id = self._generate_id(tag["tag_name"])
                embedding_text = self._create_embedding_text(tag)
                # Documents should NOT use instruction prefix
                embedding = self._encode_text(embedding_text, is_query=False)

                ids.append(tag_id)
                documents.append(embedding_text)
                embeddings.append(embedding)
                metadatas.append(
                    {"tag_name": tag["tag_name"], "description": tag["description"]}
                )

            self.collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

            print(f"  Progress: {min(i + batch_size, total)}/{total} tags added")

        print(f"Successfully added {total} tags to collection")

    def initialize_from_json(self, json_path: str, force_reload: bool = False):
        """
        Initialize vector store from JSON file

        Args:
            json_path: Path to JSON file
            force_reload: If True, clear existing collection and reload
        """
        # Check if collection already has data
        existing_count = self.collection.count()

        if existing_count > 0 and not force_reload:
            print(
                f"Collection already contains {existing_count} tags. Skipping initialization."
            )
            print("Use force_reload=True to reinitialize.")
            return

        if force_reload and existing_count > 0:
            print("Clearing existing collection...")
            if self.collection_name:
                self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name or "tag_library",
                metadata={"hnsw:space": "cosine"},
            )

        # Load and add tags
        tags = self.load_tags_from_json(json_path)
        self.add_tags(tags)

    def search(
        self, query: str, top_k: int = 10, similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tags based on query text

        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1).
                                 If None, uses default from config (0.5)

        Returns:
            List of matching tags with similarity scores
        """
        # Use default threshold from config if not specified
        if similarity_threshold is None:
            try:
                from app.config import settings

                similarity_threshold = getattr(
                    settings, "RAG_SIMILARITY_THRESHOLD", 0.5
                )
            except ImportError:
                similarity_threshold = 0.5

        # Ensure threshold is not None
        if similarity_threshold is None:
            similarity_threshold = 0.5

        # Generate embedding for query with instruction if applicable
        query_embedding = self._encode_text(query, is_query=True)

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )

        # Format results
        matches = []
        if results.get("ids") and results["ids"][0] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            documents = results.get("documents", [[]])[0] or []

            for i, (tag_id, metadata, distance, document) in enumerate(
                zip(ids, metadatas, distances, documents)
            ):
                # Convert cosine distance to similarity score
                similarity = 1 - distance

                if similarity >= similarity_threshold:
                    matches.append(
                        {
                            "id": tag_id,
                            "tag_name": metadata["tag_name"],
                            "description": metadata["description"],
                            "similarity": round(similarity, 4),
                            "full_text": document,
                        }
                    )

        return matches

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        return {
            "collection_name": self.collection_name,
            "total_tags": self.collection.count(),
            "persist_directory": self.persist_directory,
        }


# Convenience function for quick initialization
def init_tag_store(
    json_path: str = "51標籤庫.json",
    persist_directory: str = "./chroma_db",
    force_reload: bool = False,
) -> TagVectorStore:
    """
    Initialize tag vector store from JSON file

    Args:
        json_path: Path to tag JSON file
        persist_directory: Directory for ChromaDB persistence
        force_reload: Force reload even if collection exists

    Returns:
        Initialized TagVectorStore instance
    """
    store = TagVectorStore(persist_directory=persist_directory)
    store.initialize_from_json(json_path, force_reload=force_reload)
    return store


if __name__ == "__main__":
    # Test the module
    import sys

    json_file = "51標籤庫.json"
    if len(sys.argv) > 1:
        json_file = sys.argv[1]

    # Initialize store
    store = init_tag_store(json_file, force_reload=True)

    # Print stats
    stats = store.get_collection_stats()
    print(f"\nCollection Stats: {stats}")

    # Test search
    print("\n--- Test Search ---")
    test_queries = ["cat girl with ears", "pregnant woman", "large breasts"]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = store.search(query, top_k=3)
        for r in results:
            print(f"  - {r['tag_name']} (score: {r['similarity']})")
