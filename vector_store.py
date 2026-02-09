"""
Vector Store Module
Handles ChromaDB initialization with OpenAI embeddings for RAG
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from pydantic import BaseModel, Field

from tag_loader import TagLoader, Document


class VectorStoreConfig(BaseModel):
    """Configuration for VectorStore"""
    
    collection_name: str = Field(default="tags", description="Name of the ChromaDB collection")
    persist_directory: Optional[str] = Field(default="./chroma_db", description="Directory to persist ChromaDB data")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model to use")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    class Config:
        arbitrary_types_allowed = True


class VectorStore:
    """
    Vector Store for managing tag embeddings using ChromaDB and OpenAI
    """
    
    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        collection_name: str = "tags",
        persist_directory: Optional[str] = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize VectorStore with ChromaDB and OpenAI embeddings
        
        Args:
            config: VectorStoreConfig object (optional)
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        # Use config if provided, otherwise create from parameters
        if config:
            self.config = config
        else:
            self.config = VectorStoreConfig(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_model=embedding_model,
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
        # Initialize ChromaDB client
        self._init_chromadb()
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if self.config.persist_directory:
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.chroma_client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"VectorStore initialized with collection: {self.config.collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to vector store
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
        """
        total = len(documents)
        print(f"Adding {total} documents to vector store...")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            
            ids = []
            contents = []
            embeddings = []
            metadatas = []
            
            for doc in batch:
                # Get embedding for document content
                embedding = self._get_embedding(doc.content)
                
                ids.append(doc.id)
                contents.append(doc.content)
                embeddings.append(embedding)
                metadatas.append(doc.metadata)
            
            # Add batch to collection
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"  Progress: {min(i + batch_size, total)}/{total} documents added")
        
        print(f"Successfully added {total} documents to collection")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with similarity scores
        """
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'distances', 'documents']
        )
        
        # Format results
        matches = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, (doc_id, metadata, distance, document) in enumerate(zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['documents'][0]
            )):
                # Convert cosine distance to similarity score
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    matches.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity': round(similarity, 4)
                    })
        
        return matches
    
    def initialize_from_json(
        self,
        json_path: str,
        force_reload: bool = False
    ) -> None:
        """
        Initialize vector store from JSON file
        
        Args:
            json_path: Path to JSON file containing tags
            force_reload: If True, clear existing collection and reload
        """
        # Check if collection already has data
        existing_count = self.collection.count()
        
        if existing_count > 0 and not force_reload:
            print(f"Collection already contains {existing_count} documents. Skipping initialization.")
            print("Use force_reload=True to reinitialize.")
            return
        
        if force_reload and existing_count > 0:
            print("Clearing existing collection...")
            self.chroma_client.delete_collection(self.config.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load documents using TagLoader
        loader = TagLoader(json_path)
        documents = loader.load()
        
        # Add documents to vector store
        self.add_documents(documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        return {
            'collection_name': self.config.collection_name,
            'total_documents': self.collection.count(),
            'persist_directory': self.config.persist_directory,
            'embedding_model': self.config.embedding_model
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.chroma_client.delete_collection(self.config.collection_name)
        print(f"Collection '{self.config.collection_name}' deleted")


# Convenience function for quick initialization
def init_vector_store(
    json_path: str = "51標籤庫.json",
    persist_directory: str = "./chroma_db",
    collection_name: str = "tags",
    embedding_model: str = "text-embedding-3-small",
    force_reload: bool = False,
    openai_api_key: Optional[str] = None
) -> VectorStore:
    """
    Initialize vector store from JSON file
    
    Args:
        json_path: Path to tag JSON file
        persist_directory: Directory for ChromaDB persistence
        collection_name: Name of the collection
        embedding_model: OpenAI embedding model
        force_reload: Force reload even if collection exists
        openai_api_key: OpenAI API key (optional, defaults to env var)
        
    Returns:
        Initialized VectorStore instance
    """
    store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key
    )
    store.initialize_from_json(json_path, force_reload=force_reload)
    return store


if __name__ == "__main__":
    # Test the module
    import sys
    
    json_file = "51標籤庫.json"
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Initialize store
    store = init_vector_store(json_file, force_reload=True)
    
    # Print stats
    stats = store.get_stats()
    print(f"\nCollection Stats: {stats}")
    
    # Test search
    print("\n--- Test Search ---")
    test_queries = [
        "cat girl with ears",
        "pregnant woman",
        "large breasts"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = store.search(query, top_k=3)
        for r in results:
            print(f"  - {r['metadata']['tag_name']} (score: {r['similarity']})")
