"""
Tag Loader Module
Handles loading tags from JSON and converting to Document objects for vector storage
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document object representing a tag with its metadata"""
    
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Combined text content for embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        arbitrary_types_allowed = True


class TagLoader:
    """Loads tags from JSON files and converts them to Document objects"""
    
    def __init__(self, json_path: str):
        """
        Initialize TagLoader
        
        Args:
            json_path: Path to JSON file containing tags
        """
        self.json_path = Path(json_path)
        self._documents: List[Document] = []
        
    def load(self) -> List[Document]:
        """
        Load tags from JSON and convert to Document objects
        
        Returns:
            List of Document objects
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"Tag file not found: {self.json_path}")
        
        # Load JSON data
        with open(self.json_path, 'r', encoding='utf-8') as f:
            tags = json.load(f)
        
        # Convert each tag to Document
        documents = []
        for idx, tag in enumerate(tags):
            doc = self._convert_to_document(tag, idx)
            documents.append(doc)
        
        self._documents = documents
        return documents
    
    def _convert_to_document(self, tag: Dict[str, Any], index: int) -> Document:
        """
        Convert a tag dictionary to a Document object
        
        Args:
            tag: Dictionary containing tag data
            index: Index for generating unique ID
            
        Returns:
            Document object
        """
        tag_name = tag.get('tag_name', '')
        description = tag.get('description', '')
        
        # Create unique ID
        doc_id = f"tag_{index:04d}_{self._generate_id(tag_name)}"
        
        # Create content for embedding (combine tag_name and description)
        content = self._create_content(tag_name, description)
        
        # Create metadata
        metadata = {
            'tag_name': tag_name,
            'description': description,
            'source': str(self.json_path),
            'index': index
        }
        
        return Document(
            id=doc_id,
            content=content,
            metadata=metadata
        )
    
    def _generate_id(self, tag_name: str) -> str:
        """Generate a short unique identifier from tag name"""
        import hashlib
        return hashlib.md5(tag_name.encode('utf-8')).hexdigest()[:8]
    
    def _create_content(self, tag_name: str, description: str) -> str:
        """
        Create content text for embedding
        Combines tag name and description for better semantic matching
        """
        if description:
            return f"{tag_name}: {description}"
        return tag_name
    
    def get_documents(self) -> List[Document]:
        """Get loaded documents (load if not already loaded)"""
        if not self._documents:
            return self.load()
        return self._documents
    
    def get_tag_names(self) -> List[str]:
        """Get list of all tag names"""
        docs = self.get_documents()
        return [doc.metadata.get('tag_name', '') for doc in docs]
    
    def get_by_tag_name(self, tag_name: str) -> Optional[Document]:
        """
        Get document by tag name
        
        Args:
            tag_name: Name of the tag to find
            
        Returns:
            Document if found, None otherwise
        """
        docs = self.get_documents()
        for doc in docs:
            if doc.metadata.get('tag_name') == tag_name:
                return doc
        return None
    
    def __len__(self) -> int:
        """Return number of documents"""
        return len(self.get_documents())
    
    def __iter__(self):
        """Make TagLoader iterable"""
        return iter(self.get_documents())


if __name__ == "__main__":
    # Test the module
    import sys
    
    json_file = "51標籤庫.json"
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # Initialize loader
    loader = TagLoader(json_file)
    
    # Load documents
    documents = loader.load()
    
    # Print statistics
    print(f"Loaded {len(documents)} documents")
    print("\n--- Sample Documents ---")
    
    # Show first 5 documents
    for doc in documents[:5]:
        print(f"\nID: {doc.id}")
        print(f"Tag: {doc.metadata['tag_name']}")
        print(f"Content: {doc.content[:100]}...")
        print(f"Description: {doc.metadata['description'][:100]}...")
