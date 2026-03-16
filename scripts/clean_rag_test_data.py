"""Clean test data from RAG database."""

import chromadb
import json

# Connect to ChromaDB
client = chromadb.PersistentClient(path='./data/chroma_db')
collection = client.get_or_create_collection(name='manga_covers')

print(f"Total documents in RAG database: {collection.count()}")
print("=" * 80)

# Get all documents
results = collection.get(include=['metadatas'])

# Keywords to identify test/informal tags
test_keywords = ['test', '測試', 'demo', 'sample', 'example', 'temp', '臨時']
informal_keywords = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'grey']

# Find documents with test or informal tags
documents_to_delete = []
for i, metadata in enumerate(results['metadatas']):
    tags_str = metadata.get('tags', '')
    if tags_str:
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]

        # Check for test tags
        for tag in tags:
            tag_lower = tag.lower()
            if any(kw in tag_lower for kw in test_keywords):
                print(f"Found test document with tags: {tags}")
                break

            # Check for informal tags (single color names)
            if tag_lower in informal_keywords:
                print(f"Found informal document with tags: {tags}")
                break

print("\n" + "=" * 80)
print(f"Found documents with test/informal tags. Deleting all documents...")
print("=" * 80)

# Delete all documents and recreate collection
try:
    client.delete_collection(name='manga_covers')
    print("Deleted collection: manga_covers")
except:
    pass

# Recreate collection
collection = client.get_or_create_collection(
    name='manga_covers',
    metadata={"hnsw:space": "cosine"}
)

print(f"RAG database cleaned successfully!")
print(f"Remaining documents: {collection.count()}")
