"""Check RAG database for test tags and informal tags."""

import chromadb
import json

# Connect to ChromaDB
client = chromadb.PersistentClient(path='./data/chroma_db')
collection = client.get_or_create_collection(name='manga_covers')

print(f"Total documents in RAG database: {collection.count()}")
print("=" * 80)

# Get all documents
results = collection.get(include=['metadatas'])

# Extract all tags
all_tags = set()
test_tags = set()
informal_tags = set()

# Keywords to identify test/informal tags
test_keywords = ['test', '測試', 'demo', 'sample', 'example', 'temp', '臨時', 'red', 'blue', 'green']
informal_keywords = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'gray', 'grey']

for metadata in results['metadatas']:
    tags_str = metadata.get('tags', '')
    if tags_str:
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        all_tags.update(tags)

        # Check for test tags
        for tag in tags:
            tag_lower = tag.lower()
            if any(kw in tag_lower for kw in test_keywords):
                test_tags.add(tag)

            # Check for informal tags (single color names)
            if tag_lower in informal_keywords:
                informal_tags.add(tag)

print(f"\nTotal unique tags in RAG database: {len(all_tags)}")
print(f"Test tags found: {len(test_tags)}")
print(f"Informal tags found: {len(informal_tags)}")

print("\n" + "=" * 80)
print("TEST TAGS:")
print("=" * 80)
for tag in sorted(test_tags):
    print(f"  - {tag}")

print("\n" + "=" * 80)
print("INFORMAL TAGS (single color names):")
print("=" * 80)
for tag in sorted(informal_tags):
    print(f"  - {tag}")

print("\n" + "=" * 80)
print("ALL TAGS:")
print("=" * 80)
for tag in sorted(all_tags):
    print(f"  - {tag}")

# Save results to file
output = {
    "total_documents": collection.count(),
    "total_unique_tags": len(all_tags),
    "test_tags": sorted(list(test_tags)),
    "informal_tags": sorted(list(informal_tags)),
    "all_tags": sorted(list(all_tags))
}

with open('rag_data_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("Results saved to: rag_data_analysis.json")
print("=" * 80)
