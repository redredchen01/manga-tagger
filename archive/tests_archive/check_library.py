"""Check actual tags in library."""

import sys

sys.path.insert(0, ".")

import json

print("=" * 70)
print("Tag Library Content Analysis")
print("=" * 70)

# Load tags
with open("data/tags.json", "r", encoding="utf-8") as f:
    tags = json.load(f)

print(f"\nTotal tags: {len(tags)}")

# Print first 30 tags
print("\nFirst 30 tags:")
for i, tag in enumerate(tags[:30], 1):
    print(f"  {i}. {tag['tag_name']}")

# Search for specific keywords
print("\n" + "=" * 70)
print("Searching for keywords")
print("=" * 70)

search_terms = ["demon", "惡魔", "loli", "蘿莉", "cat", "貓", "school", "學"]

for term in search_terms:
    matching = [t["tag_name"] for t in tags if term.lower() in t["tag_name"].lower()]
    print(f"\n'{term}': {len(matching)} matches")
    if matching:
        print(f"  Sample: {matching[:5]}")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
