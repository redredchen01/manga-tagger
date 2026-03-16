"""Test tag mapping functionality."""

from app.services.tag_mapper import get_tag_mapper

# Test hair color mappings
test_tags = [
    "purple hair",
    "red hair",
    "blue hair",
    "green hair",
    "pink hair",
    "black hair",
    "blonde hair",
    "silver hair",
    "brown hair",
    "white hair",
    "orange hair",
    "gray hair",
    "gradient hair",
    "multicolor hair",
]

mapper = get_tag_mapper()

print("Testing hair color mappings:")
print("=" * 60)

for tag in test_tags:
    cn_tag = mapper.to_chinese(tag)
    if cn_tag:
        print(f"  {tag:20} -> {cn_tag}")
    else:
        print(f"  {tag:20} -> NOT FOUND!")

print("\n" + "=" * 60)
print(f"Total mappings: {len(mapper.en_to_cn)}")
