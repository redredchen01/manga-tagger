# Test RAG and embedding services
import asyncio
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw
from app.services.rag_service import RAGService


def create_test_image(color="red"):
    """Create a simple test image."""
    img = Image.new("RGB", (512, 512), color=color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_rag_service():
    """Test RAG service functionality."""
    print("=" * 60)
    print("TESTING RAG SERVICE")
    print("=" * 60)

    rag = RAGService()

    # Test 1: Add images to RAG
    print("\n[1] Adding images to RAG...")

    red_image = create_test_image("red")
    blue_image = create_test_image("blue")

    await rag.add_image(
        red_image,
        tags=["紅髮", "蘿莉", "校服"],
        metadata={"description": "Red character"},
    )
    print("  - Added red image with tags: 紅髮, 蘿莉, 校服")

    await rag.add_image(
        blue_image,
        tags=["藍髮", "少女", "泳裝"],
        metadata={"description": "Blue character"},
    )
    print("  - Added blue image with tags: 藍髮, 少女, 泳裝")

    # Test 2: Search similar
    print("\n[2] Searching for similar images...")
    test_image = create_test_image("red")
    matches = await rag.search_similar(test_image, top_k=5)

    print(f"  - Found {len(matches)} matches:")
    for i, match in enumerate(matches[:3]):
        print(
            f"    {i + 1}. Score: {match.get('score', 0):.3f}, Tags: {match.get('tags', [])}"
        )

    # Test 3: Get stats
    print("\n[3] RAG Stats:")
    stats = rag.get_stats()
    print(f"  - Total documents: {stats.get('total_documents', 0)}")
    print(f"  - Embedding dimension: {stats.get('embedding_dimension', 'N/A')}")
    print(f"  - Collection name: {stats.get('collection_name', 'N/A')}")

    print("\n" + "=" * 60)
    if matches:
        print("SUCCESS! RAG service is working!")
    else:
        print("WARNING: RAG service returned no matches")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_rag_service())
