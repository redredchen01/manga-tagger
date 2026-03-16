# End-to-End Pipeline Test (Simplified)
import sys
import asyncio
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Enable mock mode
import app.config
app.config.settings.USE_MOCK_SERVICES = True

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_recommender_service import get_tag_recommender_service


async def test_complete_pipeline():
    """Test complete tagging pipeline."""
    print("=" * 60)
    print("完整貼標系統測試 (Mock Mode)")
    print("=" * 60)
    
    # 1. Create test image
    print("\n[1/3] 建立測試圖片...")
    img = Image.new("RGB", (512, 512), color=(255, 100, 150))  # Pink image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_data = img_bytes.getvalue()
    print(f"      圖片大小: {len(image_data)} bytes")
    
    # 2. VLM Analysis
    print("\n[2/3] VLM 圖片分析...")
    vlm_service = LMStudioVLMService()
    vlm_result = await vlm_service.extract_metadata(image_data)
    print(f"      描述: {vlm_result.get('description', 'N/A')[:80]}...")
    print(f"      角色類型: {vlm_result.get('character_types', [])}")
    print(f"      服裝: {vlm_result.get('clothing', [])}")
    print(f"      身體特徵: {vlm_result.get('body_features', [])}")
    print(f"      原始關鍵詞: {vlm_result.get('raw_keywords', [])}")
    
    # 3. Tag Recommendation
    print("\n[3/3] 標籤推薦...")
    recommender = get_tag_recommender_service()
    recommendations = await recommender.recommend_tags(
        vlm_analysis=vlm_result,
        rag_matches=[],  # Skip RAG for simplified test
        top_k=5,
        confidence_threshold=0.3,
        vlm_service=vlm_service,
        image_bytes=image_data,
    )
    
    print("\n" + "=" * 60)
    print("🏷️  最終推薦標籤")
    print("=" * 60)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.tag} (置信度: {rec.confidence:.2f})")
            print(f"     來源: {rec.source} | 原因: {rec.reason[:50]}...")
    else:
        print("  ❌ 沒有推薦標籤！")
    
    print("\n" + "=" * 60)
    if recommendations:
        print("✅ 貼標系統運作正常！")
    else:
        print("❌ 貼標系統有問題，需要進一步排查")
    print("=" * 60)
    
    return len(recommendations) > 0


if __name__ == "__main__":
    success = asyncio.run(test_complete_pipeline())
    sys.exit(0 if success else 1)
import sys
import asyncio
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# Enable mock mode
import app.config

app.config.settings.USE_MOCK_SERVICES = True

from app.services.lm_studio_vlm_service_v2 import LMStudioVLMService
from app.services.tag_recommender_service import get_tag_recommender_service
from app.services.rag_service import get_rag_service


async def test_complete_pipeline():
    """Test complete tagging pipeline."""
    print("=" * 60)
    print("完整貼標系統測試 (Mock Mode)")
    print("=" * 60)

    # 1. Create test image
    print("\n[1/4] 建立測試圖片...")
    img = Image.new("RGB", (512, 512), color=(255, 100, 150))  # Pink image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_data = img_bytes.getvalue()
    print(f"      圖片大小: {len(image_data)} bytes")

    # 2. VLM Analysis
    print("\n[2/4] VLM 圖片分析...")
    vlm_service = LMStudioVLMService()
    vlm_result = await vlm_service.extract_metadata(image_data)
    print(f"      描述: {vlm_result.get('description', 'N/A')[:80]}...")
    print(f"      角色類型: {vlm_result.get('character_types', [])}")
    print(f"      服裝: {vlm_result.get('clothing', [])}")
    print(f"      身體特徵: {vlm_result.get('body_features', [])}")
    print(f"      原始關鍵詞: {vlm_result.get('raw_keywords', [])}")

    # 3. RAG Search
    print("\n[3/4] RAG 相似圖搜尋...")
    rag_service = get_rag_service()
    try:
        rag_matches = await rag_service.search_similar(image_data, top_k=3)
        print(f"      找到 {len(rag_matches)} 個相似圖片")
    except Exception as e:
        print(f"      RAG 搜尋略過 (可能未初始化): {e}")
        rag_matches = []

    # 4. Tag Recommendation
    print("\n[4/4] 標籤推薦...")
    recommender = get_tag_recommender_service()
    recommendations = await recommender.recommend_tags(
        vlm_analysis=vlm_result,
        rag_matches=rag_matches,
        top_k=5,
        confidence_threshold=0.3,
        vlm_service=vlm_service,
        image_bytes=image_data,
    )

    print("\n" + "=" * 60)
    print("🏷️  最終推薦標籤")
    print("=" * 60)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.tag} (置信度: {rec.confidence:.2f})")
            print(f"    來源: {rec.source} | 原因: {rec.reason}")
    else:
        print("  ❌ 沒有推薦標籤！")

    print("\n" + "=" * 60)
    if recommendations:
        print("✅ 貼標系統運作正常！")
    else:
        print("❌ 貼標系統有問題，需要進一步排查")
    print("=" * 60)

    return len(recommendations) > 0


if __name__ == "__main__":
    success = asyncio.run(test_complete_pipeline())
    sys.exit(0 if success else 1)
