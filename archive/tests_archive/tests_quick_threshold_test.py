"""快速測試新的閾值配置"""

from app.config import settings

print("="*60)
print("當前閾值配置")
print("="*60)

print(f"\n【匹配閾值】")
print(f"  RAG_SIMILARITY_THRESHOLD: {settings.RAG_SIMILARITY_THRESHOLD}")
print(f"  LEXICAL_MATCH_THRESHOLD:  {getattr(settings, 'LEXICAL_MATCH_THRESHOLD', 'N/A')}")
print(f"  CHINESE_EMBEDDING_THRESHOLD: {settings.CHINESE_EMBEDDING_THRESHOLD}")
print(f"  HYBRID_SCORING_ALPHA: {settings.HYBRID_SCORING_ALPHA}")

print(f"\n【類別權重】")
for cat, weight in settings.CATEGORY_WEIGHTS.items():
    print(f"  {cat}: {weight}")

print(f"\n【功能開關】")
print(f"  CONFLICT_CHECK_ENABLED: {getattr(settings, 'CONFLICT_CHECK_ENABLED', False)}")

print("\n" + "="*60)
print("閾值已調整到平衡位置，請重新測試！")
print("="*60)
