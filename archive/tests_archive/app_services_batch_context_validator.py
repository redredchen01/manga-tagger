"""Batch Context Validator Service for verifying tag consistency across multiple images.

This service validates tag recommendations by checking their consistency
across a batch of related images (e.g., a manga series).
"""

import logging
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class BatchContextValidator:
    """批量圖片上下文驗證器
    
    通過檢查標籤在批量圖片中的一致性來驗證標籤推薦的準確性。
    例如：如果系列圖片中只有 1 張有某標籤，但其他張都沒有，
    該標籤可能是誤判。
    """
    
    def __init__(self):
        """初始化 BatchContextValidator."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        self.model = settings.LM_STUDIO_VISION_MODEL
        
        # 配置
        self.min_images = settings.BATCH_CONTEXT_MIN_IMAGES
        self.consistency_threshold = settings.BATCH_CONTEXT_CONSISTENCY_THRESHOLD
        
        # 敏感標籤列表（需要更嚴格的一致性檢查）
        self.sensitive_tags = {
            "loli", "shota", "蘿莉", "正太", "幼女", "少女",
            "anal", "肛交", "rape", "強姦", "bestiality", "獸交",
            "巨乳", "貧乳", "爆乳", "貧乳"
        }
    
    def _prepare_image(self, image_bytes: bytes) -> Optional[str]:
        """準備圖片為 base64 編碼"""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"圖片準備失敗: {e}")
            return None
    
    async def _verify_tag_in_image(
        self,
        image_bytes: bytes,
        tag: str
    ) -> bool:
        """驗證單張圖片中是否存在指定標籤"""
        if settings.USE_MOCK_SERVICES:
            return True
        
        b64_img = self._prepare_image(image_bytes)
        if not b64_img:
            return False
        
        prompt = f"Does this image contain '{tag}'? Answer ONLY YES or NO."
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        },
                    ],
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1,
        }
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                
                content = result["choices"][0]["message"]["content"].strip().upper()
                return content.startswith("YES") or "YES" in content.split()[0]
        except Exception as e:
            logger.warning(f"標籤 '{tag}' 驗證失敗: {e}")
            return False
    
    async def validate_batch_tags(
        self,
        images: List[bytes],
        candidate_tags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """驗證批量圖片中的標籤一致性
        
        Args:
            images: 批量圖片字節列表
            candidate_tags: 候選標籤列表，每項包含 'tag' 和 'confidence'
            
        Returns:
            過濾後的標籤列表
        """
        if not settings.ENABLE_BATCH_CONTEXT_VALIDATION:
            return candidate_tags
        
        if len(images) < self.min_images:
            logger.debug(f"圖片數量 {len(images)} < 最小要求 {self.min_images}")
            return candidate_tags
        
        logger.info(f"開始批量上下文驗證，共 {len(images)} 張圖片，{len(candidate_tags)} 個候選標籤")
        
        # 過濾低置信度標籤（不需要批量驗證）
        high_conf_tags = [t for t in candidate_tags if t.get("confidence", 0) >= 0.6]
        borderline_tags = [t for t in candidate_tags if 0.4 <= t.get("confidence", 0) < 0.6]
        
        validated_tags = high_conf_tags.copy()
        
        # 對邊界置信度標籤進行批量驗證
        for tag_item in borderline_tags:
            tag_name = tag_item.get("tag", "")
            if not tag_name:
                continue
            
            # 檢查是否是敏感標籤
            is_sensitive = tag_name.lower() in self.sensitive_tags or tag_name in self.sensitive_tags
            
            # 在批量圖片中統計標籤出現次數
            present_count = 0
            present_in_high_conf = False
            
            for img_bytes in images:
                if await self._verify_tag_in_image(img_bytes, tag_name):
                    present_count += 1
            
            # 計算一致性分數
            consistency_score = present_count / len(images)
            
            # 敏感標籤需要更高的出現率
            required_threshold = 0.7 if is_sensitive else self.consistency_threshold
            
            logger.debug(
                f"標籤 '{tag_name}' 一致性分數: {consistency_score:.2f} "
                f"(出現 {present_count}/{len(images)} 張)"
            )
            
            if consistency_score >= required_threshold:
                # 一致性足夠，保留標籤
                tag_item["consistency_score"] = consistency_score
                tag_item["validated"] = True
                validated_tags.append(tag_item)
            else:
                # 一致性不足，降低置信度或過濾
                if consistency_score >= 0.3:
                    # 部分一致，降低置信度但保留
                    adjusted_confidence = tag_item.get("confidence", 0) * consistency_score
                    tag_item["confidence"] = adjusted_confidence
                    tag_item["consistency_score"] = consistency_score
                    tag_item["validated"] = False
                    tag_item["note"] = "降低置信度（批量一致性不足）"
                    validated_tags.append(tag_item)
                    logger.info(f"標籤 '{tag_name}' 置信度降低: {adjusted_confidence:.2f}")
                else:
                    # 幾乎不一致，過濾掉
                    logger.info(f"標籤 '{tag_name}' 被過濾（一致性 {consistency_score:.2f} < {required_threshold}）")
        
        # 按置信度排序
        validated_tags.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        logger.info(f"批量驗證完成，保留 {len(validated_tags)}/{len(candidate_tags)} 個標籤")
        
        return validated_tags
    
    async def validate_single_image_tags(
        self,
        image_bytes: bytes,
        candidate_tags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """對單張圖片進行標籤驗證（邊界置信度檢查）
        
        當只有單張圖片時，使用 VLM 進行二次驗證。
        """
        if not settings.ENABLE_BATCH_CONTEXT_VALIDATION:
            return candidate_tags
        
        # 只對邊界置信度標籤進行驗證
        validated_tags = []
        
        for tag_item in candidate_tags:
            confidence = tag_item.get("confidence", 0)
            tag_name = tag_item.get("tag", "")
            
            # 只驗證 0.4-0.6 區間的標籤
            if 0.4 <= confidence < 0.6 and tag_name:
                is_present = await self._verify_tag_in_image(image_bytes, tag_name)
                
                if is_present:
                    # 驗證通過，稍微提升置信度
                    new_confidence = min(0.75, confidence + 0.1)
                    tag_item["confidence"] = new_confidence
                    tag_item["validated"] = True
                    tag_item["note"] = "VLM 二次驗證通過"
                    validated_tags.append(tag_item)
                    logger.info(f"標籤 '{tag_name}' VLM 驗證通過，置信度提升至 {new_confidence:.2f}")
                else:
                    # 驗證不通過，過濾
                    logger.info(f"標籤 '{tag_name}' VLM 驗證不通過，已過濾")
                    continue
            else:
                validated_tags.append(tag_item)
        
        return validated_tags
    
    def calculate_tag_pair_consistency(
        self,
        tag1: str,
        tag2: str,
        tag1_present_images: List[bool],
        tag2_present_images: List[bool]
    ) -> float:
        """計算兩個標籤在批量圖片中出現的一致性
        
        用於檢查標籤組合是否合理（例如：貓娘 和 普通女孩 不應同時出現）
        
        Returns:
            一致性分數（0-1），1 表示完全一致，0 表示完全不一致
        """
        if len(tag1_present_images) != len(tag2_present_images):
            return 0.0
        
        if len(tag1_present_images) == 0:
            return 1.0
        
        # 計算同時出現或同時不出現的比例
        consistent_count = 0
        for present1, present2 in zip(tag1_present_images, tag2_present_images):
            if present1 == present2:
                consistent_count += 1
        
        return consistent_count / len(tag1_present_images)


# Singleton instance
_batch_context_validator: Optional[BatchContextValidator] = None


def get_batch_context_validator() -> BatchContextValidator:
    """獲取或創建 BatchContextValidator 單例"""
    global _batch_context_validator
    if _batch_context_validator is None:
        _batch_context_validator = BatchContextValidator()
    return _batch_context_validator
