"""Multi-Model Voter Service for cross-validating tags with multiple VLMs.

Uses multiple vision-language models to vote on tag presence,
increasing accuracy for sensitive or controversial tags.
"""

import logging
import re
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class MultiModelVoter:
    """多模型投票驗證器
    
    使用多個 VLM 模型對敏感標籤進行交叉驗證，
    提高標籤判斷的準確性。
    """
    
    # 可用的模型列表（需根據實際 LM Studio 配置調整）
    AVAILABLE_MODELS = {
        "glm-4.6v-flash": {
            "endpoint": "/chat/completions",
            "timeout": 60,
        },
        "qwen-vl-max": {
            "endpoint": "/chat/completions", 
            "timeout": 45,
        },
        "llava-next": {
            "endpoint": "/chat/completions",
            "timeout": 40,
        },
    }
    
    # 需要多模型驗證的敏感標籤
    SENSITIVE_TAGS_REQUIRING_VOTING = {
        "loli", "shota", "蘿莉", "正太", "幼女",
        "anal", "肛交", "rape", "強姦", "bestiality", "獸交",
        "ntr", "調教", "凌辱", "群交",
    }
    
    def __init__(self):
        """初始化 MultiModelVoter."""
        self.base_url = settings.LM_STUDIO_BASE_URL.rstrip("/")
        self.api_key = settings.LM_STUDIO_API_KEY
        
        # 選擇用於投票的模型（預設使用可用的）
        self.vote_models = ["glm-4.6v-flash"]
        
        self.vote_threshold = settings.MULTI_MODEL_VOTE_THRESHOLD
        self._executor = ThreadPoolExecutor(max_workers=2)
    
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
    
    def _get_verification_question(self, tag: str) -> str:
        """獲取特定標籤的驗證問題"""
        tag_lower = tag.lower()
        
        if tag_lower in ["loli", "蘿莉", "幼女"]:
            return "Is the character explicitly a child (prepubescent, under 12)? Answer ONLY YES or NO."
        elif tag_lower in ["shota", "正太"]:
            return "Is the character explicitly a young boy (under 12)? Answer ONLY YES or NO."
        elif tag_lower in ["anal", "肛交"]:
            return "Does this image explicitly depict anal intercourse? Answer ONLY YES or NO."
        elif tag_lower in ["rape", "強姦"]:
            return "Does this image explicitly depict rape or sexual assault? Answer ONLY YES or NO."
        elif tag_lower in ["bestiality", "獸交"]:
            return "Does this image explicitly depict bestiality (human with animal)? Answer ONLY YES or NO."
        elif tag_lower in ["ntr"]:
            return "Does this image depict cuckold/NTR (one person watching their partner with another)? Answer ONLY YES or NO."
        else:
            return f"Does this image explicitly contain or depict '{tag}'? Answer ONLY YES or NO."
    
    def _parse_response(self, content: str) -> bool:
        """解析 YES/NO 回應"""
        content = content.strip().upper()
        
        # 嚴格匹配
        if content.startswith("YES") and ("NO" not in content or content.startswith("YES ")):
            return True
        if content == "NO" or content.startswith("NO "):
            return False
        
        # 包含 YES 且不包含 NO
        if "YES" in content and "NO" not in content:
            return True
        if "NO" in content and "YES" not in content:
            return False
        
        # 不確定
        return False
    
    async def _query_model(
        self,
        model: str,
        image_bytes: bytes,
        tag: str
    ) -> Tuple[str, bool]:
        """查詢單個模型的驗證結果"""
        if settings.USE_MOCK_SERVICES:
            return (model, True)
        
        b64_img = self._prepare_image(image_bytes)
        if not b64_img:
            return (model, False)
        
        question = self._get_verification_question(tag)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
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
                
                content = result["choices"][0]["message"]["content"].strip()
                is_present = self._parse_response(content)
                
                logger.debug(f"模型 {model} 對 '{tag}' 的判斷: {'是' if is_present else '否'}")
                return (model, is_present)
                
        except Exception as e:
            logger.warning(f"模型 {model} 查詢失敗: {e}")
            return (model, False)
    
    async def vote_verify(
        self,
        image_bytes: bytes,
        tag: str
    ) -> Tuple[bool, float]:
        """多模型投票驗證標籤
        
        Args:
            image_bytes: 圖片字節
            tag: 要驗證的標籤
            
        Returns:
            (是否通過, 投票比例)
        """
        if not settings.ENABLE_MULTI_MODEL_VOTING:
            # 未啟用時返回預設值
            return (True, 1.0)
        
        # 檢查是否需要投票
        tag_lower = tag.lower()
        if tag_lower not in self.SENSITIVE_TAGS_REQUIRING_VOTING:
            # 非敏感標籤，不需要投票
            return (True, 1.0)
        
        logger.info(f"對敏感標籤 '{tag}' 進行多模型投票驗證")
        
        # 異步並行查詢所有模型
        import asyncio
        
        tasks = [
            self._query_model(model, image_bytes, tag)
            for model in self.vote_models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 統計投票結果
        total_models = 0
        positive_votes = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"投票查詢異常: {result}")
                continue
            
            model, is_positive = result
            total_models += 1
            if is_positive:
                positive_votes += 1
        
        if total_models == 0:
            logger.warning("所有模型查詢失敗")
            return (False, 0.0)
        
        # 計算投票比例
        vote_ratio = positive_votes / total_models
        
        # 判斷是否通過
        is_passed = vote_ratio >= self.vote_threshold
        
        logger.info(
            f"投票結果: {positive_votes}/{total_models} ({vote_ratio:.2%}), "
            f"通過閾值 {self.vote_threshold:.2%}: {'是' if is_passed else '否'}"
        )
        
        return (is_passed, vote_ratio)
    
    async def batch_vote_verify(
        self,
        image_bytes: bytes,
        tags: List[str]
    ) -> Dict[str, Tuple[bool, float]]:
        """批量投票驗證多個標籤
        
        Args:
            image_bytes: 圖片字節
            tags: 要驗證的標籤列表
            
        Returns:
            字典，key 為標籤名稱，value 為 (是否通過, 投票比例)
        """
        results = {}
        
        for tag in tags:
            is_passed, vote_ratio = await self.vote_verify(image_bytes, tag)
            results[tag] = (is_passed, vote_ratio)
        
        return results


# Singleton instance
_multi_model_voter: Optional[MultiModelVoter] = None


def get_multi_model_voter() -> MultiModelVoter:
    """獲取或創建 MultiModelVoter 單例"""
    global _multi_model_voter
    if _multi_model_voter is None:
        _multi_model_voter = MultiModelVoter()
    return _multi_model_voter