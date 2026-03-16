#!/usr/bin/env python3
"""
測試各種 VLM 修復方案
"""

import asyncio
import base64
import io
import json
import httpx
from PIL import Image

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "zai-org/glm-4.6v-flash"


def create_test_image():
    """創建測試圖片"""
    img = Image.new("RGB", (512, 512), color="white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([150, 100, 350, 300], outline="black", width=3)
    draw.ellipse([200, 160, 230, 190], fill="blue")
    draw.text((180, 350), "Anime Girl", fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def test_vlm_fix(method_name, payload_func):
    """測試特定修復方案"""
    print(f"\n{'=' * 60}")
    print(f"測試: {method_name}")
    print("=" * 60)

    try:
        image_bytes = create_test_image()
        payload = payload_func(image_bytes)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "")

                # 檢查是否有有效內容
                effective_content = content if content else reasoning

                if effective_content and len(effective_content.strip()) > 10:
                    # 檢查是否不是錯誤消息
                    if not any(
                        err in effective_content.lower()
                        for err in ["unable to", "failed", "error", "cannot"]
                    ):
                        print(f"[OK] 成功!")
                        print(f"內容: {effective_content[:200]}...")
                        return True, effective_content

        print(f"[FAIL] 失敗: 無有效內容")
        return False, ""

    except Exception as e:
        print(f"[ERROR] 錯誤: {e}")
        return False, str(e)


def make_payload_v1(image_bytes):
    """方案 1: 原始格式 + thinking disabled"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List anime character tags:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.5,
        "thinking": {"type": "disabled"},
    }


def make_payload_v2(image_bytes):
    """方案 2: 更簡單的提示"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tags:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3,
    }


def make_payload_v3(image_bytes):
    """方案 3: 使用 system message"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an anime image tagger. List relevant tags.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What anime tags apply?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "max_tokens": 150,
        "temperature": 0.5,
        "thinking": {"type": "disabled"},
    }


def make_payload_v4(image_bytes):
    """方案 4: 更低的 temperature 和 top_p"""
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": " anime tags: loli, catgirl, etc."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "top_p": 0.1,
        "thinking": {"type": "disabled"},
    }


async def main():
    """測試所有方案"""
    print("=" * 60)
    print("VLM Bug 修復測試")
    print("=" * 60)

    methods = [
        ("原始格式 + thinking disabled", make_payload_v1),
        ("簡單提示", make_payload_v2),
        ("System message", make_payload_v3),
        ("低 temperature", make_payload_v4),
    ]

    results = []
    for name, func in methods:
        success, content = await test_vlm_fix(name, func)
        results.append((name, success, content))

    # 總結
    print("\n" + "=" * 60)
    print("測試結果總結")
    print("=" * 60)

    working_methods = [(name, content) for name, success, content in results if success]

    if working_methods:
        print(f"\n[OK] 找到 {len(working_methods)} 個可行方案:")
        for name, content in working_methods:
            print(f"  - {name}")
    else:
        print("\n[FAIL] 所有方案都失敗")
        print("\n建議:")
        print("  1. 檢查 LM Studio 是否運行")
        print("  2. 嘗試重新加載 GLM-4.6v-flash 模型")
        print("  3. 更新 LM Studio 到最新版本")
        print("  4. 考慮使用替代模型 (Qwen2-VL)")


if __name__ == "__main__":
    asyncio.run(main())
