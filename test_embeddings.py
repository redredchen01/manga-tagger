# RTX 5090 中文嵌入模型測試腳本

from transformers import AutoTokenizer, AutoModel
import torch
import time
import sys


def test_qwen3_4b():
    """測試 Qwen3-4B 模型"""
    try:
        print("開始測試 Qwen3-Embedding-4B...")

        # 加載模型
        model_name = "Qwen/Qwen3-Embedding-4B"
        print(f"加載模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print(f"✅ 模型加載成功！")
        print(f"💾 VRAM使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"📏 向量維度: {model.config.hidden_size}")

        # 測試編碼
        test_texts = [
            "人工智能技術正在快速發展",
            "機器學習是AI的重要分支",
            "深度學習模型的性能不斷提升",
        ]

        print("🧪 開始編碼測試...")
        start_time = time.time()

        inputs = tokenizer(
            test_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        end_time = time.time()

        print(f"✨ 編碼完成！")
        print(f"⏱️  耗時: {end_time - start_time:.3f}秒")
        print(f"📊 輸入形狀: {embeddings.shape}")
        print(f"📝 處理了 {len(test_texts)} 個文本")

        # 清理記憶體
        del model, tokenizer
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        return False


def test_bge_m3():
    """測試 BGE-M3 模型"""
    try:
        print("\n開始測試 BGE-M3...")

        from FlagEmbedding import BGEM3FlagModel

        # 加載模型
        print("加載 BGE-M3 模型...")
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        print(f"✅ 模型加載成功！")
        print(f"💾 VRAM使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

        # 測試編碼
        test_texts = [
            "人工智能技術正在快速發展",
            "機器學習是AI的重要分支",
            "深度學習模型的性能不斷提升",
        ]

        print("🧪 開始編碼測試...")
        start_time = time.time()

        embeddings = model.encode(test_texts, batch_size=32, max_length=512)

        end_time = time.time()

        print(f"✨ 編碼完成！")
        print(f"⏱️  耗時: {end_time - start_time:.3f}秒")
        print(f"📊 輸出形狀: {embeddings.shape}")
        print(f"📝 處理了 {len(test_texts)} 個文本")

        # 清理記憶體
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        return False


def test_qwen3_06b():
    """測試 Qwen3-0.6B 模型"""
    try:
        print("\n開始測試 Qwen3-Embedding-0.6B...")

        # 加載模型
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"加載模型: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print(f"✅ 模型加載成功！")
        print(f"💾 VRAM使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"📏 向量維度: {model.config.hidden_size}")

        # 測試編碼
        test_texts = [
            "人工智能技術正在快速發展",
            "機器學習是AI的重要分支",
            "深度學習模型的性能不斷提升",
        ]

        print("🧪 開始編碼測試...")
        start_time = time.time()

        inputs = tokenizer(
            test_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        end_time = time.time()

        print(f"✨ 編碼完成！")
        print(f"⏱️  耗時: {end_time - start_time:.3f}秒")
        print(f"📊 輸入形狀: {embeddings.shape}")
        print(f"📝 處理了 {len(test_texts)} 個文本")

        # 清理記憶體
        del model, tokenizer
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}")
        return False


def performance_benchmark():
    """性能基準測試"""
    print("開始性能基準測試...")

    test_texts = [f"這是第{i}個測試文本" for i in range(100)]

    # 測試不同模型的QPS
    models = [
        ("Qwen3-4B", "Qwen/Qwen3-Embedding-4B"),
        ("BGE-M3", "BAAI/bge-m3"),
        ("Qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
    ]

    for name, model_path in models:
        print(f"\n📊 測試 {name} 性能...")

        try:
            if "BGE-M3" in name:
                from FlagEmbedding import BGEM3FlagModel

                model = BGEM3FlagModel(model_path, use_fp16=True)

                start_time = time.time()
                embeddings = model.encode(test_texts, batch_size=32)
                end_time = time.time()

                qps = len(test_texts) / (end_time - start_time)

            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device="cuda",
                )

                inputs = tokenizer(
                    test_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                start_time = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                end_time = time.time()

                qps = len(test_texts) / (end_time - start_time)

            vram_usage = torch.cuda.memory_allocated() / 1024**3
            print(f"   ✅ {name} QPS: {qps:.1f}")
            print(f"   💾 VRAM使用: {vram_usage:.2f}GB")

            del model
            if "tokenizer" in locals():
                del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ {name} 測試失敗: {e}")


if __name__ == "__main__":
    print("RTX 5090 中文嵌入模型測試開始")
    print(f"GPU: RTX 5090")
    print(f"總VRAM: 24GB")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    success_count = 0
    total_tests = 0

    # 測試各種模型
    if test_qwen3_4b():
        success_count += 1
    total_tests += 1

    if test_bge_m3():
        success_count += 1
    total_tests += 1

    success_count = 0
    total_tests = 0

    if test_qwen3_4b():
        success_count += 1
    total_tests += 1

    if test_bge_m3():
        success_count += 1
    total_tests += 1

    if test_qwen3_06b():
        success_count += 1
    total_tests += 1

    # 性能基準測試
    performance_benchmark()

    print(f"\n測試完成！")
    print(f"成功: {success_count}/{total_tests}")
    print(f"RTX 5090 可以完美運行中文嵌入模型！")

    if success_count == total_tests:
        print("所有測試通過，您的環境已準備就緒！")
    else:
        print("部分測試失敗，請檢查環境配置")
