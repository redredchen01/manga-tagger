import torch
import sys


def check_environment():
    """檢查RTX 5090環境配置"""
    print("=== RTX 5090 環境檢查 ===")

    # 檢查Python版本
    print(f"Python版本: {sys.version}")

    # 檢查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")

    # 檢查CUDA可用性
    if torch.cuda.is_available():
        print("✅ CUDA 可用")

        # 檢查GPU設備
        device_count = torch.cuda.device_count()
        print(f"GPU數量: {device_count}")

        if device_count > 0:
            # 檢查GPU詳情
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU名稱: {gpu_props.name}")
            print(f"總VRAM: {gpu_props.total_memory / 1024**3:.1f}GB")

            # 檢查當前記憶體使用
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"當前VRAM使用: {current_memory:.2f}GB")

            return True
        else:
            print("❌ 未檢測到GPU設備")
            return False
    else:
        print("❌ CUDA 不可用")
        return False


def test_simple_models():
    """測試簡單模型加載"""
    try:
        from transformers import AutoTokenizer, AutoModel

        print("\n=== 測試模型加載 ===")

        # 測試最小的模型 - Qwen3-0.6B
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"嘗試加載: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("✅ Qwen3-0.6B 加載成功!")

        # 簡單測試
        test_text = "這是一個測試文本"
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]

        print(f"✅ 測試成功! 嵌入向量形狀: {embedding.shape}")

        # 清理
        del model, tokenizer
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ 模型測試失敗: {str(e)}")
        return False


def test_bge_m3_simple():
    """測試BGE-M3"""
    try:
        print("\n=== 測試BGE-M3 ===")

        # 檢查FlagEmbedding是否可用
        try:
            from FlagEmbedding import BGEM3FlagModel

            print("✅ FlagEmbedding 可用")
        except ImportError as e:
            print(f"❌ FlagEmbedding不可用: {e}")
            return False

        # 嘗試加載BGE-M3
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("✅ BGE-M3 加載成功!")

        # 簡單測試
        test_text = "這是一個測試文本"
        embeddings = model.encode([test_text], batch_size=1)

        print(f"✅ BGE-M3測試成功! 嵌入向量形狀: {embeddings.shape}")

        # 清理
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ BGE-M3測試失敗: {str(e)}")
        return False


if __name__ == "__main__":
    print("RTX 5090 中文嵌入模型環境驗證")
    print("====================================")

    # 環境檢查
    env_ok = check_environment()

    if not env_ok:
        print("❌ 環境檢查失敗，請先安裝CUDA驅動")
        sys.exit(1)

    # 模型測試
    success_count = 0
    total_tests = 2

    if test_simple_models():
        success_count += 1
        print("✅ Qwen3-0.6B 測試通過")

    if test_bge_m3_simple():
        success_count += 1
        print("✅ BGE-M3 測試通過")

    print(f"\n=== 測試結果 ===")
    print(f"通過測試: {success_count}/{total_tests}")

    if success_count == total_tests:
        print("🎉 所有測試通過! 您的RTX 5090環境已準備就緒!")
        print("\n推薦的模型配置:")
        print("- 輕量級應用: Qwen3-Embedding-0.6B (快速、低VRAM)")
        print("- 通用用途: BGE-M3 (多功能、穩定)")
        print("- 高精度需求: Qwen3-Embedding-4B (需要更多VRAM)")
    else:
        print("⚠️ 部分測試失敗，請檢查環境配置")
