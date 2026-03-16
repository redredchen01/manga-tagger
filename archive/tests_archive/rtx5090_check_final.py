import torch
import sys


def check_environment():
    """Check RTX 5090 environment configuration"""
    print("=== RTX 5090 Environment Check ===")

    # Check Python version
    print(f"Python Version: {sys.version}")

    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA Available: YES")

        # Check GPU devices
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")

        if device_count > 0:
            # Check GPU details
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU Name: {gpu_props.name}")
            print(f"Total VRAM: {gpu_props.total_memory / 1024**3:.1f}GB")

            # Check current memory usage
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Current VRAM Usage: {current_memory:.2f}GB")

            return True
        else:
            print("ERROR: No GPU devices detected")
            return False
    else:
        print("ERROR: CUDA not available")
        return False


def test_simple_models():
    """Test simple model loading"""
    try:
        from transformers import AutoTokenizer, AutoModel

        print("\n=== Testing Model Loading ===")

        # Test smallest model - Qwen3-0.6B
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"Trying to load: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("SUCCESS: Qwen3-0.6B loaded successfully!")

        # Simple test
        test_text = "This is a test text"
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

        print(f"SUCCESS: Test completed! Embedding shape: {embedding.shape}")

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"ERROR: Model test failed: {str(e)}")
        return False


def test_bge_m3_simple():
    """Test BGE-M3"""
    try:
        print("\n=== Testing BGE-M3 ===")

        # Check FlagEmbedding availability
        try:
            from FlagEmbedding import BGEM3FlagModel

            print("SUCCESS: FlagEmbedding available")
        except ImportError as e:
            print(f"ERROR: FlagEmbedding not available: {e}")
            return False

        # Try loading BGE-M3
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("SUCCESS: BGE-M3 loaded successfully!")

        # Simple test
        test_text = "This is a test text"
        embeddings = model.encode([test_text], batch_size=1)

        print(f"SUCCESS: BGE-M3 test completed! Embedding shape: {embeddings.shape}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"ERROR: BGE-M3 test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("RTX 5090 Chinese Embedding Models Environment Verification")
    print("=============================================")

    # Environment check
    env_ok = check_environment()

    if not env_ok:
        print("ERROR: Environment check failed, please install CUDA drivers first")
        sys.exit(1)

    # Model tests
    success_count = 0
    total_tests = 2

    if test_simple_models():
        success_count += 1
        print("SUCCESS: Qwen3-0.6B test passed")

    if test_bge_m3_simple():
        success_count += 1
        print("SUCCESS: BGE-M3 test passed")

    print(f"\n=== Test Results ===")
    print(f"Passed Tests: {success_count}/{total_tests}")

    if success_count == total_tests:
        print("EXCELLENT: All tests passed! Your RTX 5090 environment is ready!")
        print("\nRecommended model configurations:")
        print("- Lightweight applications: Qwen3-Embedding-0.6B (fast, low VRAM)")
        print("- General purpose: BGE-M3 (multi-functional, stable)")
        print("- High accuracy needs: Qwen3-Embedding-4B (requires more VRAM)")
        print("\nNext steps:")
        print("1. Install models for production use")
        print("2. Set up API service")
        print("3. Configure RAG system")
        print("4. Start embedding your content!")
    else:
        print("WARNING: Some tests failed, please check environment configuration")
