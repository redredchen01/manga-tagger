import torch
from FlagEmbedding import FlagModel
import time
import numpy as np

print("=== Testing BGE-M3 Model ===")

try:
    # Clear GPU memory
    torch.cuda.empty_cache()

    print("Loading BGE-M3 model...")
    start_time = time.time()

    # Initialize BGE-M3 model
    model = FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    # Check VRAM usage
    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM usage: {vram_used:.2f} GB / {vram_total:.2f} GB")

    # Test Chinese texts
    chinese_texts = [
        "这是一个测试文本",
        "深度学习模型",
        "自然语言处理技术",
        "人工智能发展",
        "机器学习算法优化",
    ]

    print(f"\nTesting {len(chinese_texts)} Chinese texts...")

    # Benchmark encoding
    times = []
    for i, text in enumerate(chinese_texts):
        torch.cuda.synchronize()
        start_time = time.time()

        # Encode using BGE-M3
        embeddings = model.encode(text)

        torch.cuda.synchronize()
        end_time = time.time()
        encoding_time = (end_time - start_time) * 1000  # Convert to ms

        times.append(encoding_time)
        print(f'Text {i + 1}: "{text}"')
        print(f"  Encoding time: {encoding_time:.2f} ms")
        print(f"  Embedding shape: {embeddings.shape}")

    # Performance summary
    avg_time = np.mean(times)
    vram_final = torch.cuda.memory_allocated(0) / 1024**3
    print(f"\n=== BGE-M3 Performance Summary ===")
    print(f"Average encoding time: {avg_time:.2f} ms")
    print(f"VRAM usage: {vram_final:.2f} GB")
    print(f"Throughput: {1000 / avg_time:.1f} texts/second")
    print(f"Model supports Chinese: ✓")
    print("BGE-M3 test completed successfully!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
