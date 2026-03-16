import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np

print("=== Testing Embedding Models on CPU (RTX 5090 Compatibility Issue) ===")

models_to_test = [
    ("intfloat/multilingual-e5-large", "E5-Multilingual-Large"),
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6-v2"),
]

for model_name, display_name in models_to_test:
    try:
        print(f"\n=== Testing {display_name} on CPU ===")

        # Load model on CPU (due to RTX 5090 CUDA compatibility)
        print(f"Loading {display_name}...")
        start_time = time.time()

        model = SentenceTransformer(model_name, device="cpu")
        model.eval()

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # Test Chinese texts
        chinese_texts = [
            "这是一个测试文本",
            "深度学习模型",
            "自然语言处理技术",
            "人工智能发展",
            "机器学习算法优化",
        ]

        print(f"Testing {len(chinese_texts)} Chinese texts...")

        # Benchmark encoding
        times = []
        for i, text in enumerate(chinese_texts):
            start_time = time.time()

            # Encode using SentenceTransformer
            embeddings = model.encode(text, convert_to_tensor=True)

            end_time = time.time()
            encoding_time = (end_time - start_time) * 1000  # Convert to ms

            times.append(encoding_time)
            print(f'  Text {i + 1}: "{text[:10]}..." - {encoding_time:.2f} ms')

        # Performance summary
        avg_time = np.mean(times)
        print(f"{display_name} Summary (CPU):")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Throughput: {1000 / avg_time:.1f} texts/second")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Success: ✓")

    except Exception as e:
        print(f"Error with {display_name}: {e}")
        continue

print("\n=== Recommendations for RTX 5090 ===")
print("1. CUDA Compatibility Issue: RTX 5090 uses sm_120 capability")
print("2. Current PyTorch 2.5.1 supports up to sm_90")
print("3. Wait for PyTorch update with RTX 5090 support")
print("4. In meantime, use CPU mode or older GPU models")
print("\n=== CPU Testing Complete ===")
