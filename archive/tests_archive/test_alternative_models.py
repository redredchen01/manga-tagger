import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np

print("=== Testing Alternative Embedding Models ===")

models_to_test = [
    ("intfloat/multilingual-e5-large", "E5-Multilingual-Large"),
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6-v2"),
    (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "Paraphrase-MiniLM-L12-v2",
    ),
]

for model_name, display_name in models_to_test:
    try:
        print(f"\n=== Testing {display_name} ===")

        # Clear GPU memory
        torch.cuda.empty_cache()

        print(f"Loading {display_name}...")
        start_time = time.time()

        # Load model
        model = SentenceTransformer(model_name)
        model = model.cuda()
        model.eval()

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        # Check VRAM usage
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM usage: {vram_used:.2f} GB / {vram_total:.2f} GB")

        # Test Chinese texts
        chinese_texts = ["这是一个测试文本", "深度学习模型", "自然语言处理技术"]

        print(f"Testing {len(chinese_texts)} Chinese texts...")

        # Benchmark encoding
        times = []
        for i, text in enumerate(chinese_texts):
            torch.cuda.synchronize()
            start_time = time.time()

            # Encode using SentenceTransformer
            with torch.no_grad():
                embeddings = model.encode(text, convert_to_tensor=True)

            torch.cuda.synchronize()
            end_time = time.time()
            encoding_time = (end_time - start_time) * 1000  # Convert to ms

            times.append(encoding_time)
            print(f'  Text {i + 1}: "{text[:10]}..." - {encoding_time:.2f} ms')

        # Performance summary
        avg_time = np.mean(times)
        vram_final = torch.cuda.memory_allocated(0) / 1024**3
        print(f"{display_name} Summary:")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  VRAM usage: {vram_final:.2f} GB")
        print(f"  Throughput: {1000 / avg_time:.1f} texts/second")
        print(f"  Success: ✓")

        # Clean up
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error with {display_name}: {e}")
        continue

print("\n=== Alternative Model Testing Complete ===")
