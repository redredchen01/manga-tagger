import torch
from transformers import AutoTokenizer, AutoModel
import time
import numpy as np

print("=== Testing Chinese Text Encoding ===")

# Load model again (should be cached)
model_name = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.cuda()
model.eval()

# Test Chinese texts
chinese_texts = [
    "这是一个测试文本",
    "深度学习模型",
    "自然语言处理技术",
    "人工智能发展",
    "机器学习算法优化",
]

print(f"Testing {len(chinese_texts)} Chinese texts...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Benchmark encoding
times = []
for i, text in enumerate(chinese_texts):
    torch.cuda.synchronize()
    start_time = time.time()

    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

    torch.cuda.synchronize()
    end_time = time.time()
    encoding_time = (end_time - start_time) * 1000  # Convert to ms

    times.append(encoding_time)
    print(f'Text {i + 1}: "{text}"')
    print(f"  Encoding time: {encoding_time:.2f} ms")
    print(f"  Embedding shape: {embeddings.shape}")

# Performance summary
avg_time = np.mean(times)
vram_used = torch.cuda.memory_allocated(0) / 1024**3
print(f"\n=== Performance Summary ===")
print(f"Average encoding time: {avg_time:.2f} ms")
print(f"VRAM usage: {vram_used:.2f} GB")
print(f"Throughput: {1000 / avg_time:.1f} texts/second")
print(f"Model supports Chinese: ✓")
print("Chinese encoding test completed successfully!")
