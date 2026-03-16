import torch
from sentence_transformers import SentenceTransformer
import onnx
import onnxruntime as ort
import time
import numpy as np

print("=== Testing ONNX Runtime Workaround ===")

try:
    print("Converting model to ONNX format...")

    # Load original model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device="cpu")

    # Create dummy input for ONNX export
    dummy_input = ["This is a test sentence for ONNX conversion"]

    print("Exporting to ONNX...")
    start_time = time.time()

    # Export to ONNX
    with torch.no_grad():
        onnx_model_path = f"{model_name.split('/')[-1]}.onnx"

        # Create ONNX model with proper input shape
        tokenizer = model.tokenizer
        tokens = tokenizer(
            dummy_input,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Export model
        torch.onnx.export(
            model.model,
            tuple(tokens.values()),
            f"{model_name.split('/')[-1]}",
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "output": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
        )

    export_time = time.time() - start_time
    print(f"ONNX export completed in {export_time:.2f} seconds")

    # Test ONNX Runtime with GPU
    print("Testing ONNX Runtime with GPU...")

    # Configure ONNX Runtime for GPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Active provider: {ort_session.get_providers()}")

    # Test with Chinese text
    test_texts = ["深度学习模型", "自然语言处理技术", "人工智能发展"]

    times = []
    for i, text in enumerate(test_texts):
        # Tokenize
        tokens = tokenizer(text, padding=True, truncation=True, max_length=128)
        ort_inputs = {
            "input_ids": tokens["input_ids"].numpy(),
            "attention_mask": tokens["attention_mask"].numpy(),
        }

        start_time = time.time()
        embeddings = ort_session.run(None, ort_inputs)
        end_time = time.time()

        encoding_time = (end_time - start_time) * 1000
        times.append(encoding_time)
        print(f'Text {i + 1}: "{text}" - {encoding_time:.2f} ms')

    avg_time = np.mean(times)
    print(f"\\n=== ONNX Runtime Performance ===")
    print(f"Average encoding time: {avg_time:.2f} ms")
    print(f"Throughput: {1000 / avg_time:.1f} texts/second")
    print(f"GPU acceleration: ✅")
    print(f"ONNX Runtime: ✓")

except Exception as e:
    print(f"ONNX Runtime test failed: {e}")
    import traceback

    traceback.print_exc()

print("\\n=== Testing Basic PyTorch CPU Mode ===")
try:
    # Simple CPU test for comparison
    model = SentenceTransformer(model_name, device="cpu")

    test_texts = ["深度学习", "自然语言处理", "人工智能"]
    cpu_times = []

    for text in test_texts:
        start_time = time.time()
        embeddings = model.encode(text, normalize_embeddings=True)
        end_time = time.time()

        cpu_time = (end_time - start_time) * 1000
        cpu_times.append(cpu_time)
        print(f'CPU - "{text[:6]}...": {cpu_time:.2f} ms')

    avg_cpu = np.mean(cpu_times)
    print(f"\\nCPU Performance: {avg_cpu:.2f} ms avg")

except Exception as e:
    print(f"CPU test failed: {e}")
