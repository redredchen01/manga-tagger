# Chinese Embedding Integration Documentation

## Overview

This document describes the integration of Chinese text embedding capabilities into the existing manga tagging system. The integration adds text-based tag search and hybrid search functionality while maintaining full backward compatibility.

## New Features

### 1. Chinese Embedding Service (`app/services/chinese_embedding_service.py`)

A new service that provides Chinese text embedding capabilities using sentence-transformers models.

**Key Features:**
- **Batch Processing**: Efficient encoding of multiple texts simultaneously
- **Similarity Calculations**: Cosine similarity between text embeddings
- **Tag Search**: Find relevant tags based on text similarity
- **Fallback Support**: Deterministic embeddings when model is unavailable
- **Async Support**: Non-blocking operations for better performance

**Model Used:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Supports Chinese and multiple languages
- 384-dimensional embeddings (automatically adjusted to 512 for compatibility)

### 2. Enhanced RAG Service (`app/services/rag_service.py`)

The existing RAG service has been enhanced with text-based search capabilities.

**New Methods:**

#### `search_tags_by_text()`
Search for tags by text similarity using Chinese embeddings.

```python
results = await rag_service.search_tags_by_text(
    query_text="一个有猫耳朵和尾巴的女孩",
    tag_list=["猫娘", "萝莉", "白发", "黑发"],
    top_k=10,
    threshold=0.3
)
```

#### `hybrid_search()`
Combine image similarity and text similarity for comprehensive search.

```python
results = await rag_service.hybrid_search(
    image_bytes=image_data,
    query_text="猫耳少女",
    tag_list=available_tags,
    image_top_k=5,
    text_top_k=10,
    image_weight=0.7,
    text_weight=0.3
)
```

**Returns:**
```python
{
    "image_results": [...],      # Traditional image similarity results
    "text_results": [...],       # Text-based tag similarity results
    "combined_results": [...]    # Merged and ranked results
}
```

### 3. Configuration Updates (`app/config.py`)

New configuration options have been added:

```python
# Chinese Embedding Settings
CHINESE_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
USE_CHINESE_EMBEDDINGS: bool = True
CHINESE_EMBEDDING_THRESHOLD: float = 0.3
CHINESE_EMBEDDING_TOP_K: int = 10

# Hybrid Search Settings
HYBRID_SEARCH_IMAGE_WEIGHT: float = 0.7
HYBRID_SEARCH_TEXT_WEIGHT: float = 0.3
HYBRID_SEARCH_ENABLED: bool = True
```

## Usage Examples

### Basic Text-Based Tag Search

```python
from app.services.rag_service import RAGService

rag_service = RAGService()

# Search tags by text description
results = await rag_service.search_tags_by_text(
    query_text="白发红眼的猫娘",
    tag_list=["猫娘", "白发", "红眼", "黑发", "蓝眼"],
    top_k=5,
    threshold=0.3
)

for result in results:
    print(f"Tag: {result['tag']}, Similarity: {result['similarity']:.3f}")
```

### Hybrid Search (Image + Text)

```python
# Combine image and text search
results = await rag_service.hybrid_search(
    image_bytes=cover_image_bytes,
    query_text="穿着校服的猫娘",
    tag_list=available_tags,
    image_top_k=5,
    text_top_k=10,
    image_weight=0.7,
    text_weight=0.3
)

# Access different result types
image_results = results["image_results"]
text_results = results["text_results"]
combined_results = results["combined_results"]
```

### Direct Chinese Embedding Service Usage

```python
from app.services.chinese_embedding_service import get_chinese_embedding_service

service = get_chinese_embedding_service()

# Encode texts
embeddings = await service.encode_batch(["猫娘", "萝莉", "白发"])

# Calculate similarity
similarity = await service.calculate_similarity("猫娘", "猫耳少女")

# Find similar texts
results = await service.find_most_similar(
    query_text="猫耳少女",
    candidate_texts=["猫娘", "萝莉", "白发", "机械"],
    top_k=3
)
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Chinese Embedding Settings
CHINESE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
USE_CHINESE_EMBEDDINGS=true
CHINESE_EMBEDDING_THRESHOLD=0.3
CHINESE_EMBEDDING_TOP_K=10

# Hybrid Search Settings
HYBRID_SEARCH_IMAGE_WEIGHT=0.7
HYBRID_SEARCH_TEXT_WEIGHT=0.3
HYBRID_SEARCH_ENABLED=true
```

### Model Requirements

The Chinese embedding service requires:
- `sentence-transformers` package
- PyTorch (automatically installed with sentence-transformers)
- Sufficient memory for the model (~500MB)

## Backward Compatibility

The integration maintains full backward compatibility:

1. **Existing API Endpoints**: All existing endpoints continue to work unchanged
2. **Image Search**: Traditional image-based search remains the default
3. **Optional Features**: Chinese embeddings can be disabled via configuration
4. **Fallback Support**: System gracefully degrades if Chinese embeddings are unavailable

### Migration Path

Existing code requires no changes. To use new features:

```python
# Existing code continues to work
results = await rag_service.search_similar(image_bytes, top_k=5)

# New features are opt-in
if settings.USE_CHINESE_EMBEDDINGS:
    text_results = await rag_service.search_tags_by_text(query_text, tags)
    hybrid_results = await rag_service.hybrid_search(image_bytes, query_text, tags)
```

## Performance Considerations

### Model Loading
- Chinese embedding model is loaded once at startup
- Model loading takes ~2-5 seconds depending on hardware
- Subsequent operations are fast (~10-50ms per text)

### Memory Usage
- Model requires ~500MB RAM
- Embeddings are cached during operations
- Thread pool prevents blocking

### GPU Acceleration
- Automatically uses CUDA if available
- Falls back to CPU if CUDA unavailable
- Can be forced to CPU with `DEVICE=cpu` in config

## Testing

### Run Integration Tests

```bash
python test_chinese_embedding_integration.py
```

The test script verifies:
- ✅ Service initialization
- ✅ Text encoding functionality
- ✅ Similarity calculations
- ✅ Tag search performance
- ✅ RAG service integration
- ✅ Hybrid search capabilities
- ✅ Backward compatibility
- ✅ Configuration validation

### Test Coverage

The integration includes comprehensive tests for:
- Normal operation scenarios
- Error handling and fallbacks
- Configuration validation
- Performance benchmarks
- Compatibility checks

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```
   Solution: Check internet connection and sentence-transformers installation
   pip install sentence-transformers --upgrade
   ```

2. **CUDA Out of Memory**
   ```
   Solution: Force CPU usage or reduce batch size
   DEVICE=cpu in .env file
   ```

3. **Chinese Text Not Supported**
   ```
   Solution: Ensure using multilingual model
   CHINESE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("app.services.chinese_embedding_service").setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor service performance:

```python
stats = chinese_service.get_stats()
print(f"Model: {stats['model_name']}")
print(f"Device: {stats['device']}")
print(f"Available: {stats['available']}")
```

## API Integration

### New API Endpoints (Optional)

While not required, you can add these endpoints to expose the new functionality:

```python
@router.post("/search-tags-by-text")
async def search_tags_by_text(request: TagSearchRequest):
    results = await rag_service.search_tags_by_text(
        query_text=request.query_text,
        tag_list=request.tag_list,
        top_k=request.top_k,
        threshold=request.threshold
    )
    return {"results": results}

@router.post("/hybrid-search")
async def hybrid_search(request: HybridSearchRequest):
    results = await rag_service.hybrid_search(
        image_bytes=request.image_bytes,
        query_text=request.query_text,
        tag_list=request.tag_list,
        image_top_k=request.image_top_k,
        text_top_k=request.text_top_k
    )
    return results
```

## Future Enhancements

### Potential Improvements

1. **Model Selection**: Support for different embedding models
2. **Caching**: Persistent caching of embeddings
3. **Fine-tuning**: Domain-specific model fine-tuning
4. **Multilingual**: Extended support for other languages
5. **Performance**: Optimized batch processing

### Extension Points

The service is designed for easy extension:

```python
class CustomEmbeddingService(ChineseEmbeddingService):
    def __init__(self):
        super().__init__()
        # Custom initialization
    
    async def custom_encode_method(self, texts):
        # Custom encoding logic
        pass
```

## Conclusion

The Chinese embedding integration successfully adds text-based tag search and hybrid search capabilities to the manga tagging system while maintaining full backward compatibility. The implementation follows existing code patterns, includes comprehensive error handling, and provides extensive configuration options.

The system now supports:
- ✅ Traditional image-based similarity search
- ✅ Text-based tag similarity search
- ✅ Hybrid search combining both approaches
- ✅ Graceful fallbacks and error handling
- ✅ Comprehensive configuration options
- ✅ Full backward compatibility

This enhancement significantly improves the tagging system's ability to find relevant tags based on both visual and textual similarity, providing more accurate and comprehensive tagging results.