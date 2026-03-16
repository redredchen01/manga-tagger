"""Configuration management for Manga Tagger."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    DEBUG: bool = False
    USE_MOCK_SERVICES: bool = False  # Set to True to use mock VLM for testing

    # LM Studio Configuration
    LM_STUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LM_STUDIO_API_KEY: str = "lm-studio"
    LM_STUDIO_VISION_MODEL: str = "zai-org/glm-4.6v-flash"
    LM_STUDIO_TEXT_MODEL: str = "qwen/qwen3-coder-next"
    USE_LM_STUDIO: bool = True

    # Model Inference Settings
    VLM_MAX_TOKENS: int = 2048
    LLM_MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    DEVICE: str = "cuda"

    # Legacy Model Configuration (for local models - disabled when USE_LM_STUDIO=true)
    VLM_MODEL: str = "Qwen/Qwen2-VL-7B-Instruct"
    EMBEDDING_MODEL: str = "openai/clip-vit-large-patch14"
    LLM_MODEL: str = "meta-llama/Llama-3.2-8B-Instruct"

    # Quantization Settings
    QUANTIZATION_CONFIG: Optional[str] = None

    # RAG / ChromaDB Settings
    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "manga_covers"
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.50  # Raised for better precision (was 0.25)
    LEXICAL_MATCH_THRESHOLD: float = 0.6  # Threshold for keyword-to-tag matching

    # Tag Library
    TAG_LIBRARY_PATH: str = "./data/tags.json"

    # Performance Settings
    BATCH_SIZE: int = 1
    MAX_CONCURRENT_REQUESTS: int = 1
    REQUEST_TIMEOUT: int = 120

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Chinese Embedding Settings
    CHINESE_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHINESE_EMBEDDING_DIM: int = 1024
    USE_CHINESE_EMBEDDINGS: bool = True
    CHINESE_EMBEDDING_THRESHOLD: float = 0.50  # Raised for better precision (was 0.4)
    CHINESE_EMBEDDING_TOP_K: int = 10

    # Hybrid Search Settings
    HYBRID_SEARCH_IMAGE_WEIGHT: float = 0.7
    HYBRID_SEARCH_TEXT_WEIGHT: float = 0.3
    HYBRID_SEARCH_ENABLED: bool = True

    # Special tokens for GLM model responses
    GLM_BEGIN_TOKEN: str = "<|begin_of_text|>"
    GLM_END_TOKEN: str = "<|end_of_text|>"
    GLM_SPECIAL_TOKENS: list = [GLM_BEGIN_TOKEN, GLM_END_TOKEN]

    # Precision Settings (strict tagging mode)
    MIN_CONFIDENCE_THRESHOLD: float = (
        0.55  # Global minimum - tags below this are rejected
    )
    USE_STRICT_PRECISION: bool = True  # Enable all precision optimizations

    @property
    def chroma_db_path(self) -> Path:
        """Get ChromaDB path as Path object."""
        return Path(self.CHROMA_DB_PATH)

    @property
    def tag_library_path(self) -> Path:
        """Get tag library path as Path object."""
        return Path(self.TAG_LIBRARY_PATH)

    def ensure_directories(self):
        """Ensure all necessary directories exist."""
        Path(self.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        Path("./data/rag_dataset").mkdir(parents=True, exist_ok=True)
        Path("./models").mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
