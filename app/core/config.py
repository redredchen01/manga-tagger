"""Configuration management for Manga Tagger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import business constants from domain layer
from app.domain.constants import (
    TAG_FREQUENCY_CALIBRATION,
    SEMANTIC_SIBLINGS,
    MIN_ACCEPTABLE_CONFIDENCE,
    EXACT_MATCH_BOOST,
    PARTIAL_MATCH_BOOST,
    RAG_SUPPORT_BOOST,
    RAG_SUPPORT_DECAY,
    SEMANTIC_MATCH_PENALTY,
    SENSITIVE_TAGS,
    RAG_SIMILARITY_THRESHOLD,
    EXACT_MATCH_PENALTY,
    MUTUAL_EXCLUSIVITY,
    TAG_HIERARCHY,
    VISUAL_FEATURE_BOOST,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_KEY: str | None = None
    DEBUG: bool = False
    USE_MOCK_SERVICES: bool = False

    # LM Studio Configuration (NEW - Recommended)
    # Note: LM_STUDIO_API_KEY must be set via environment variable - no hardcoded default for security
    LM_STUDIO_API_KEY: str | None = None
    LM_STUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LM_STUDIO_VISION_MODEL: str = "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"
    LM_STUDIO_TEXT_MODEL: str = "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"
    LM_STUDIO_EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B-GGUF"
    LM_STUDIO_EMBEDDING_DIM: int = 4096
    USE_LM_STUDIO: bool = True

    # Ollama Configuration (Alternative to LM Studio)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_VISION_MODEL: str = "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"
    USE_OLLAMA: bool = False

    VLM_MAX_TOKENS: int = 2048
    LLM_MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    DEVICE: str = "cuda"

    VLM_MODEL: str = "Qwen/Qwen2-VL-7B-Instruct"
    EMBEDDING_MODEL: str = "openai/clip-vit-large-patch14"
    LLM_MODEL: str = "meta-llama/Llama-3.2-8B-Instruct"

    QUANTIZATION_CONFIG: str | None = None

    CHROMA_DB_PATH: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "manga_covers"
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = RAG_SIMILARITY_THRESHOLD
    LEXICAL_MATCH_THRESHOLD: float = 0.6

    TAG_LIBRARY_PATH: str = "./data/tags.json"

    BATCH_SIZE: int = 1
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 120

    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_BURST: int = 10

    # Redis Configuration
    REDIS_ENABLED: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    CACHE_TTL: int = 3600  # 1 hour

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    CORS_ORIGINS: str | list[str] = []

    # Security headers configuration
    SECURITY_HEADERS_ENABLED: bool = True

    # GZip compression configuration
    GZIP_MINIMUM_SIZE: int = 500

    CHINESE_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    CHINESE_EMBEDDING_DIM: int = 1024
    USE_CHINESE_EMBEDDINGS: bool = True
    CHINESE_EMBEDDING_THRESHOLD: float = 0.75
    CHINESE_EMBEDDING_TOP_K: int = 10
    SEMANTIC_FALLBACK_TRIGGER_COUNT: int = 3  # only run semantic if VLM gave fewer than this
    SEMANTIC_FALLBACK_MAX_ADDITIONS: int = 2  # cap how many semantic tags to add
    RAG_INFLUENCE_ENABLED: bool = False  # Phase 1: RAG library too small to trust for scoring

    HYBRID_SEARCH_IMAGE_WEIGHT: float = 0.7
    HYBRID_SEARCH_TEXT_WEIGHT: float = 0.3
    HYBRID_SEARCH_ENABLED: bool = True

    GLM_BEGIN_TOKEN: str = "<|begin_of_text|>"
    GLM_END_TOKEN: str = "<|end_of_text|>"
    GLM_SPECIAL_TOKENS: list[str] = Field(
        default_factory=lambda: ["<|begin_of_text|>", "<|end_of_text|>"]
    )

    MIN_CONFIDENCE_THRESHOLD: float = 0.55
    USE_STRICT_PRECISION: bool = True

    # Performance tuning for Python 3.12+
    MAX_IMAGE_SIZE_MB: int = 10
    VLM_TIMEOUT_SECONDS: int = 300  # qwen3.6-35b-a3b on consumer GPU + 2048-token prompt
    RAG_TIMEOUT_SECONDS: int = 60
    TAG_RECOMMENDATION_TIMEOUT_SECONDS: int = 20
    MAX_CONCURRENT_VLM_CALLS: int = 1  # Reduce concurrency to ensure stability on consumer GPUs
    MAX_CONCURRENT_RAG_CALLS: int = 10

    # Confidence scoring constants - reference domain constants
    MIN_ACCEPTABLE_CONFIDENCE: float = MIN_ACCEPTABLE_CONFIDENCE
    EXACT_MATCH_BOOST: float = EXACT_MATCH_BOOST
    PARTIAL_MATCH_BOOST: float = PARTIAL_MATCH_BOOST
    RAG_SUPPORT_BOOST: float = RAG_SUPPORT_BOOST
    RAG_SUPPORT_DECAY: float = RAG_SUPPORT_DECAY
    SEMANTIC_MATCH_PENALTY: float = SEMANTIC_MATCH_PENALTY
    EXACT_MATCH_PENALTY: dict[str, float] = Field(
        default_factory=lambda: EXACT_MATCH_PENALTY.copy()
    )
    MUTUAL_EXCLUSIVITY: dict[str, set[str]] = Field(
        default_factory=lambda: {k: v.copy() for k, v in MUTUAL_EXCLUSIVITY.items()}
    )
    TAG_HIERARCHY: dict[str, str] = Field(default_factory=lambda: TAG_HIERARCHY.copy())
    VISUAL_FEATURE_BOOST: dict[str, float] = Field(
        default_factory=lambda: VISUAL_FEATURE_BOOST.copy()
    )

    # Sensitive tags - allow override via environment variable
    SENSITIVE_TAGS: str = Field(
        default=",".join(sorted(SENSITIVE_TAGS)),
        description="Comma-separated list of sensitive tags (override via SENSITIVE_TAGS env var)",
    )

    @computed_field
    @property
    def sensitive_tags(self) -> set[str]:
        return {t.strip() for t in self.SENSITIVE_TAGS.split(",") if t.strip()}

    # Tag matching constants - reference domain constants
    TAG_FREQUENCY_CALIBRATION: dict[str, float] = Field(
        default_factory=lambda: TAG_FREQUENCY_CALIBRATION.copy()
    )

    SEMANTIC_SIBLINGS: dict[str, set[str]] = Field(
        default_factory=lambda: {k: v.copy() for k, v in SEMANTIC_SIBLINGS.items()}
    )

    def _resolve_project_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def model_post_init(self, __context: Any) -> None:
        self.CHROMA_DB_PATH = str(self._resolve_project_path(self.CHROMA_DB_PATH))
        self.TAG_LIBRARY_PATH = str(self._resolve_project_path(self.TAG_LIBRARY_PATH))

    @property
    def chroma_db_path(self) -> Path:
        return self._resolve_project_path(self.CHROMA_DB_PATH)

    @property
    def tag_library_path(self) -> Path:
        return self._resolve_project_path(self.TAG_LIBRARY_PATH)

    @property
    def cors_origins(self) -> list[str]:
        value = self.CORS_ORIGINS
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def ensure_directories(self) -> None:
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "data" / "rag_dataset").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

    def as_runtime_dict(self) -> dict[str, Any]:
        """Return normalized settings for debugging/logging."""
        data = self.model_dump()
        data["CHROMA_DB_PATH"] = str(self.chroma_db_path)
        data["TAG_LIBRARY_PATH"] = str(self.tag_library_path)
        data["CORS_ORIGINS"] = self.cors_origins
        return data


settings = Settings()
