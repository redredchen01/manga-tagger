"""Prometheus metrics for monitoring the Manga Cover Auto-Tagger."""

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency in seconds", ["method", "endpoint"]
)

# Pipeline metrics
VLM_REQUEST_COUNT = Counter("vlm_requests_total", "Total VLM requests", ["status"])

VLM_LATENCY = Histogram("vlm_request_duration_seconds", "VLM request latency in seconds")

RAG_REQUEST_COUNT = Counter("rag_requests_total", "Total RAG requests", ["status"])

RAG_LATENCY = Histogram("rag_request_duration_seconds", "RAG request latency in seconds")

TAG_RECOMMENDATION_COUNT = Counter(
    "tag_recommendations_total", "Total tag recommendations", ["status"]
)

TAG_RECOMMENDATION_LATENCY = Histogram(
    "tag_recommendation_duration_seconds", "Tag recommendation latency in seconds"
)

# Cache metrics
CACHE_HITS = Counter("cache_hits_total", "Cache hits", ["cache_type"])

CACHE_MISSES = Counter("cache_misses_total", "Cache misses", ["cache_type"])

# Gauge metrics
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")
