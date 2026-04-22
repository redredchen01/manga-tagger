# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Background task processing with RQ
- WebSocket support for real-time updates
- Distributed tracing (Request ID, Correlation ID)
- Security headers middleware
- Response compression (GZip)
- Performance monitoring utilities
- Memory tracking decorator

### Changed
- Increased default API_WORKERS to 4
- Increased default MAX_CONCURRENT_REQUESTS to 10
- Parallelized VLM+RAG pipeline (40-50% latency reduction)

### Fixed
- Updated 4 dependencies to patch critical CVEs
- CORS wildcard default now returns empty list
- SENSITIVE_TAGS externalized to environment variable

### Security
- Pillow updated to 12.1.1 (CVE-2026-25990)
- torch updated to 2.10.0 (CVE-2026-24747)
- transformers updated to 4.53.0 (Multiple CVEs)
- python-multipart updated to 0.0.22 (CVE-2026-24486)

## [2.0.0] - 2026-03-26

### Added
- Structured JSON logging with structlog
- Prometheus metrics and /metrics endpoint
- Redis caching for VLM and RAG results
- Custom exception classes for better error handling
- Integration tests (14 tests)
- Unit tests for core services (65 tests)
- CI/CD pipeline with GitHub Actions

### Changed
- Refactored large files (reduced 19-62% code)
- Extracted prompts, tag_parser, pipeline modules
- Externalized SENSITIVE_TAGS to environment variable

### Fixed
- Fixed CORS wildcard default security issue
- Implemented JSON logging (was configured but not used)

## [1.0.0] - 2026-03-01

### Added
- Initial release
- FastAPI REST API
- LM Studio VLM/LLM integration
- ChromaDB RAG system
- Multi-stage tagging pipeline
- API key authentication
- Rate limiting middleware
- Streamlit frontend
