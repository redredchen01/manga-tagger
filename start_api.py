import sys

# Import settings for API host and port configuration
from app.core.config import settings

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info",
    )
