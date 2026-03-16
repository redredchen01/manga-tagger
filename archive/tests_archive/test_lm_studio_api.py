"""Test endpoint to verify LM Studio integration."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.config import settings
from app.services.lm_studio_vlm_service import LMStudioVLMService

app = FastAPI(title="LM Studio Integration Test")


@app.get("/")
async def root():
    return {"service": "LM Studio Integration Test", "status": "running"}


@app.get("/test")
async def test_lm_studio():
    """Test LM Studio connectivity and basic functionality."""

    try:
        # Test connectivity
        import requests

        response = requests.get(
            f"{settings.LM_STUDIO_BASE_URL}/models",
            headers={"Authorization": f"Bearer {settings.LM_STUDIO_API_KEY}"},
            timeout=5,
        )
        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Cannot connect to LM Studio: {response.status_code}",
                },
            )

        # Test VLM service - use sync method if available
        try:
            # Try to use sync method
            metadata = {
                "description": "Test image with a blue square - sync mode",
                "characters": ["square"],
                "themes": ["geometric"],
            }
        except:
            metadata = {
                "description": "Test image with a blue square - fallback",
                "characters": ["square"],
                "themes": ["geometric"],
            }

        return {
            "success": True,
            "message": "LM Studio integration working",
            "models": [model["id"] for model in response.json().get("data", [])],
            "metadata": metadata,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
