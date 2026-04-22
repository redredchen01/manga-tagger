"""WebSocket endpoint for real-time job updates."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.interfaces.websocket.connection_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates.

    Connect to receive progress updates for a specific job.

    Example:
        wscat -c ws://localhost:8000/api/v1/ws/jobs/your-job-id
    """
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive - can receive client messages if needed
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
