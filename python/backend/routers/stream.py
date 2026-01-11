"""HLS streaming for RGB + Depth side-by-side video.

This module provides HLS streaming for processed video with:
- Left half: Original RGB video
- Right half: RGB-packed 16-bit depth data (R=high byte, G=low byte, B=128)

The HLS stream is standards-compliant and can be played by any HLS-compatible player.
The depth is unpacked on the client using the metadata (z_min, z_max, scale).

Note: The legacy VDZ WebSocket streaming has been removed in favor of HLS.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Depends,
)
from pydantic import BaseModel

from backend.config import get_settings
from backend.video.hls_generator import (
    HlsGenerator,
    HlsMetadata,
    get_hls_generator,
    register_hls_generator,
    unregister_hls_generator,
)
from backend.video.session import get_session_manager, SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["stream"])


# =============================================================================
# HLS Endpoints (New)
# =============================================================================


class HlsRequest(BaseModel):
    """Request model for HLS generation."""

    fps: float = 30.0
    segment_duration: float = 2.0
    process_res: int = 640


@router.post("/{session_id}/hls")
async def create_hls_stream(
    session_id: str,
    request: HlsRequest,
    background_tasks: BackgroundTasks,
    manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Start HLS stream generation for a session.

    This processes the video to generate a side-by-side HLS stream with:
    - Left half: Original RGB video
    - Right half: RGB-packed 16-bit depth data

    Returns immediately with a status URL. Use GET /hls/status to poll for completion.
    """
    session = await manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    settings = get_settings()
    data_root = settings.data_root

    # Create output directory
    hls_output_dir = data_root / session_id / "hls"
    if hls_output_dir.exists():
        shutil.rmtree(hls_output_dir)
    hls_output_dir.mkdir(parents=True)

    # Create generator
    generator = HlsGenerator(
        session_id=session_id,
        source_path=session.source_path,
        output_dir=hls_output_dir,
        fps=request.fps,
        segment_duration=request.segment_duration,
        process_res=request.process_res,
    )

    # Register generator
    register_hls_generator(session_id, generator)

    # Start generation in background
    def run_generation():
        import asyncio

        try:
            asyncio.run(generator.generate())
        except Exception as e:
            logger.error(f"HLS generation failed for session {session_id}: {e}")
            with generator._lock:
                generator.state.status = "error"
                generator.state.error_message = str(e)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    return {
        "session_id": session_id,
        "status_url": f"/api/sessions/{session_id}/hls/status",
        "message": "HLS generation started",
    }


@router.get("/{session_id}/hls/status")
async def get_hls_status(session_id: str) -> dict:
    """Get HLS generation status."""
    generator = get_hls_generator(session_id)
    if not generator:
        # Check if there's a completed HLS stream
        settings = get_settings()
        metadata_path = settings.data_root / session_id / "hls" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                "status": "ready",
                "progress": 1.0,
                **metadata,
            }
        raise HTTPException(status_code=404, detail="HLS stream not found")

    state = generator.get_state()
    response = {
        "status": state.status,
        "progress": state.progress,
        "frame_count": state.frame_count,
        "total_frames": state.total_frames,
        "eta_seconds": state.eta_seconds,
        "frames_per_second": state.frames_per_second,
    }

    if state.error_message:
        response["error"] = state.error_message

    if state.status == "ready":
        # Load metadata
        settings = get_settings()
        metadata_path = settings.data_root / session_id / "hls" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            response.update(metadata)

    return response


@router.delete("/{session_id}/hls")
async def cancel_hls_stream(session_id: str) -> dict:
    """Cancel HLS generation."""
    generator = get_hls_generator(session_id)
    if generator:
        generator.cancel()
        unregister_hls_generator(session_id)
        return {"status": "cancelled"}
    raise HTTPException(status_code=404, detail="HLS stream not found")


