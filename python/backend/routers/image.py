"""Endpoints for single-image Gaussian Splatting."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, Response
from pydantic import BaseModel

from backend.utils.gs_ops import generate_gs_ply

router = APIRouter(prefix="/api/image", tags=["image"])


class ImagePathRequest(BaseModel):
    """Request model for processing local image file."""

    path: str


@router.post("/upload")
async def upload_image(file: UploadFile) -> Response:
    """Upload an image and convert it to a 3D Gaussian Splatting PLY file."""
    try:
        content = await file.read()
        ply_bytes = await generate_gs_ply(content)

        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{file.filename}.ply"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/path")
async def process_image_path(request: ImagePathRequest) -> Response:
    """Process a local image file path and return 3DGS PLY file."""
    path = Path(request.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        content = path.read_bytes()
        ply_bytes = await generate_gs_ply(content)

        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{path.name}.ply"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
