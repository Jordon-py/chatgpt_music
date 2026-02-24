"""
AuralMind MCP Server
Streamable HTTP
ChatGPT Developer Mode Compatible
"""
import time
import logging
import os
import uvicorn
import httpx
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from auralmind_engine.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    MasterRequest,
    MasterResponse,
    PresetsResponse,
)
from auralmind_engine.auralmind_match_maestro_v7_3_expert1 import (
    master,
    get_presets,
    load_audio,
    analyze_track_features
)


# ---------------------------------------------------------
# Server Instance
# ---------------------------------------------------------

mcp = FastMCP("AuralMind Mastering MCP")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("auralmind_mcp")

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not (1 <= value <= 65535):
        raise ValueError(f"{name} must be between 1 and 65535")
    return value

def _normalize_path(path: str) -> str:
    if not path.startswith("/"):
        return f"/{path}"
    return path

TEMP_DIR = os.getenv("AURALMIND_JOBS_DIR", "./jobs")
TEMP_DIR = os.path.abspath(TEMP_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)

def download_file(url: str) -> str:

    local_path = os.path.join(
        TEMP_DIR,
        os.path.basename(url)
    )

    with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    return local_path


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------


@mcp.tool
def list_presets() -> PresetsResponse:
    """
    Returns available mastering presets.
    """
    presets = get_presets()
    return PresetsResponse(
        success=True,
        message="Presets loaded",
        presets=presets
    )
@mcp.tool
def analyze_audio(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze audio from URL.

    Steps:

    1 download file
    2 load audio
    3 analyze
    """

    file_path = download_file(request.file_url)

    audio, sr = load_audio(file_path)

    analysis = analyze_track_features(audio, sr)

    return AnalyzeResponse(
        success=True,
        message="Analysis complete",
        analysis=analysis
    )


@mcp.tool
def master_audio(request: MasterRequest) -> MasterResponse:
    """
    Master audio from URL.

    Full mastering pipeline:

    download target
    download reference optional
    run AuralMind master
    return output
    """

    target = download_file(request.file_url)

    reference = None

    if request.reference_url:

        reference = download_file(request.reference_url)

    output_path = os.path.join(
        TEMP_DIR,
        request.output_name
    )

    result = master(
        target_path=target,
        reference_path=reference,
        preset=request.preset,
        output_path=output_path,
    )

    return MasterResponse(
        success=True,
        message="Master complete",
        output_url=output_path,
        report=result
    )


# ---------------------------------------------------------
# REQUIRED FOR CHATGPT
# ---------------------------------------------------------

app = mcp.http_app(
    path="/mcp",
    transport="streamable-http",
    json_response=True,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)


# ---------------------------------------------------------
# Local Run
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
