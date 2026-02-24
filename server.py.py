"""
AuralMind MCP Server
Streamable HTTP
ChatGPT Developer Mode Compatible (Render-friendly)
"""
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import uvicorn
from fastapi import Response
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# IMPORTANT: use the official MCP SDK import
from mcp.server.fastmcp import FastMCP

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
    analyze_track_features,
)

# ---------------------------------------------------------
# Server Instance
# ---------------------------------------------------------

mcp = FastMCP("AuralMind Mastering MCP")

# ---------------------------------------------------------
# Helpers / Config
# ---------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("auralmind_mcp")

def _env_int(name: str, default: int, *, low: int = 1, high: int = 10_000_000) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return value

def _normalize_path(path: str) -> str:
    if not path.startswith("/"):
        return f"/{path}"
    return path.rstrip("/") or "/"

def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"

def _safe_output_name(name: str) -> str:
    base = os.path.basename(name or "")
    base = _safe_slug(base)
    if not base.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".aif")):
        base += ".wav"
    return base

TEMP_DIR = Path(os.getenv("AURALMIND_JOBS_DIR", "./jobs")).resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

HTTP_PATH = _normalize_path(os.getenv("FASTMCP_HTTP_PATH", "/mcp"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
MAX_DOWNLOAD_MB = _env_int("MAX_DOWNLOAD_MB", 250, low=1, high=5000)
DOWNLOAD_TIMEOUT_S = _env_int("DOWNLOAD_TIMEOUT_S", 180, low=5, high=3600)
LINK_BANK_FILE = os.getenv("LINK_BANK_FILE", "/etc/secrets/vegeta.txt")
LINK_BANK_FALLBACK = os.getenv("LINK_BANK_FALLBACK", "./list_urls.txt")

def _pick_suffix_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_name = os.path.basename(parsed.path)
    if "." in path_name:
        ext = os.path.splitext(path_name)[1]
        if ext and len(ext) <= 10:
            return ext
    return ".bin"

def _file_id_from_google_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    # Match docs.google.com/uc?export=download&id=...
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    # Match drive.google.com/file/d/<id>/...
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    return m.group(1) if m else None

def _normalize_google_drive_url(url: str) -> str:
    file_id = _file_id_from_google_url(url)
    if file_id:
        return f"https://docs.google.com/uc?export=download&id={file_id}"
    return url

def _build_local_download_path(url: str) -> Path:
    parsed = urlparse(url)
    suffix = _pick_suffix_from_url(url)
    gid = _file_id_from_google_url(url)
    stem = gid if gid else _safe_slug(os.path.basename(parsed.path) or "download")
    return TEMP_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

def _build_output_url(output_name: str) -> str:
    filename = _safe_output_name(output_name)
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/jobs/{filename}"
    return f"/jobs/{filename}"

def load_link_bank() -> List[str]:
    """
    Loads URLs from Render secret file (/etc/secrets/vegeta.txt) or local fallback.
    One URL per line. Ignores blanks/comments and de-dupes while preserving order.
    """
    paths = [Path(LINK_BANK_FILE), Path(LINK_BANK_FALLBACK)]
    seen = set()
    for p in paths:
        if not p.exists():
            continue
        urls: List[str] = []
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("http://") or line.startswith("https://"):
                line = _normalize_google_drive_url(line)
                if line not in seen:
                    seen.add(line)
                    urls.append(line)
        if urls:
            return urls
    return []

def download_file(url: str) -> str:
    """
    Downloads a remote file to TEMP_DIR.
    Uses longer timeout for large audio and guards max file size.
    """
    url = _normalize_google_drive_url(url)
    local_path = _build_local_download_path(url)
    max_bytes = MAX_DOWNLOAD_MB * 1024 * 1024

    headers = {
        "User-Agent": "AuralMindMCP/1.0 (+Render)",
        "Accept": "*/*",
    }

    timeout = httpx.Timeout(connect=20.0, read=float(DOWNLOAD_TIMEOUT_S), write=60.0, pool=60.0)

    logger.info("Downloading url=%s -> %s", url, local_path.name)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()

        ctype = (response.headers.get("content-type") or "").lower()
        clen_header = response.headers.get("content-length")
        if clen_header:
            try:
                clen = int(clen_header)
                if clen > max_bytes:
                    raise ValueError(f"Download exceeds MAX_DOWNLOAD_MB ({MAX_DOWNLOAD_MB} MB)")
            except ValueError:
                # Ignore malformed content-length; enforce while streaming below
                pass

        total = 0
        with open(local_path, "wb") as f:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Download exceeds MAX_DOWNLOAD_MB ({MAX_DOWNLOAD_MB} MB)")
                f.write(chunk)

    # Very common failure mode: Drive returns HTML interstitial instead of audio bytes
    if local_path.stat().st_size > 0 and local_path.suffix in {".bin", ".html", ""}:
        try:
            head = local_path.read_bytes()[:512].lower()
            if b"<html" in head or b"google drive" in head:
                raise ValueError("Drive link returned HTML instead of raw audio. Use a direct/raw file URL.")
        except Exception:
            # keep original file if binary read fails unexpectedly
            pass

    return str(local_path)

class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)

# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------

@mcp.tool
def list_presets() -> PresetsResponse:
    """Returns available mastering presets."""
    presets = get_presets()
    return PresetsResponse(success=True, message="Presets loaded", presets=presets)

@mcp.tool
def list_link_bank() -> dict:
    """List URLs loaded from Render secret file or local fallback."""
    links = load_link_bank()
    return {
        "success": True,
        "message": "Link bank loaded" if links else "No links found",
        "count": len(links),
        "links": links,
        "source_primary": LINK_BANK_FILE,
        "source_fallback": LINK_BANK_FALLBACK,
    }

@mcp.tool
def analyze_bank_link(index: int) -> AnalyzeResponse:
    """Analyze audio from the link bank by index."""
    links = load_link_bank()
    if index < 0 or index >= len(links):
        raise ValueError(f"index out of range (0..{max(len(links)-1, 0)})")
    return analyze_audio(AnalyzeRequest(file_url=links[index]))

@mcp.tool
def analyze_audio(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze audio from URL.
    Steps:
    1) download file
    2) load audio
    3) analyze
    """
    file_path = download_file(request.file_url)
    audio, sr = load_audio(file_path)
    analysis = analyze_track_features(audio, sr)
    return AnalyzeResponse(success=True, message="Analysis complete", analysis=analysis)

@mcp.tool
def master_bank_link(target_index: int, reference_index: Optional[int] = None, preset: str = "") -> MasterResponse:
    """
    Master from link bank by index (target + optional reference).
    """
    links = load_link_bank()
    if target_index < 0 or target_index >= len(links):
        raise ValueError(f"target_index out of range (0..{max(len(links)-1, 0)})")
    ref_url = None
    if reference_index is not None:
        if reference_index < 0 or reference_index >= len(links):
            raise ValueError(f"reference_index out of range (0..{max(len(links)-1, 0)})")
        ref_url = links[reference_index]
    out_name = f"master_{target_index}_{reference_index if reference_index is not None else 'noref'}_{uuid.uuid4().hex[:6]}.wav"
    return master_audio(
        MasterRequest(
            file_url=links[target_index],
            reference_url=ref_url,
            preset=(preset or os.getenv("AURALMIND_DEFAULT_PRESET", "competitive_trap")),
            output_name=out_name,
        )
    )

@mcp.tool
def master_audio(request: MasterRequest) -> MasterResponse:
    """
    Master audio from URL.
    Full mastering pipeline:
    - download target
    - download reference optional
    - run AuralMind master
    - return output metadata
    """
    target = download_file(request.file_url)
    reference = download_file(request.reference_url) if request.reference_url else None

    output_name = _safe_output_name(request.output_name)
    output_path = str(TEMP_DIR / output_name)

    result = master(
        target_path=target,
        reference_path=reference,
        preset=request.preset,
        output_path=output_path,
    )

    return MasterResponse(
        success=True,
        message="Master complete",
        output_url=_build_output_url(output_name),  # public-ish path if PUBLIC_BASE_URL is set
        report=result,
    )

# ---------------------------------------------------------
# REQUIRED FOR CHATGPT (Streamable HTTP)
# ---------------------------------------------------------

app = mcp.http_app(
    path=HTTP_PATH,
    transport="streamable-http",
    json_response=True,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(SecurityHeadersMiddleware),
    ],
)

def healthcheck(_request):
    return JSONResponse(
        {
            "status": "ok",
            "mcp_path": HTTP_PATH,
            "jobs_dir": str(TEMP_DIR),
            "link_bank_file": LINK_BANK_FILE,
            "link_bank_count": len(load_link_bank()),
        }
    )

def root(_request):
    return JSONResponse(
        {
            "name": "AuralMind MCP Server",
            "status": "ok",
            "health": "/healthz",
            "mcp": HTTP_PATH,
        }
    )

def mcp_head_probe(_request):
    # Helps client UIs that probe with HEAD /mcp before connecting.
    return Response(status_code=200)

def jobs_download(request):
    filename = _safe_output_name(request.path_params["filename"])
    file_path = TEMP_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(file_path), filename=filename)

def _on_startup() -> None:
    logger.info("AuralMind MCP starting")
    logger.info("mcp_path=%s", HTTP_PATH)
    logger.info("jobs_dir=%s", TEMP_DIR)
    logger.info("link_bank_primary=%s exists=%s", LINK_BANK_FILE, Path(LINK_BANK_FILE).exists())
    logger.info("link_bank_fallback=%s exists=%s", LINK_BANK_FALLBACK, Path(LINK_BANK_FALLBACK).exists())

def _on_shutdown() -> None:
    logger.info("AuralMind MCP shutting down")

app.add_route("/", root, methods=["GET"])
app.add_route("/healthz", healthcheck, methods=["GET"])
app.add_route("/jobs/{filename:str}", jobs_download, methods=["GET"])
# Add explicit HEAD route for the MCP probe; if your FastMCP version conflicts, remove this line.
app.add_route(HTTP_PATH, mcp_head_probe, methods=["HEAD"])
app.add_event_handler("startup", _on_startup)
app.add_event_handler("shutdown", _on_shutdown)

# ---------------------------------------------------------
# Local Run
# ---------------------------------------------------------

if __name__ == "__main__":
    port = _env_int("PORT", 10000, low=1, high=65535)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=LOG_LEVEL.lower())
