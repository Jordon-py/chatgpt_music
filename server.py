"""AuralMind FastMCP server (Heroku-ready) for LLM-orchestrated music mastering.

What this gives you
-------------------
- Remote MCP tools ChatGPT Developer Mode can call (over Streaming HTTP/SSE depending fastmcp version)
- HTTP endpoints for uploads, job status, report, and mastered-file download
- Job queue + polling pattern for long-running mastering operations
- Safe(ish) URL ingestion with SSRF guards and size limits
- Dynamic preset overrides so the LLM can act as the "brain" controlling your AuralMind engine

Deployment target
-----------------
- Works locally (`uvicorn server:app --reload`)
- Heroku-ready via Procfile (`uvicorn server:app --host 0.0.0.0 --port $PORT`)
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import shutil
import socket

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    # Official MCP Python SDK path (preferred)
    from mcp.server.fastmcp import FastMCP
except Exception:
    try:
        # Some community examples/packages expose FastMCP at top-level
        from fastmcp import FastMCP  # type: ignore
    except Exception as e:  # pragma: no cover - import fallback message for runtime debugging
        raise RuntimeError(
            "FastMCP import failed. Install dependencies from requirements.txt. "
            f"Original error: {e}"
        ) from e

from auralmind_engine import AuralMindAdapter, EngineNotReadyError


# -------------------------------------------------------------------------
# Logging / config helpers
# -------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("auralmind_mcp")

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = os.getenv(
    "AURALMIND_SCRIPT_PATH",
    str(BASE_DIR / "auralmind_engine" / "auralmind_match_maestro_v7_3_expert1.py"),
)
JOBS_DIR = Path(os.getenv("AURALMIND_JOBS_DIR", "/tmp/auralmind_jobs")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "250"))
JOB_MAX_WORKERS = max(1, int(os.getenv("JOB_MAX_WORKERS", "2")))
DEFAULT_PRESET = os.getenv("AURALMIND_DEFAULT_PRESET", "competitive_trap")
ALLOW_LOCAL_FILES = os.getenv("ALLOW_LOCAL_FILES", "false").strip().lower() in {"1", "true", "yes", "on"}

_allowed_hosts_env = os.getenv("ALLOWED_DOWNLOAD_HOSTS", "").strip()
ALLOWED_DOWNLOAD_HOSTS = {h.strip().lower() for h in _allowed_hosts_env.split(",") if h.strip()}

PUBLIC_BASE_URL = os.getenv("MCP_PUBLIC_BASE_URL", "").rstrip("/")


def _now_ts() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or _now_ts()))


def _public_url(path: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{path}"
    return path


# -------------------------------------------------------------------------
# Security / file utilities
# -------------------------------------------------------------------------
def _ensure_private_path(p: Path, *, must_exist: bool = False) -> Path:
    p = p.resolve()
    try:
        p.relative_to(JOBS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes jobs directory")
    if must_exist and not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return p


def _safe_filename(name: str, fallback: str) -> str:
    # Keep letters/digits/._- and collapse the rest.
    import re
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    cleaned = cleaned.strip("._") or fallback
    return cleaned[:180]


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    host = parsed.hostname.lower()

    if ALLOWED_DOWNLOAD_HOSTS and host not in ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Host '{host}' is not in ALLOWED_DOWNLOAD_HOSTS")

    # Resolve DNS and block localhost/private/link-local/etc. (basic SSRF guard).
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for host '{host}': {e}") from e

    for info in infos:
        ip_str = info[4][0]
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Blocked URL target IP ({ip}); private/local addresses are not allowed")


def _stream_download_to_file(url: str, dest_path: Path, max_mb: int = MAX_DOWNLOAD_MB) -> dict[str, Any]:
    _validate_remote_url(url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = max_mb * 1024 * 1024
    total = 0
    headers: dict[str, str] = {"User-Agent": "AuralMindMCP/1.0"}
    timeout = httpx.Timeout(20.0, connect=10.0, read=20.0, write=20.0)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if content_type and ("audio" not in content_type and "octet-stream" not in content_type):
                log.warning("Remote file content-type is %s (continuing)", content_type)

            with dest_path.open("wb") as f:
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Downloaded file exceeds MAX_DOWNLOAD_MB ({max_mb} MB)")
                    f.write(chunk)
    return {"bytes": total, "path": str(dest_path)}


async def _save_uploadfile(upload: UploadFile, dest: Path, max_mb: int = MAX_UPLOAD_MB) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = max_mb * 1024 * 1024
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"Upload exceeds MAX_UPLOAD_MB ({max_mb} MB)")
            f.write(chunk)
    await upload.close()
    return {"bytes": total, "path": str(dest)}


# -------------------------------------------------------------------------
# Engine + job models
# -------------------------------------------------------------------------
engine = AuralMindAdapter(SCRIPT_PATH)


class TrapMasterIntent(BaseModel):
    """High-level, LLM-friendly control surface for trap mastering recommendations.

    Use this model with the plan_trap_master tool to translate a human-readable
    creative direction (e.g. 'punchy, competitive loudness') into concrete
    preset + override values the mastering engine understands.
    """
    style: Literal["clean", "punchy", "wide", "dark", "aggressive", "radio-ready", "streaming"] = Field(
        "punchy",
        description="Overall sonic character. 'punchy' = tight low-end + transient snap; "
                    "'wide' = stereo spread emphasis; 'dark' = rolled-off highs; "
                    "'aggressive' = pushed loudness + distortion; 'radio-ready' = broadcast-safe; "
                    "'clean' = transparent; 'streaming' = safe for Spotify/Apple normalization."
    )
    loudness_goal: Literal["safe_streaming", "competitive", "very_loud"] = Field(
        "competitive",
        description="Target loudness tier. 'safe_streaming' ≈ -13 LUFS (Spotify safe); "
                    "'competitive' ≈ -10.5 LUFS (loud but clean); 'very_loud' ≈ -9 LUFS (max loudness, may sacrifice dynamics)."
    )
    brightness: int = Field(0, ge=-2, le=2, description="High-frequency tilt: -2 = noticeably darker, +2 = airy/bright. 0 = neutral.")
    width: int = Field(0, ge=-2, le=2, description="Stereo width: -2 = narrow/mono-ish, +2 = wide stereo image. 0 = neutral.")
    punch: int = Field(1, ge=-2, le=2, description="Transient attack energy: -2 = soft/smooth, +2 = aggressive snap. 1 = slight emphasis (trap default).")
    sibilance_sensitivity: int = Field(0, ge=-2, le=2, description="De-esser sensitivity: -2 = less de-essing (brighter vocals), +2 = aggressive sibilance control. 0 = moderate.")
    preserve_transients: bool = Field(True, description="Keep True for trap/hip-hop to preserve kick/snare attack. Set False only for ambient/pad-heavy material.")
    stem_separation: bool = Field(False, description="Run Demucs stem separation before mastering. Slower + requires more RAM. Usually False unless vocals need isolated processing.")


class StartJobInput(BaseModel):
    """MCP tool input to create a mastering job.

    Provide exactly ONE source (source_url OR local_target_path).
    Optionally provide ONE reference track for tonal matching.

    Typical ChatGPT flow:
      1. Call plan_trap_master to get a preset + overrides.
      2. Pass those into this tool along with a source_url.
      3. Poll get_job_status until status is 'completed' or 'failed'.
    """
    source_url: Optional[str] = Field(
        default=None,
        description="HTTPS URL pointing to the target audio file (WAV/MP3/FLAC). "
                    "This is the preferred input for remote ChatGPT flows. Use a pre-signed S3/GCS link or any public audio URL."
    )
    local_target_path: Optional[str] = Field(
        default=None,
        description="Absolute local filesystem path to the target audio file. "
                    "Only works when ALLOW_LOCAL_FILES=true (dev mode). Do NOT use in production/ChatGPT flows."
    )
    reference_url: Optional[str] = Field(
        default=None,
        description="HTTPS URL to a reference track for tonal/spectral matching. Optional but improves results."
    )
    local_reference_path: Optional[str] = Field(
        default=None,
        description="Local path to reference track. Dev-only; requires ALLOW_LOCAL_FILES=true."
    )
    preset: str = Field(
        default=DEFAULT_PRESET,
        description="Engine preset name. Call list_presets first to see available options. "
                    "Common: 'competitive_trap', 'hi_fi_streaming', 'radio_loud', 'club_clean'."
    )
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Dict of preset field overrides. Keys must be from the safe allowlist "
                    "(e.g. target_lufs, softclip_mix, width_hi). Call get_preset to see all fields. "
                    "Tip: use plan_trap_master to auto-generate good overrides from high-level intent."
    )
    dither_seed: int = Field(
        default=0,
        description="Random seed for dithering. 0 = random. Set a fixed value for reproducible output."
    )
    notes_for_llm: Optional[str] = Field(
        default=None,
        description="Free-text note stored alongside the job. Useful for tracking rationale or user instructions."
    )


class JobStatusView(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    preset: str
    source: dict[str, Any]
    reference: Optional[dict[str, Any]] = None
    notes_for_llm: Optional[str] = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class JobRecord:
    def __init__(self, *, job_id: str, source: dict[str, Any], reference: Optional[dict[str, Any]], preset: str, overrides: dict[str, Any], notes_for_llm: Optional[str]):
        now = _now_ts()
        self.job_id = job_id
        self.status: str = "queued"
        self.created_ts = now
        self.updated_ts = now
        self.preset = preset
        self.source = source
        self.reference = reference
        self.overrides = overrides
        self.notes_for_llm = notes_for_llm
        self.progress: dict[str, Any] = {}
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.paths: dict[str, str] = {}

    def touch(self):
        self.updated_ts = _now_ts()

    def to_view(self) -> JobStatusView:
        return JobStatusView(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            created_at=_iso(self.created_ts),
            updated_at=_iso(self.updated_ts),
            preset=self.preset,
            source=self.source,
            reference=self.reference,
            notes_for_llm=self.notes_for_llm,
            overrides=self.overrides,
            progress=self.progress,
            result=self.result,
            error=self.error,
            artifacts=self._artifact_view(),
        )

    def _artifact_view(self) -> dict[str, Any]:
        out = {}
        if "mastered_audio" in self.paths:
            out["mastered_audio_url"] = _public_url(f"/api/jobs/{self.job_id}/download")
        if "report_md" in self.paths and Path(self.paths["report_md"]).exists():
            out["report_url"] = _public_url(f"/api/jobs/{self.job_id}/report")
        if "result_json" in self.paths and Path(self.paths["result_json"]).exists():
            out["result_json_url"] = _public_url(f"/api/jobs/{self.job_id}/result.json")
        return out


class JobManager:
    def __init__(self, *, jobs_dir: Path, engine_adapter: AuralMindAdapter, max_workers: int = 2):
        self.jobs_dir = jobs_dir
        self.engine = engine_adapter
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="master-job")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _job_dir(self, job_id: str) -> Path:
        d = (self.jobs_dir / job_id).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def list_recent(self, limit: int = 20) -> list[JobStatusView]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_ts, reverse=True)
        return [j.to_view() for j in jobs[:max(1, min(limit, 100))]]

    def submit(
        self,
        *,
        source: dict[str, Any],
        reference: Optional[dict[str, Any]],
        preset: str,
        overrides: dict[str, Any],
        notes_for_llm: Optional[str] = None,
        dither_seed: int = 0,
    ) -> JobStatusView:
        job_id = uuid.uuid4().hex
        job = JobRecord(
            job_id=job_id,
            source=source,
            reference=reference,
            preset=preset,
            overrides=overrides,
            notes_for_llm=notes_for_llm,
        )
        with self._lock:
            self._jobs[job_id] = job

        self.executor.submit(self._run_job, job_id, dither_seed)
        return job.to_view()

    def _persist_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _run_job(self, job_id: str, dither_seed: int) -> None:
        job = self.get(job_id)
        job_dir = self._job_dir(job_id)
        source_audio = job_dir / "input_target.wav"
        reference_audio: Optional[Path] = None
        out_audio = job_dir / "mastered.wav"
        report_md = job_dir / "report.md"
        result_json = job_dir / "result.json"

        def set_status(status: str, **progress: Any) -> None:
            job.status = status
            if progress:
                job.progress.update(progress)
            job.touch()
            self._persist_json(job_dir / "job_status.json", job.to_view().model_dump())

        try:
            set_status("running", stage="preparing_inputs")

            # Ingestion
            src_mode = job.source.get("mode")
            if src_mode == "url":
                src_url = str(job.source["url"])
                src_name = _safe_filename(job.source.get("filename") or "target.wav", "target.wav")
                source_audio = job_dir / src_name
                _stream_download_to_file(src_url, source_audio)
                job.source["stored_path"] = str(source_audio)
            elif src_mode == "local_path":
                src = Path(str(job.source["path"])).expanduser().resolve()
                if not ALLOW_LOCAL_FILES:
                    raise PermissionError("Local file mode is disabled (ALLOW_LOCAL_FILES=false)")
                if not src.exists():
                    raise FileNotFoundError(f"Target file not found: {src}")
                shutil.copy2(src, source_audio)
                job.source["stored_path"] = str(source_audio)
            else:
                raise ValueError(f"Unsupported source mode: {src_mode!r}")

            if job.reference:
                ref_mode = job.reference.get("mode")
                if ref_mode == "url":
                    ref_name = _safe_filename(job.reference.get("filename") or "reference.wav", "reference.wav")
                    reference_audio = job_dir / ref_name
                    _stream_download_to_file(str(job.reference["url"]), reference_audio)
                    job.reference["stored_path"] = str(reference_audio)
                elif ref_mode == "local_path":
                    ref_src = Path(str(job.reference["path"])).expanduser().resolve()
                    if not ALLOW_LOCAL_FILES:
                        raise PermissionError("Local reference mode disabled (ALLOW_LOCAL_FILES=false)")
                    if not ref_src.exists():
                        raise FileNotFoundError(f"Reference file not found: {ref_src}")
                    reference_audio = job_dir / "reference.wav"
                    shutil.copy2(ref_src, reference_audio)
                    job.reference["stored_path"] = str(reference_audio)
                else:
                    raise ValueError(f"Unsupported reference mode: {ref_mode!r}")

            set_status("running", stage="mastering", message="Running AuralMind engine")

            result = self.engine.run_master(
                target_path=str(source_audio),
                out_path=str(out_audio),
                preset_name=job.preset,
                reference_path=str(reference_audio) if reference_audio else None,
                report_path=str(report_md),
                overrides=job.overrides,
                dither_seed=int(dither_seed),
            )

            job.result = result
            job.paths["mastered_audio"] = str(out_audio)
            if report_md.exists():
                job.paths["report_md"] = str(report_md)
            self._persist_json(result_json, result)
            job.paths["result_json"] = str(result_json)

            set_status("completed", stage="done")
        except Exception as e:
            log.exception("Job %s failed", job_id)
            job.error = f"{type(e).__name__}: {e}"
            set_status("failed", stage="failed")


jobs = JobManager(jobs_dir=JOBS_DIR, engine_adapter=engine, max_workers=JOB_MAX_WORKERS)


# -------------------------------------------------------------------------
# LLM helper logic (rule-based presets for trap mastering)
# -------------------------------------------------------------------------
def recommend_trap_overrides(intent: TrapMasterIntent) -> dict[str, Any]:
    """Simple deterministic mapper the LLM can call before starting a job.

    ChatGPT (the brain) can use this as a *baseline*, then edit the returned overrides.
    """
    preset = "competitive_trap"
    if intent.loudness_goal == "safe_streaming":
        preset = "hi_fi_streaming"
    elif intent.style in {"radio-ready"}:
        preset = "radio_loud"
    elif intent.style in {"clean", "wide"}:
        preset = "club_clean"

    overrides: dict[str, Any] = {
        "enable_stem_separation": bool(intent.stem_separation),
        "enable_transient_sculpt": bool(intent.preserve_transients),
        "target_lufs": -10.5 if intent.loudness_goal == "competitive" else (-12.8 if intent.loudness_goal == "safe_streaming" else -9.3),
        "softclip_mix": 0.22,
        "softclip_drive_db": 1.2,
        "microdetail_mix": 0.55,
        "hooklift_auto": True,
        "hooklift_mix": 0.22,
        "movement_amount": 0.10,
        "enable_movement": True,
        "governor_gr_limit_db": -2.0 if intent.loudness_goal != "very_loud" else -3.5,
        "out_subtype": "PCM_24",
        "dither": True,
    }

    # Brightness / width / punch shaping
    overrides["glow_mix"] = round(min(1.0, max(0.0, 0.48 + (0.08 * intent.brightness))), 3)
    overrides["deess_mix"] = round(min(1.0, max(0.0, 0.48 + (0.10 * intent.sibilance_sensitivity))), 3)
    overrides["width_hi"] = round(min(1.45, max(0.95, 1.18 + (0.08 * intent.width))), 3)
    overrides["width_mid"] = round(min(1.18, max(0.92, 1.03 + (0.03 * intent.width))), 3)
    overrides["microshift_mix"] = round(min(0.35, max(0.0, 0.12 + (0.04 * max(0, intent.width)))), 3)

    punch_map = {
        -2: (0.10, 0.4, 0.18),
        -1: (0.16, 0.8, 0.22),
         0: (0.22, 1.2, 0.26),
         1: (0.28, 1.8, 0.32),
         2: (0.34, 2.4, 0.38),
    }
    t_mix, t_boost, md_amount = punch_map[int(intent.punch)]
    overrides["transient_sculpt_mix"] = t_mix
    overrides["transient_sculpt_boost_db"] = t_boost
    overrides["microdetail_amount"] = md_amount

    if intent.style == "dark":
        overrides["warmth"] = 0.35
        overrides["glow_mix"] = round(max(0.0, overrides["glow_mix"] - 0.15), 3)
    elif intent.style == "wide":
        overrides["width_hi"] = min(1.5, float(overrides["width_hi"]) + 0.08)
        overrides["width_mid"] = min(1.2, float(overrides["width_mid"]) + 0.03)
    elif intent.style == "aggressive":
        overrides["target_lufs"] = max(-9.0, float(overrides["target_lufs"]))
        overrides["softclip_drive_db"] = min(4.0, float(overrides["softclip_drive_db"]) + 0.7)
        overrides["governor_gr_limit_db"] = -3.2

    rationale = [
        f"Base preset selected: {preset}",
        "Transient sculpt + movement/hooklift enabled for trap energy and perceived motion.",
        "Governor ceiling kept within a quality-first range to avoid crushed masters.",
        "Output defaults to PCM_24 + dither=true for portable deliverables.",
    ]
    return {"preset": preset, "overrides": overrides, "rationale": rationale}


# -------------------------------------------------------------------------
# MCP tool schemas (define Pydantic models before decorators)
# -------------------------------------------------------------------------
class PresetLookupInput(BaseModel):
    preset: str = Field(description="Exact preset name (case-sensitive). Call list_presets first to see valid names.")


class JobLookupInput(BaseModel):
    job_id: str = Field(description="32-char hex job ID returned by start_master_job.")


class RecentJobsInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)


# -------------------------------------------------------------------------
# FastMCP tools
# -------------------------------------------------------------------------
# Bind FastMCP to 0.0.0.0 in server mode so host-header protection
# does not auto-lock to localhost-only defaults.
mcp = FastMCP(
    "AuralMind Mastering Server",
    host=os.getenv("FASTMCP_BIND_HOST", "0.0.0.0"),
    instructions=(
        "You are connected to the AuralMind mastering engine. "
        "Follow this workflow to master a track:\n"
        "1. Call server_health to confirm the engine is loaded.\n"
        "2. Call list_presets to see available mastering presets.\n"
        "3. (Optional) Call plan_trap_master with a high-level TrapMasterIntent to get recommended preset + overrides.\n"
        "4. Call start_master_job with a source_url (or local_target_path in dev), preset, and overrides.\n"
        "5. Poll get_job_status every 5-10 seconds until status is 'completed' or 'failed'.\n"
        "6. When completed, the response includes download URLs for the mastered audio, report, and result JSON.\n"
        "If a job fails, check the 'error' field in get_job_status for diagnostics."
    ),
)

@mcp.tool(
    name="server_health",
    description=(
        "Check server and AuralMind engine readiness. Call this FIRST when starting a session "
        "or if any other tool returns an error. Returns: ok (bool), preset count, config summary. "
        "If ok=false, the engine script is missing or failed to load — report the error to the user."
    ),
)
def server_health() -> dict[str, Any]:
    try:
        info = engine.reload() if os.getenv("ENGINE_RELOAD_ON_HEALTH", "false").lower() in {"1","true","yes"} else {
            "script_path": SCRIPT_PATH,
            "preset_count": len(engine.get_preset_names()),
        }
        return {
            "ok": True,
            "time": _iso(),
            "jobs_dir": str(JOBS_DIR),
            "default_preset": DEFAULT_PRESET,
            "allow_local_files": ALLOW_LOCAL_FILES,
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_download_mb": MAX_DOWNLOAD_MB,
            "engine": info,
        }
    except EngineNotReadyError as e:
        return {"ok": False, "time": _iso(), "error": str(e)}

@mcp.tool(
    name="list_presets",
    description=(
        "List all available mastering preset names with a compact summary of key settings "
        "(target_lufs, ceiling, sample rate, stem sep). Call this before start_master_job "
        "to choose a preset. Common presets: competitive_trap, hi_fi_streaming, radio_loud, club_clean."
    ),
)
def list_presets() -> dict[str, Any]:
    presets = engine.get_presets()
    compact = {}
    for name, p in presets.items():
        compact[name] = {
            "target_lufs": p.get("target_lufs"),
            "ceiling_dbfs": p.get("ceiling_dbfs"),
            "sr": p.get("sr"),
            "enable_stem_separation": p.get("enable_stem_separation"),
        }
    return {"preset_count": len(compact), "presets": compact}

@mcp.tool(
    name="get_preset",
    description=(
        "Return ALL fields of a single preset as a JSON object. Use this to see every tunable parameter "
        "and its current value before crafting overrides for start_master_job. "
        "Input: {\"preset\": \"competitive_trap\"}. Returns: full preset dict."
    ),
)
def get_preset(payload: PresetLookupInput) -> dict[str, Any]:
    return {"preset": payload.preset, "details": engine.get_preset_details(payload.preset)}

@mcp.tool(
    name="plan_trap_master",
    description=(
        "Translate a high-level creative direction into a concrete preset name + safe overrides dict. "
        "Call this BEFORE start_master_job when the user describes a vibe (e.g. 'punchy and loud', "
        "'dark and wide', 'streaming-safe'). Returns: {preset, overrides, rationale[]}. "
        "Pass the returned preset and overrides directly into start_master_job. "
        "You can also edit the overrides before submitting if the user requests fine-tuning."
    ),
)
def plan_trap_master(payload: TrapMasterIntent) -> dict[str, Any]:
    return recommend_trap_overrides(payload)

@mcp.tool(
    name="start_master_job",
    description=(
        "Queue a new mastering job and return immediately with a job_id. "
        "You MUST provide exactly one source: source_url (preferred for ChatGPT) or local_target_path (dev only). "
        "Optionally provide a reference track URL for tonal matching. "
        "After calling this, poll get_job_status with the returned job_id every 5-10 seconds. "
        "When status='completed', the response includes download URLs for the mastered audio and report."
    ),
)
def start_master_job(payload: StartJobInput) -> dict[str, Any]:
    if not payload.source_url and not payload.local_target_path:
        raise ValueError("Provide either source_url or local_target_path")

    if payload.source_url and payload.local_target_path:
        raise ValueError("Provide only one of source_url or local_target_path")

    if payload.reference_url and payload.local_reference_path:
        raise ValueError("Provide only one of reference_url or local_reference_path")

    source: dict[str, Any]
    if payload.source_url:
        source = {"mode": "url", "url": payload.source_url}
    else:
        source = {"mode": "local_path", "path": payload.local_target_path}

    reference: Optional[dict[str, Any]] = None
    if payload.reference_url:
        reference = {"mode": "url", "url": payload.reference_url}
    elif payload.local_reference_path:
        reference = {"mode": "local_path", "path": payload.local_reference_path}

    # Validate overrides up front so the LLM gets immediate feedback.
    engine.validate_overrides(payload.overrides)
    engine.split_virtual_master_args(payload.overrides)

    view = jobs.submit(
        source=source,
        reference=reference,
        preset=payload.preset,
        overrides=payload.overrides,
        notes_for_llm=payload.notes_for_llm,
        dither_seed=payload.dither_seed,
    )
    return {
        "job": view.model_dump(),
        "next_step": "Call get_job_status with the returned job_id until status is completed or failed."
    }

@mcp.tool(
    name="get_job_status",
    description=(
        "Check the current status of a mastering job. Call repeatedly after start_master_job "
        "until status is 'completed' or 'failed'. Status lifecycle: queued → running → completed|failed. "
        "On 'completed': artifacts dict contains mastered_audio_url, report_url, result_json_url. "
        "On 'failed': error field contains the exception message for diagnostics."
    ),
)
def get_job_status(payload: JobLookupInput) -> dict[str, Any]:
    return jobs.get(payload.job_id).to_view().model_dump()

@mcp.tool(
    name="list_recent_jobs",
    description=(
        "List the most recent mastering jobs (newest first). Returns up to 'limit' jobs (default 10, max 50). "
        "Each entry includes job_id, status, preset, timestamps, and artifact URLs if completed. "
        "Use this to find a previous job's ID, check what's queued, or resume a session."
    ),
)
def list_recent_jobs(payload: RecentJobsInput) -> dict[str, Any]:
    return {"jobs": [j.model_dump() for j in jobs.list_recent(payload.limit)]}


# -------------------------------------------------------------------------
# HTTP API (uploads + artifact delivery + status)
# -------------------------------------------------------------------------
# NOTE: We must build mcp_app before creating FastAPI so we can pass its lifespan.
# The _build_mcp_app() call is at the bottom of this file, so we use a lazy
# lifespan that forwards to mcp_app once it exists.
from contextlib import asynccontextmanager

@asynccontextmanager
async def _combined_lifespan(application):
    """Forward lifespan to the MCP sub-app so session task-groups initialize."""
    # mcp_app is created at module level below; by the time uvicorn starts
    # the async lifespan, it will be available.
    _mcp_lf = getattr(globals().get("mcp_app", None), "lifespan", None)
    if _mcp_lf is not None:
        async with _mcp_lf(application):
            yield
    else:
        yield

app = FastAPI(
    title="AuralMind FastMCP Mastering Server",
    version="1.0.0",
    description="FastAPI + FastMCP wrapper around a user-supplied AuralMind Python mastering script",
    lifespan=_combined_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if you front this with a trusted UI
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        preset_names = engine.get_preset_names()
        return {
            "ok": True,
            "time": _iso(),
            "jobs_dir": str(JOBS_DIR),
            "engine_script_path": SCRIPT_PATH,
            "preset_count": len(preset_names),
            "presets": preset_names,
        }
    except Exception as e:
        return {"ok": False, "time": _iso(), "error": f"{type(e).__name__}: {e}"}

@app.get("/api/presets")
def api_presets() -> dict[str, Any]:
    return list_presets()

@app.get("/api/presets/{preset_name}")
def api_preset_detail(preset_name: str) -> dict[str, Any]:
    try:
        return get_preset(PresetLookupInput(preset=preset_name))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/jobs/upload")
async def create_job_via_upload(
    request: Request,
    target_file: UploadFile = File(..., description="Target song to master"),
    reference_file: UploadFile | None = File(default=None, description="Optional reference track"),
    preset: str = Form(default=DEFAULT_PRESET),
    overrides_json: str = Form(default="{}"),
    notes_for_llm: str | None = Form(default=None),
    dither_seed: int = Form(default=0),
) -> JSONResponse:
    """Multipart upload endpoint for non-MCP clients (browser/cURL/Postman).

    ChatGPT MCP tool calls usually prefer `start_master_job(source_url=...)` because tool inputs are JSON.
    """
    try:
        overrides = json.loads(overrides_json or "{}")
        if not isinstance(overrides, dict):
            raise ValueError("overrides_json must decode to an object")
        engine.validate_overrides(overrides)
        engine.split_virtual_master_args(overrides)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid overrides_json: {e}") from e

    job_id = uuid.uuid4().hex
    job_dir = jobs._job_dir(job_id)
    target_name = _safe_filename(target_file.filename or "target.wav", "target.wav")
    target_dest = job_dir / target_name
    await _save_uploadfile(target_file, target_dest)

    source = {"mode": "local_path", "path": str(target_dest), "uploaded_via": "multipart"}

    reference = None
    if reference_file is not None:
        ref_name = _safe_filename(reference_file.filename or "reference.wav", "reference.wav")
        ref_dest = job_dir / ref_name
        await _save_uploadfile(reference_file, ref_dest)
        reference = {"mode": "local_path", "path": str(ref_dest), "uploaded_via": "multipart"}

    # Temporarily allow local file semantics for server-side staged uploads.
    # We pass local paths inside the jobs dir and the job runner copies them into canonical names.
    prev = os.environ.get("ALLOW_LOCAL_FILES")
    os.environ["ALLOW_LOCAL_FILES"] = "true"
    global ALLOW_LOCAL_FILES
    old_allow_local = ALLOW_LOCAL_FILES
    ALLOW_LOCAL_FILES = True
    try:
        view = jobs.submit(
            source=source,
            reference=reference,
            preset=preset,
            overrides=overrides,
            notes_for_llm=notes_for_llm,
            dither_seed=dither_seed,
        )
    finally:
        ALLOW_LOCAL_FILES = old_allow_local
        if prev is None:
            os.environ.pop("ALLOW_LOCAL_FILES", "true")
        else:
            os.environ["ALLOW_LOCAL_FILES"] = prev

    return JSONResponse(
        status_code=202,
        content={
            "job": view.model_dump(),
            "poll_url": _public_url(f"/api/jobs/{view.job_id}"),
            "download_url": _public_url(f"/api/jobs/{view.job_id}/download"),
            "report_url": _public_url(f"/api/jobs/{view.job_id}/report"),
        },
    )

@app.post("/api/jobs/from-url")
def create_job_from_url(payload: StartJobInput) -> JSONResponse:
    # Reuse MCP validation logic
    try:
        data = start_master_job(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(status_code=202, content=data)

@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> dict[str, Any]:
    return {"jobs": [j.model_dump() for j in jobs.list_recent(limit)]}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return jobs.get(job_id).to_view().model_dump()
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/jobs/{job_id}/result.json")
def get_job_result_json(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.paths.get("result_json")
    if not path:
        raise HTTPException(status_code=404, detail="Result JSON not available yet")
    p = _ensure_private_path(Path(path), must_exist=True)
    return FileResponse(p, media_type="application/json", filename=f"{job_id}_result.json")

@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.paths.get("report_md")
    if not path:
        raise HTTPException(status_code=404, detail="Report not available yet")
    p = _ensure_private_path(Path(path), must_exist=True)
    return FileResponse(p, media_type="text/markdown", filename=f"{job_id}_report.md")

@app.get("/api/jobs/{job_id}/download")
def download_master(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed (status={job.status})")

    path = job.paths.get("mastered_audio")
    if not path:
        raise HTTPException(status_code=404, detail="Mastered output missing")

    p = _ensure_private_path(Path(path), must_exist=True)
    filename = _safe_filename(f"{job_id}_mastered{p.suffix or '.wav'}", f"{job_id}_mastered.wav")
    return FileResponse(p, media_type="audio/wav", filename=filename)

@app.post("/api/admin/reload-engine")
def reload_engine() -> dict[str, Any]:
    try:
        return engine.reload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# -------------------------------------------------------------------------
# Mount the MCP ASGI app
# -------------------------------------------------------------------------
# Per FastMCP docs for FastAPI integration:
#   - Create MCP ASGI app with path="/" (no internal prefix)
#   - Mount it at "/mcp" on the FastAPI app
#   - Pass lifespan=mcp_app.lifespan to FastAPI so session task-groups initialize
# This makes the MCP endpoint reachable at /mcp/ (streamable HTTP / SSE).

def _build_mcp_app():
    """Support minor FastMCP version differences."""
    # Use path="/" here — the mount point on FastAPI provides the prefix.
    if hasattr(mcp, "http_app"):
        try:
            return mcp.http_app(path="/")
        except TypeError:
            return mcp.http_app()
    if hasattr(mcp, "streamable_http_app"):
        return mcp.streamable_http_app()
    raise RuntimeError("FastMCP version does not expose http_app() / streamable_http_app()")

mcp_app = _build_mcp_app()

# Mount at /mcp so /healthz and /api/* routes are matched by FastAPI first.
app.mount("/mcp", mcp_app)


# -------------------------------------------------------------------------
# Local entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
