"""
AuralMind Mastering MCP Server (ChatGPT Developer Mode / Custom MCP)
--------------------------------------------------------------------
Purpose:
- Expose your AuralMind mastering script as MCP tools so ChatGPT can act as the "brain"
  and dynamically choose presets/overrides for trap masters.
- Provide job-based mastering (start -> poll -> fetch report/artifacts)
- Add safe file ingest/upload and artifact download endpoints.

Notes:
- Designed around auralmind_match_maestro_v7_3_expert1.py exposing:
  - get_presets()
  - master(target_path, out_path, preset, reference_path=None, report_path=None, ...)
- Mounts MCP endpoint at /mcp (streamable HTTP via FastMCP http_app()).
- Includes REST helper endpoints for uploads/downloads.

If your fastmcp version has different APIs, adapt:
- mcp = FastMCP(...)
- mcp_app = mcp.http_app(path="/mcp")
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import os
import re
import socket
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import requests
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# FastMCP import (common package name)
from fastmcp import FastMCP

# -----------------------------
# Config
# -----------------------------

APP_NAME = "AuralMind Mastering MCP"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Set AURALMIND_SCRIPT_PATH env var to the mastering script location.
# Default is a relative path next to this server file.
SCRIPT_PATH = Path(
    os.getenv("AURALMIND_SCRIPT_PATH", "./auralmind_match_maestro_v7_3_expert1.py")
).resolve()

DATA_DIR = Path(os.getenv("AURALMIND_DATA_DIR", "./auralmind_data")).resolve()
INBOX_DIR = DATA_DIR / "inbox"
JOBS_DIR = DATA_DIR / "jobs"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "250"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Public base URL is optional but helpful for returning absolute artifact links in tool responses.
# Example: https://your-domain.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# URL ingest hardening
ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "0") == "1"
URL_DOWNLOAD_TIMEOUT_CONNECT = float(os.getenv("URL_DOWNLOAD_TIMEOUT_CONNECT", "10"))
URL_DOWNLOAD_TIMEOUT_READ = float(os.getenv("URL_DOWNLOAD_TIMEOUT_READ", "120"))

# Job execution
MAX_WORKERS = int(os.getenv("AURALMIND_MAX_WORKERS", "2"))

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}

for p in (INBOX_DIR, JOBS_DIR):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utility helpers
# -----------------------------

def _now_ts() -> float:
    return time.time()


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_filename(name: str, default_stem: str = "audio") -> str:
    name = (name or "").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        name = f"{default_stem}.wav"
    stem, ext = os.path.splitext(name)
    if not ext:
        ext = ".wav"
    if ext.lower() not in ALLOWED_AUDIO_EXTS:
        ext = ".wav"
    stem = stem or default_stem
    return f"{stem}{ext.lower()}"


def _ensure_within(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if root == candidate or root in candidate.parents:
        return candidate
    raise ValueError(f"Path escapes sandbox: {candidate}")


def _public_url(path: str) -> Optional[str]:
    if not PUBLIC_BASE_URL:
        return None
    # path should already start with /
    return f"{PUBLIC_BASE_URL}{path}"


def _is_blocked_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_url_for_download(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed.")
    if not parsed.hostname:
        raise ValueError("URL hostname is required.")

    if not ALLOW_PRIVATE_URLS:
        try:
            infos = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
            for info in infos:
                ip = info[4][0]
                if _is_blocked_ip(ip):
                    raise ValueError(f"Blocked private/local address in URL resolution: {ip}")
        except socket.gaierror as e:
            raise ValueError(f"Failed to resolve URL host: {e}") from e

    return url


def _write_stream_to_file(resp: requests.Response, out_path: Path, max_bytes: int) -> int:
    total = 0
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"File exceeds max size limit ({max_bytes} bytes).")
            f.write(chunk)
    return total


# -----------------------------
# AuralMind script dynamic loader
# -----------------------------

_AURALMIND_MOD = None
_AURALMIND_LOCK = threading.Lock()


def get_auralmind_module():
    """
    Dynamically import the user's mastering script from SCRIPT_PATH.
    """
    global _AURALMIND_MOD
    with _AURALMIND_LOCK:
        if _AURALMIND_MOD is not None:
            return _AURALMIND_MOD

        if not SCRIPT_PATH.exists():
            raise FileNotFoundError(f"AuralMind script not found: {SCRIPT_PATH}")

        spec = importlib.util.spec_from_file_location("auralmind_user_script", str(SCRIPT_PATH))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec for AuralMind script.")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        missing = [name for name in ("get_presets", "master") if not hasattr(mod, name)]
        if missing:
            raise RuntimeError(f"AuralMind script missing required callables: {missing}")

        _AURALMIND_MOD = mod
        return mod


def _preset_to_dict(preset_obj: Any) -> Dict[str, Any]:
    if is_dataclass(preset_obj):
        return asdict(preset_obj)
    if hasattr(preset_obj, "__dict__"):
        return dict(vars(preset_obj))
    raise TypeError("Unsupported preset object type.")


def _dump_model(model: BaseModel) -> Dict[str, Any]:
    # Pydantic v2 / v1 compatibility helper
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    return model.dict(exclude_none=True)  # type: ignore[attr-defined]


# -----------------------------
# File registry (ingested audio)
# -----------------------------

FILES_LOCK = threading.Lock()
FILES: Dict[str, Dict[str, Any]] = {}  # audio_id -> metadata


def _register_file(path: Path, original_name: str, source: str) -> str:
    path = _ensure_within(DATA_DIR, path)
    audio_id = uuid.uuid4().hex
    meta = {
        "audio_id": audio_id,
        "path": str(path),
        "original_name": original_name,
        "source": source,
        "created_at": _iso_now(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    with FILES_LOCK:
        FILES[audio_id] = meta
    return audio_id


def _resolve_audio_path(audio_id: Optional[str], explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        # We only allow explicit_path if it's inside DATA_DIR by default.
        # If you want broader access during local dev, register it first via register_local_audio.
        return _ensure_within(DATA_DIR, p)

    if audio_id:
        with FILES_LOCK:
            meta = FILES.get(audio_id)
        if not meta:
            raise ValueError(f"Unknown audio_id: {audio_id}")
        return _ensure_within(DATA_DIR, Path(meta["path"]))

    return None


# -----------------------------
# Job models and state
# -----------------------------

class TrapProfile(BaseModel):
    """
    High-level user intent that ChatGPT can populate instead of guessing low-level DSP numbers.
    """
    loudness_style: Literal["streaming_clean", "competitive", "radio_loud"] = "competitive"
    vibe: Literal["clean", "punchy", "dark", "airy", "warm"] = "punchy"
    vocal_priority: Literal["high", "balanced", "low"] = "balanced"
    bass_priority: Literal["tight", "big", "sub_heavy"] = "big"
    preserve_dynamics: bool = True
    notes: Optional[str] = None


class PresetOverrides(BaseModel):
    """
    Safe subset of frequently useful overrides (mapped into dataclass replace()).
    You can extend this list over time.
    """
    target_lufs: Optional[float] = Field(default=None, ge=-20.0, le=-6.0)
    ceiling_dbfs: Optional[float] = Field(default=None, ge=-3.0, le=-0.1)

    limiter_mode: Optional[Literal["v1", "v2"]] = None
    enable_limiter: Optional[bool] = None
    enable_softclip: Optional[bool] = None

    enable_microdetail: Optional[bool] = None
    microdetail_amount: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    enable_movement: Optional[bool] = None
    movement_amount: Optional[float] = Field(default=None, ge=0.0, le=0.5)

    enable_hooklift: Optional[bool] = None
    hooklift_auto: Optional[bool] = None
    hooklift_mix: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    enable_stem_separation: Optional[bool] = None
    demucs_device: Optional[Literal["cpu", "cuda"]] = None
    demucs_overlap: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    demucs_shifts: Optional[int] = Field(default=None, ge=1, le=8)

    fir_streaming: Optional[Literal["auto", "on", "off"]] = None
    fir_block_pow2: Optional[int] = Field(default=None, ge=12, le=20)

    warmth: Optional[float] = Field(default=None, ge=-3.0, le=3.0)

    transient_sculpt_boost_db: Optional[float] = Field(default=None, ge=0.0, le=8.0)
    transient_sculpt_mix: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    transient_sculpt_crest_guard_db: Optional[float] = Field(default=None, ge=6.0, le=30.0)
    transient_sculpt_decay_ms: Optional[float] = Field(default=None, ge=0.5, le=50.0)

    # Escape hatch for future preset fields without changing schema every time
    extra: Dict[str, Any] = Field(default_factory=dict)


class MasteringRequest(BaseModel):
    """
    Main job request for the mastering engine.
    """
    source_audio_id: Optional[str] = None
    source_path: Optional[str] = None

    reference_audio_id: Optional[str] = None
    reference_path: Optional[str] = None

    preset_name: str = "hi_fi_streaming"
    trap_profile: Optional[TrapProfile] = None
    overrides: Optional[PresetOverrides] = None

    output_basename: Optional[str] = None
    write_report: bool = True

    out_subtype: Optional[str] = None  # e.g. PCM_24
    dither: Optional[bool] = None
    dither_seed: int = 0

    # Use script-native auto tuning functions if available
    use_script_auto_tune: bool = False

    # Optional short user intent (good for audit trail)
    user_goal: Optional[str] = None


class MasteringJobView(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    request: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    report_excerpt: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updated_at"] = _iso_now()


def _create_job_record(req: MasteringRequest) -> str:
    job_id = uuid.uuid4().hex
    now = _iso_now()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "request": _dump_model(req),
            "result": None,
            "error": None,
            "report_excerpt": None,
            "artifacts": {},
        }
    return job_id


def _recommend_trap_params(profile: TrapProfile) -> Dict[str, Any]:
    """
    Deterministic server-side helper.
    ChatGPT can call this tool and then still adjust values based on taste.
    """
    preset_name = "hi_fi_streaming"
    updates: Dict[str, Any] = {}

    if profile.loudness_style == "radio_loud":
        preset_name = "radio_loud"
    elif profile.loudness_style == "competitive":
        preset_name = "radio_loud"
        updates["target_lufs"] = -10.8 if not profile.preserve_dynamics else -11.4
        updates["ceiling_dbfs"] = -1.0
    else:  # streaming_clean
        preset_name = "hi_fi_streaming"
        updates["target_lufs"] = -12.5
        updates["ceiling_dbfs"] = -1.0

    # Vibe shaping
    if profile.vibe == "punchy":
        updates["transient_sculpt_mix"] = 0.42
        updates["transient_sculpt_boost_db"] = 2.8
        updates["enable_softclip"] = True
    elif profile.vibe == "airy":
        updates["enable_microdetail"] = True
        updates["microdetail_amount"] = 0.22
        updates["warmth"] = -0.2
    elif profile.vibe == "warm":
        updates["warmth"] = 0.6
        updates["microdetail_amount"] = 0.14
    elif profile.vibe == "dark":
        updates["microdetail_amount"] = 0.10
        updates["warmth"] = 0.4

    # Bass / movement / hook lift
    if profile.bass_priority == "sub_heavy":
        updates["enable_hooklift"] = True
        updates["hooklift_auto"] = True
        updates["hooklift_mix"] = 0.24
    elif profile.bass_priority == "tight":
        updates["hooklift_mix"] = 0.12
        updates["transient_sculpt_crest_guard_db"] = 18.5

    if profile.vocal_priority == "high":
        updates["enable_microdetail"] = True
        updates["movement_amount"] = 0.08
    elif profile.vocal_priority == "low":
        updates["movement_amount"] = 0.12

    if profile.preserve_dynamics:
        # Loosen loudness aggression slightly
        updates.setdefault("target_lufs", -11.8 if preset_name == "radio_loud" else -12.6)

    return {
        "preset_name": preset_name,
        "overrides": updates,
        "rationale": {
            "loudness_style": profile.loudness_style,
            "vibe": profile.vibe,
            "vocal_priority": profile.vocal_priority,
            "bass_priority": profile.bass_priority,
            "preserve_dynamics": profile.preserve_dynamics,
            "notes": profile.notes,
        },
    }


def _apply_overrides_to_preset(mod: Any, preset_obj: Any, overrides: Optional[PresetOverrides]) -> Any:
    if overrides is None:
        return preset_obj

    raw = _dump_model(overrides)
    extra = raw.pop("extra", {}) or {}
    updates = {k: v for k, v in raw.items() if v is not None}
    updates.update(extra)

    if not updates:
        return preset_obj

    # replace() works with dataclass Preset instances
    try:
        return replace(preset_obj, **updates)
    except TypeError as e:
        # Return a cleaner error for model/tool layer
        raise ValueError(f"Invalid preset override keys/values: {e}") from e


def _maybe_script_auto_tune(mod: Any, preset_obj: Any, target_path: Path, reference_path: Optional[Path]) -> Any:
    """
    If your script exposes auto-tune helpers (as hinted in its CLI flow), use them.
    Otherwise no-op.
    """
    needed = ("load_audio", "analyze_track_features", "auto_select_preset_name", "auto_tune_preset")
    if not all(hasattr(mod, n) for n in needed):
        return preset_obj

    try:
        y_t, sr_t = mod.load_audio(str(target_path))
        tf = mod.analyze_track_features(y_t, sr_t)
        rf = None

        if reference_path:
            y_r, sr_r = mod.load_audio(str(reference_path))
            rf = mod.analyze_track_features(y_r, sr_r)

        presets = mod.get_presets()
        selected_name = mod.auto_select_preset_name(tf)
        candidate = presets.get(selected_name, preset_obj)
        tuned_preset, _auto_info = mod.auto_tune_preset(candidate, tf, rf)
        return tuned_preset
    except Exception:
        # Fail soft; the main mastering should still proceed
        return preset_obj


def _read_report_excerpt(report_path: Path, max_chars: int = 4000) -> Optional[str]:
    if not report_path.exists():
        return None
    try:
        txt = report_path.read_text(encoding="utf-8", errors="replace")
        return txt[:max_chars]
    except Exception:
        return None


def _run_mastering_job(job_id: str, req_dict: Dict[str, Any]) -> None:
    _update_job(job_id, status="running")

    try:
        mod = get_auralmind_module()
        req = MasteringRequest(**req_dict)

        source_path = _resolve_audio_path(req.source_audio_id, req.source_path)
        if source_path is None:
            raise ValueError("source_audio_id or source_path is required.")

        reference_path = _resolve_audio_path(req.reference_audio_id, req.reference_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        if reference_path and not reference_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")

        presets = mod.get_presets()
        if req.preset_name not in presets:
            raise ValueError(
                f"Unknown preset_name '{req.preset_name}'. Available: {sorted(list(presets.keys()))}"
            )
        preset_obj = presets[req.preset_name]

        # Optional server-side recommendation override
        recommendation_applied = None
        if req.trap_profile is not None:
            rec = _recommend_trap_params(req.trap_profile)
            recommendation_applied = rec

            # Switch preset if needed and available
            rec_preset_name = rec["preset_name"]
            if rec_preset_name in presets:
                preset_obj = presets[rec_preset_name]

            # Merge recommended updates into overrides (user overrides win)
            rec_overrides = PresetOverrides(extra=rec["overrides"])
            preset_obj = _apply_overrides_to_preset(mod, preset_obj, rec_overrides)

        # Optional script-native auto tune (if script exposes those helpers)
        if req.use_script_auto_tune:
            preset_obj = _maybe_script_auto_tune(mod, preset_obj, source_path, reference_path)

        # Final user overrides win last
        preset_obj = _apply_overrides_to_preset(mod, preset_obj, req.overrides)

        job_dir = _ensure_within(JOBS_DIR, (JOBS_DIR / job_id))
        job_dir.mkdir(parents=True, exist_ok=True)

        base = req.output_basename or Path(source_path).stem + "_master"
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "master"
        out_path = job_dir / f"{base}.wav"
        report_path = job_dir / f"{base}.md"

        result = mod.master(
            target_path=str(source_path),
            out_path=str(out_path),
            preset=preset_obj,
            reference_path=str(reference_path) if reference_path else None,
            report_path=str(report_path) if req.write_report else None,
            out_subtype=req.out_subtype,
            dither=req.dither,
            dither_seed=int(req.dither_seed),
        )

        artifacts = {
            "job_dir": str(job_dir),
            "master_wav_path": str(out_path) if out_path.exists() else None,
            "report_path": str(report_path) if report_path.exists() else None,
            "master_wav_download_url": _public_url(f"/download/{job_id}/master"),
            "report_download_url": _public_url(f"/download/{job_id}/report"),
        }

        # Attach recommendation audit if used
        if recommendation_applied:
            if is_dataclass(result) and not isinstance(result, type):
                result = asdict(result)
            elif not isinstance(result, dict):
                result = {"raw_result": str(result)}
            else:
                result = dict(result)
            result["trap_recommendation_applied"] = recommendation_applied

        report_excerpt = _read_report_excerpt(report_path) if req.write_report else None
        _update_job(
            job_id,
            status="completed",
            result=result,
            artifacts=artifacts,
            report_excerpt=report_excerpt,
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc(limit=20)
        _update_job(job_id, status="failed", error=f"{err}\n\n{tb}")


# -----------------------------
# MCP tool schemas (analysis / ingestion)
# -----------------------------

class UrlIngestRequest(BaseModel):
    url: str
    filename_hint: Optional[str] = None


class LocalRegisterRequest(BaseModel):
    """
    Register a local file path already placed inside AURALMIND_DATA_DIR.
    Safer than allowing arbitrary filesystem reads.
    """
    path: str
    label: Optional[str] = None


class AnalyzeAudioRequest(BaseModel):
    audio_id: Optional[str] = None
    path: Optional[str] = None


class JobIdRequest(BaseModel):
    job_id: str


class ReportReadRequest(BaseModel):
    job_id: str
    max_chars: int = Field(default=12000, ge=1000, le=100000)


class TrapRecommendationRequest(BaseModel):
    profile: TrapProfile


# -----------------------------
# Build MCP server
# -----------------------------

mcp = FastMCP(APP_NAME)

@mcp.tool()
def server_info() -> Dict[str, Any]:
    """
    Returns server/runtime info and whether the AuralMind script was loaded successfully.
    ChatGPT should call this early in a session before mastering.
    """
    info: Dict[str, Any] = {
        "server": APP_NAME,
        "script_path": str(SCRIPT_PATH),
        "data_dir": str(DATA_DIR),
        "inbox_dir": str(INBOX_DIR),
        "jobs_dir": str(JOBS_DIR),
        "max_upload_mb": MAX_UPLOAD_MB,
        "public_base_url": PUBLIC_BASE_URL or None,
        "time_utc": _iso_now(),
    }
    try:
        mod = get_auralmind_module()
        presets = mod.get_presets()
        info["script_loaded"] = True
        info["preset_count"] = len(presets)
        info["presets"] = sorted(list(presets.keys()))
    except Exception as e:
        info["script_loaded"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return info


@mcp.tool()
def list_presets() -> Dict[str, Any]:
    """
    List preset names and their current defaults from the AuralMind script.
    """
    mod = get_auralmind_module()
    presets = mod.get_presets()
    return {
        "preset_names": sorted(list(presets.keys())),
        "presets": {name: _preset_to_dict(obj) for name, obj in presets.items()},
    }


@mcp.tool()
def recommend_trap_mastering(req: TrapRecommendationRequest) -> Dict[str, Any]:
    """
    Deterministic recommendation helper for trap masters.
    ChatGPT can call this, then refine values based on user feedback and rerun mastering.
    """
    return _recommend_trap_params(req.profile)


@mcp.tool()
def register_local_audio(req: LocalRegisterRequest) -> Dict[str, Any]:
    """
    Register an audio file already present under AURALMIND_DATA_DIR (sandboxed).
    Useful for local/dev workflows.
    """
    p = _ensure_within(DATA_DIR, Path(req.path).resolve())
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() not in ALLOWED_AUDIO_EXTS:
        raise ValueError(f"Unsupported audio extension: {p.suffix}")
    audio_id = _register_file(p, req.label or p.name, source="local_registered")
    return {"audio_id": audio_id, "path": str(p), "name": p.name, "size_bytes": p.stat().st_size}


@mcp.tool()
def ingest_audio_from_url(req: UrlIngestRequest) -> Dict[str, Any]:
    """
    Download audio from a URL into the server inbox and register it as an audio_id.
    Includes basic SSRF protections and size limits.
    """
    url = _validate_url_for_download(req.url)
    filename = _safe_filename(req.filename_hint or Path(req.url.split("?")[0]).name or "download.wav")

    inbox_name = f"{uuid.uuid4().hex}_{filename}"
    out_path = _ensure_within(INBOX_DIR, INBOX_DIR / inbox_name)

    with requests.get(
        url,
        stream=True,
        timeout=(URL_DOWNLOAD_TIMEOUT_CONNECT, URL_DOWNLOAD_TIMEOUT_READ),
        allow_redirects=True,
    ) as resp:
        resp.raise_for_status()

        # Recheck redirect target host
        _validate_url_for_download(resp.url)

        # Optional lightweight content-type check
        ctype = (resp.headers.get("content-type") or "").lower()
        if ctype and not any(x in ctype for x in ("audio", "octet-stream", "mpeg", "mp4")):
            # Don't hard-fail because some hosts mislabel; continue but note it.
            pass

        total = _write_stream_to_file(resp, out_path, MAX_UPLOAD_BYTES)

    audio_id = _register_file(out_path, filename, source="url_ingest")
    return {
        "audio_id": audio_id,
        "stored_path": str(out_path),
        "original_name": filename,
        "size_bytes": total,
    }


@mcp.tool()
def analyze_audio(req: AnalyzeAudioRequest) -> Dict[str, Any]:
    """
    Quick audio analysis for planning mastering parameters.
    Uses soundfile for portable metadata and, if available, AuralMind helper metrics.
    """
    p = _resolve_audio_path(req.audio_id, req.path)
    if p is None:
        raise ValueError("Provide audio_id or path.")
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {p}")

    info = sf.info(str(p))
    duration = float(info.frames / info.samplerate) if info.samplerate else None

    out: Dict[str, Any] = {
        "path": str(p),
        "format": getattr(info, "format", None),
        "subtype": getattr(info, "subtype", None),
        "sample_rate": getattr(info, "samplerate", None),
        "channels": getattr(info, "channels", None),
        "frames": getattr(info, "frames", None),
        "duration_sec": duration,
    }

    # Optional deeper analysis via AuralMind script helpers, if exposed
    try:
        mod = get_auralmind_module()
        if all(hasattr(mod, n) for n in ("load_audio", "integrated_loudness_lufs")):
            y, sr = mod.load_audio(str(p))
            if hasattr(mod, "ensure_stereo"):
                y = mod.ensure_stereo(y)

            out["auralmind_sample_rate"] = int(sr)
            out["lufs_integrated"] = float(mod.integrated_loudness_lufs(y, sr))

            if all(hasattr(mod, n) for n in ("true_peak_estimate", "lin_to_db")):
                tp_lin = mod.true_peak_estimate(y, sr, oversample=4)
                out["true_peak_dbfs_est"] = float(mod.lin_to_db(tp_lin + 1e-12))

            if hasattr(mod, "analyze_track_features"):
                try:
                    feats = mod.analyze_track_features(y, sr)
                    # Convert to plain JSON-safe values where possible
                    if isinstance(feats, dict):
                        clean = {}
                        for k, v in feats.items():
                            try:
                                clean[k] = float(v) if isinstance(v, (int, float)) else v
                            except Exception:
                                clean[k] = str(v)
                        out["track_features"] = clean
                    else:
                        out["track_features"] = str(feats)
                except Exception as e:
                    out["track_features_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        out["auralmind_analysis_error"] = f"{type(e).__name__}: {e}"

    return out


@mcp.tool()
def start_mastering_job(req: MasteringRequest) -> Dict[str, Any]:
    """
    Queue a mastering job.
    Recommended ChatGPT flow:
    1) analyze_audio
    2) recommend_trap_mastering (optional)
    3) start_mastering_job
    4) get_mastering_job_status until completed
    5) read_mastering_report / fetch artifacts
    """
    if not req.source_audio_id and not req.source_path:
        raise ValueError("source_audio_id or source_path is required")

    job_id = _create_job_record(req)
    EXECUTOR.submit(_run_mastering_job, job_id, _dump_model(req))
    return {
        "job_id": job_id,
        "status": "queued",
        "poll_with": "get_mastering_job_status",
        "hint": "Poll every few seconds. Use read_mastering_report when completed.",
    }


@mcp.tool()
def get_mastering_job_status(req: JobIdRequest) -> Dict[str, Any]:
    """
    Poll job status and receive result/error/artifact metadata.
    """
    with JOBS_LOCK:
        job = JOBS.get(req.job_id)
        if not job:
            raise ValueError(f"Unknown job_id: {req.job_id}")
        return dict(job)


@mcp.tool()
def read_mastering_report(req: ReportReadRequest) -> Dict[str, Any]:
    """
    Read the generated markdown report (truncated) for a completed job.
    """
    with JOBS_LOCK:
        job = JOBS.get(req.job_id)
        if not job:
            raise ValueError(f"Unknown job_id: {req.job_id}")

    artifacts = job.get("artifacts") or {}
    report_path = artifacts.get("report_path")
    if not report_path:
        raise ValueError("No report artifact found for this job.")
    rp = _ensure_within(JOBS_DIR, Path(report_path))
    if not rp.exists():
        raise FileNotFoundError(f"Report not found: {rp}")

    txt = rp.read_text(encoding="utf-8", errors="replace")
    return {
        "job_id": req.job_id,
        "report_path": str(rp),
        "report_text": txt[: req.max_chars],
        "truncated": len(txt) > req.max_chars,
    }


@mcp.tool()
def list_artifacts(req: JobIdRequest) -> Dict[str, Any]:
    """
    Return artifact paths and optional download URLs for a job.
    """
    with JOBS_LOCK:
        job = JOBS.get(req.job_id)
        if not job:
            raise ValueError(f"Unknown job_id: {req.job_id}")
    return {
        "job_id": req.job_id,
        "status": job["status"],
        "artifacts": job.get("artifacts") or {},
    }


# -----------------------------
# REST helper app (upload/download/health)
# -----------------------------

app = FastAPI(title=APP_NAME)

# Optional CORS for your own frontend; tighten in production
try:
    from fastapi.middleware.cors import CORSMiddleware

    allow_origins = [
        o.strip() for o in os.getenv("ALLOW_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


@app.get("/api/health")
def health() -> Dict[str, Any]:
    script_ok = True
    script_err = None
    try:
        get_auralmind_module()
    except Exception as e:
        script_ok = False
        script_err = f"{type(e).__name__}: {e}"

    return {
        "ok": True,
        "server": APP_NAME,
        "time_utc": _iso_now(),
        "script_ok": script_ok,
        "script_error": script_err,
        "jobs_total": len(JOBS),
        "files_total": len(FILES),
    }


@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Manual upload path (helpful if ChatGPT attachment passthrough is not available in your setup).
    After upload, pass returned audio_id to MCP tools.
    """
    original_name = _safe_filename(file.filename or "upload.wav")
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {ext}")

    out_name = f"{uuid.uuid4().hex}_{original_name}"
    out_path = _ensure_within(INBOX_DIR, INBOX_DIR / out_name)

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB} MB.")
            f.write(chunk)

    audio_id = _register_file(out_path, original_name, source="rest_upload")

    return {
        "audio_id": audio_id,
        "stored_path": str(out_path),
        "original_name": original_name,
        "size_bytes": total,
    }


@app.get("/download/{job_id}/{artifact}")
def download_artifact(job_id: str, artifact: Literal["master", "report"]):
    """
    Download generated artifacts from a mastering job.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        artifacts = job.get("artifacts") or {}

    key = "master_wav_path" if artifact == "master" else "report_path"
    p_str = artifacts.get(key)
    if not p_str:
        raise HTTPException(status_code=404, detail=f"No {artifact} artifact for job {job_id}")

    p = _ensure_within(JOBS_DIR, Path(p_str))
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Artifact missing on disk: {p}")

    media_type = "audio/wav" if artifact == "master" else "text/markdown"
    filename = p.name
    return FileResponse(str(p), media_type=media_type, filename=filename)


# Mount MCP ASGI sub-application at /mcp.
# Using "/mcp" avoids shadowing the REST routes defined above.
mcp_app = mcp.http_app(path="/mcp")
app.mount("/mcp", mcp_app)


# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"[{APP_NAME}] starting on {HOST}:{PORT}")
    print(f"[{APP_NAME}] MCP endpoint: /mcp")
    print(f"[{APP_NAME}] AuralMind script path: {SCRIPT_PATH}")
    uvicorn.run(app, host=HOST, port=PORT)