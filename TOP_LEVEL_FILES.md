# Top-Level File Documentation

This project is a **Heroku-ready FastMCP + FastAPI server** that wraps your AuralMind mastering script and exposes MCP tools + HTTP endpoints for long-running mastering jobs.

## Root Files

### `server.py`
Main application entrypoint.

**Responsibilities**
- Starts the FastAPI app
- Registers MCP tools (via FastMCP)
- Mounts the MCP transport app
- Exposes HTTP endpoints for:
  - health checks
  - presets
  - job creation (upload / URL)
  - job status polling
  - artifact downloads (mastered WAV + report + result JSON)
- Runs a thread pool for long mastering jobs
- Includes URL ingestion safeguards (basic SSRF guardrails + byte limits)

**When to edit**
- Add/modify MCP tools
- Add API endpoints
- Change job orchestration behavior
- Tighten auth/CORS

---

### `requirements.txt`
Python dependencies for local dev + Heroku deployment.

Includes:
- `fastapi`, `uvicorn`, `pydantic`
- `fastmcp`
- `httpx`
- `python-multipart`
- audio/script runtime deps (`numpy`, `scipy`, `soundfile`)

> If your AuralMind script adds new imports (e.g., `librosa`, `pyloudnorm`, `demucs`), add them here.

---

### `Procfile`
Heroku process definition.

Tells Heroku how to run the web process:
- `uvicorn server:app --host 0.0.0.0 --port $PORT`

---

### `.python-version`
Pins Python version for Heroku buildpack consistency (and helps local consistency).

---

### `.env.example`
Template for environment variables.

Copy to `.env` locally and fill in values:
- `MCP_PUBLIC_BASE_URL`
- `AURALMIND_JOBS_DIR`
- size limits
- allowed download hosts
- script path
- worker count

---

### `app.json`
Optional Heroku manifest documenting config vars and app metadata.

Useful if you want “Deploy to Heroku” workflows or just a clean reference of expected env vars.

---

### `.gitignore`
Prevents local secrets, caches, and runtime job artifacts from being committed.

---

### `README.md`
Setup + usage guide:
- local run
- Heroku deploy
- ChatGPT Developer Mode connection flow
- MCP tool usage
- troubleshooting

## Package Folder

### `auralmind_engine/`
A thin compatibility layer around your uploaded AuralMind script.

#### `auralmind_engine/engine_adapter.py`
Dynamic import + safe override validation.

**Responsibilities**
- Load the mastering script at runtime
- Confirm required functions exist (`get_presets`, `master`)
- Validate/limit LLM overrides
- Apply dataclass preset overrides safely
- Call `master(...)` and normalize result JSON

#### `auralmind_engine/auralmind_match_maestro_v7_3_expert1.py`
Your actual AuralMind mastering DSP engine (vendored copy of your uploaded file).

#### `auralmind_engine/__init__.py`
Package exports.

## Runtime Artifact Layout (`AURALMIND_JOBS_DIR`)

Each job gets its own folder:
- input target audio
- optional reference audio
- mastered output WAV
- markdown report
- JSON result
- status snapshot JSON

Example:
```text
/tmp/auralmind_jobs/<job_id>/
  input_target.wav
  reference.wav              # optional
  mastered.wav
  report.md                  # optional (if engine writes one)
  result.json
  job_status.json
```

## Safe Extension Order (recommended)

1. Add auth (API key / OAuth / signed requests)
2. Move artifacts to object storage (S3/R2/GCS)
3. Add audio analysis MCP tool (LUFS/TP/crest/stereo)
4. Add queue backend (RQ/Celery/Arq) if concurrency grows
5. Add job cancellation + timeout enforcement
