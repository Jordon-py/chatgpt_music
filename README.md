# AuralMind FastMCP Mastering Server (Heroku Ready)

A production-oriented **FastMCP + FastAPI scaffold** that wraps your AuralMind Python mastering script so ChatGPT (via Developer Mode) can plan and run mastering jobs using tool-calls.

## Features

- ✅ **MCP tools** for:
  - health check
  - list/get presets
  - trap-master planning (preset + safe overrides)
  - start job
  - poll job status
  - list recent jobs
- ✅ **HTTP API** for:
  - multipart uploads
  - URL-based job creation
  - status polling
  - WAV/report/result downloads
- ✅ **Heroku-ready**:
  - `Procfile`
  - `.python-version`
  - env-driven config
  - `/tmp` job storage default
- ✅ **LLM safety guardrails**:
  - override allowlist + value ranges
  - URL host/IP validation (basic SSRF protection)
  - upload/download size caps
- ✅ **Long-running job support**:
  - thread pool
  - job IDs
  - polling workflow

---

## Project Structure

```text
.
├─ server.py
├─ requirements.txt
├─ Procfile
├─ .python-version
├─ .env.example
├─ app.json
├─ .gitignore
├─ README.md
├─ TOP_LEVEL_FILES.md
└─ auralmind_engine/
   ├─ __init__.py
   ├─ engine_adapter.py
   └─ auralmind_match_maestro_v7_3_expert1.py
```

---

## Quick Start (Local)

### 1) Create and activate a virtual environment (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Create `.env`

```powershell
Copy-Item .env.example .env
```

Recommended local values:
```env
MCP_PUBLIC_BASE_URL=http://127.0.0.1:8000
AURALMIND_JOBS_DIR=./jobs
ALLOW_LOCAL_FILES=true
AURALMIND_DEFAULT_PRESET=competitive_trap
JOB_MAX_WORKERS=1
```

### 3) Run

```powershell
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

### 4) Smoke checks

- `GET /healthz`
- `GET /api/presets`
- MCP endpoint path is mounted by FastMCP (commonly `/mcp`, depending on your installed `fastmcp` version)

---

## Heroku Deployment

### 1) Create app
```bash
heroku create your-auralmind-mcp
```

### 2) Set config vars
```bash
heroku config:set MCP_PUBLIC_BASE_URL=https://your-auralmind-mcp.herokuapp.com
heroku config:set AURALMIND_JOBS_DIR=/tmp/auralmind_jobs
heroku config:set ALLOW_LOCAL_FILES=false
heroku config:set AURALMIND_DEFAULT_PRESET=competitive_trap
heroku config:set JOB_MAX_WORKERS=1
```

Optional hardening:
```bash
heroku config:set ALLOWED_DOWNLOAD_HOSTS=your-storage-domain.com
```

### 3) Deploy
```bash
git init
git add .
git commit -m "Add AuralMind FastMCP server"
heroku git:remote -a your-auralmind-mcp
git push heroku main
```

### 4) Verify
- open `https://your-auralmind-mcp.herokuapp.com/healthz`
- inspect logs:
```bash
heroku logs --tail
```

> Heroku dyno storage is ephemeral. For production, persist artifacts in S3/R2/GCS and return signed URLs.

---

## ChatGPT Developer Mode (MCP) Connection Flow

1. Deploy this server to a **public HTTPS** URL.
2. Enable ChatGPT **Developer Mode** (if available on your plan/workspace).
3. Add your custom connector/app and point it to your MCP URL.
4. Ask ChatGPT to:
   - call `plan_trap_master`
   - call `start_master_job`
   - poll `get_job_status`
   - return the download/report links

> If you are testing locally first, use a secure tunnel (Cloudflare Tunnel / ngrok) to expose your local server over HTTPS.

---

## MCP Tool Overview (what the LLM can call)

### `server_health`
Returns:
- engine readiness
- preset count
- job dir
- limits and flags

### `list_presets`
Returns preset list with a compact summary.

### `get_preset`
Returns a full preset object so the LLM can plan safe, precise overrides.

### `plan_trap_master`
Takes high-level musical intent (style/loudness/brightness/width/punch/etc.) and returns:
- selected preset
- safe overrides
- rationale list

### `start_master_job`
Queues a mastering job using:
- `source_url` (recommended for remote/ChatGPT)
- or `local_target_path` (dev only)
- optional reference URL/path
- preset + overrides

### `get_job_status`
Polling endpoint/tool for long-running renders.

### `list_recent_jobs`
Debug/resume helper.

---

## HTTP API Quick Examples

### Create a job from URL
```bash
curl -X POST "http://127.0.0.1:8000/api/jobs/from-url" \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://example.com/path/song.wav",
    "preset": "competitive_trap",
    "overrides": {
      "target_lufs": -10.2,
      "hooklift_auto": true,
      "transient_sculpt_mix": 0.28,
      "out_subtype": "PCM_24",
      "dither": true
    }
  }'
```

### Create a job by multipart upload
```bash
curl -X POST "http://127.0.0.1:8000/api/jobs/upload" \
  -F "target_file=@C:/path/to/song.wav" \
  -F "preset=competitive_trap" \
  -F "overrides_json={\"target_lufs\":-10.2,\"out_subtype\":\"PCM_24\",\"dither\":true}"
```

### Poll job status
```bash
curl "http://127.0.0.1:8000/api/jobs/<job_id>"
```

### Download mastered WAV
```bash
curl -L "http://127.0.0.1:8000/api/jobs/<job_id>/download" -o mastered.wav
```

---

## Environment Variables

- `MCP_PUBLIC_BASE_URL` — used to generate absolute URLs in responses
- `AURALMIND_JOBS_DIR` — working dir for job files (`/tmp/auralmind_jobs` by default)
- `MAX_UPLOAD_MB` — multipart upload cap
- `MAX_DOWNLOAD_MB` — remote URL download cap
- `ALLOWED_DOWNLOAD_HOSTS` — optional comma-separated hostname allowlist
- `ALLOW_LOCAL_FILES` — dev-only local path mode (`false` in production)
- `AURALMIND_SCRIPT_PATH` — path to your mastering script
- `AURALMIND_DEFAULT_PRESET` — fallback preset name
- `JOB_MAX_WORKERS` — thread pool workers

---

## Security Notes

This scaffold includes **basic** protections, not a full security model.

Included:
- private/local IP blocking for URL ingestion
- optional hostname allowlist
- file size limits
- override allowlist + ranges
- job files constrained to a private jobs directory

You should still add:
- authentication/authorization
- request signing or API keys
- rate limiting
- durable object storage
- malware scanning / media validation (if accepting public uploads)

---

## Troubleshooting

### `fastmcp` mount/path issues
FastMCP versions can differ (`http_app(...)` vs `streamable_http_app()` behavior).  
`server.py` includes a compatibility fallback in `_build_mcp_app()`.

### Engine import fails
- Verify `AURALMIND_SCRIPT_PATH`
- Confirm your script exports `get_presets()` and `master(...)`
- Check dependency imports in `requirements.txt`

### Jobs are slow / memory-heavy on Heroku
- set `JOB_MAX_WORKERS=1`
- avoid stem separation on small dynos
- use shorter test files first
- move to a bigger dyno or offload rendering to a worker process

---

## Next Upgrades (recommended)

1. Add **audio analysis tool** (`analyze_audio`) so the LLM can inspect LUFS/TP/crest before mastering.
2. Add **S3/R2 storage** so completed outputs survive dyno restarts.
3. Add **auth** so only your UI/LLM can call the server.
4. Add **queue backend** (RQ/Celery/Arq) for better resilience and cancellation support.
