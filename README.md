# AuralMind FastMCP Mastering Server (Heroku Ready)

A production-oriented **FastMCP server** that wraps your AuralMind Python mastering script so ChatGPT (via Developer Mode) can plan and run mastering jobs using tool-calls.

## Features

- ✅ **MCP tools** for:
  - health check
  - list/get presets
  - trap-master planning (preset + safe overrides)
  - start job
  - poll job status
  - list recent jobs
- ✅ **MCP resources** for:
  - mastered audio
  - reports
  - result JSON
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
AURALMIND_JOBS_DIR=./jobs
ALLOW_LOCAL_FILES=true
AURALMIND_DEFAULT_PRESET=competitive_trap
JOB_MAX_WORKERS=1
FASTMCP_HTTP_PATH=/mcp
```

### 3) Run

```powershell
python server.py
```

Or:
```powershell
fastmcp run server.py --transport http --host 127.0.0.1 --port 3333
```

### 4) Smoke checks

- MCP endpoint: `http://127.0.0.1:3333/mcp`
- Use your MCP client to call `server_health` and `list_presets`

---

## Heroku Deployment

### 1) Create app
```bash
heroku create your-auralmind-mcp
```

### 2) Set config vars
```bash
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
- MCP endpoint: `https://your-auralmind-mcp.herokuapp.com/mcp`
- Use your MCP client to call `server_health`
- inspect logs:
```bash
heroku logs --tail
```

> Heroku dyno storage is ephemeral. For production, persist artifacts in S3/R2/GCS and return signed URLs.

---

## ChatGPT Developer Mode (MCP) Connection Flow

1. Deploy this server to a **public HTTPS** URL.
2. Enable ChatGPT **Developer Mode** (if available on your plan/workspace).
3. Add your custom connector/app and point it to your MCP URL (e.g., `https://your-auralmind-mcp.herokuapp.com/mcp`).
4. Ask ChatGPT to:
   - call `plan_trap_master`
   - call `start_master_job_from_upload` for uploaded bytes (or `start_master_job` for URLs)
   - poll `get_job_status`
   - use `read_resource` on the returned resource URIs to fetch artifacts

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
- `source_url` (remote URL)
- or `local_target_path` (dev only)
- optional reference URL/path
- preset + overrides

### `start_master_job_from_upload`
Queues a mastering job using:
- base64 audio bytes (`source_data_base64`)
- `source_filename` for format detection
- optional base64 reference + filename
- preset + overrides

### `get_job_status`
Polling endpoint/tool for long-running renders.

Artifacts are exposed as MCP resources:
- `auralmind://jobs/<job_id>/mastered_audio`
- `auralmind://jobs/<job_id>/report`
- `auralmind://jobs/<job_id>/result_json`

Use `read_resource` (or `list_resources`) to fetch them. Binary content is base64-encoded when accessed via the tool transform.

### `list_recent_jobs`
Debug/resume helper.

---

## Environment Variables

- `AURALMIND_JOBS_DIR` - working dir for job files (`/tmp/auralmind_jobs` by default)
- `MAX_JSON_MB` - base64 payload cap (applied to MCP tool inputs)
- `MAX_UPLOAD_MB` - decoded audio size cap for base64 uploads
- `MAX_DOWNLOAD_MB` - remote URL download cap
- `ALLOWED_DOWNLOAD_HOSTS` - optional comma-separated hostname allowlist
- `ALLOW_LOCAL_FILES` - dev-only local path mode (`false` in production)
- `AURALMIND_SCRIPT_PATH` - path to your mastering script
- `AURALMIND_DEFAULT_PRESET` - fallback preset name
- `JOB_MAX_WORKERS` - thread pool workers
- `FASTMCP_HTTP_PATH` - HTTP endpoint path (default `/mcp`)

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

### MCP endpoint path
FastMCP serves the HTTP endpoint at `/mcp` by default.  
Set `FASTMCP_HTTP_PATH` if you need a custom path and update your client URL.

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
