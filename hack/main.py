import os, json, time, threading, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import tpu_worker

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"; UPLOADS.mkdir(exist_ok=True)
EVENTS  = BASE / "events";  EVENTS.mkdir(exist_ok=True)
STATE_F = BASE / "state.json"

app = FastAPI()
app.mount("/events", StaticFiles(directory=str(EVENTS)), name="events")
templates = Jinja2Templates(directory=str(BASE / "templates"))

# in-memory state (одновременно только одна задача — прототип)
state = {
    "job_id": None,
    "video": None,
    "status": "idle",          # idle | running | done | error
    "progress": 0.0,           # 0..1
    "events": [],              # список событий
    "metrics": {
        "frames_total": 0,
        "frames_done": 0,
        "fps": 0.0,
        "latency_ms_avg": 0.0,
        "latency_ms_p95": 0.0,
        "started_at": None,
        "finished_at": None,
        "error": None,
    },
}
state_lock = threading.Lock()

def _save_state():
    STATE_F.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def index(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})

@app.post("/upload")
async def upload(file: UploadFile = File(...), bg: BackgroundTasks = None):
    video_id = uuid.uuid4().hex[:8]
    ext = Path(file.filename).suffix or ".mp4"
    path = UPLOADS / f"{video_id}{ext}"
    with open(path, "wb") as f:
        f.write(await file.read())

    with state_lock:
        state["job_id"] = video_id
        state["video"] = str(path)
        state["status"] = "running"
        state["progress"] = 0.0
        state["events"] = []
        state["metrics"] = {
            "frames_total": 0, "frames_done": 0,
            "fps": 0.0, "latency_ms_avg": 0.0, "latency_ms_p95": 0.0,
            "started_at": time.time(), "finished_at": None, "error": None,
        }
        _save_state()

    # запускаем воркер в отдельном потоке
    t = threading.Thread(target=tpu_worker.run, args=(str(path), state, state_lock, _save_state), daemon=True)
    t.start()
    return {"job_id": video_id, "status": "running"}

@app.get("/status")
def status():
    with state_lock:
        # отдаём копию, чтобы не держать лок
        snap = json.loads(json.dumps(state, default=str))
    return snap

@app.get("/events.json")
def events_json():
    with state_lock:
        return state["events"]

@app.get("/event_frame/{name}")
def event_frame(name: str):
    p = EVENTS / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")