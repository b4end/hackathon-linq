"""FastAPI: uvicorn service.app:app --host 0.0.0.0 --port 8000"""
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from linq.registry import PROGRAMS
from service import storage
from service.worker import Worker, load_zones

BASE = Path(__file__).resolve().parents[1]
BENCH_DIR = BASE / 'logs'
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp'}

storage.init_db()
worker: Worker | None = None


@asynccontextmanager
async def lifespan(app):
    global worker
    zones, afps = load_zones()
    worker = Worker(zones, analysis_fps=afps)
    worker.start()
    yield
    worker.stop()


app = FastAPI(title='LinQ Safety', lifespan=lifespan)
app.mount('/evidence', StaticFiles(directory=storage.EVIDENCE_DIR), name='evidence')
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')


@app.get('/')
def index():
    return FileResponse(BASE / 'static' / 'index.html')


@app.post('/api/uploads')
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or '').suffix.lower()
    kind = 'video' if ext in VIDEO_EXT else 'image' if ext in IMAGE_EXT else None
    if kind is None:
        raise HTTPException(415, 'формат не поддерживается')
    job_id = uuid.uuid4().hex[:8]
    dest = storage.UPLOAD_DIR / f'{job_id}{ext}'
    dest.write_bytes(await file.read())
    storage.create_job(job_id, str(dest), kind)
    worker.submit(job_id)
    return {'job_id': job_id}


@app.get('/api/jobs')
def jobs():
    return storage.list_jobs()


@app.get('/api/jobs/{job_id}')
def job(job_id: str):
    row = storage.get_job(job_id)
    if row is None:
        raise HTTPException(404)
    return dict(row)


@app.get('/api/events')
def events(etype: str = '', status: str = '', job_id: str = '', limit: int = 50):
    return storage.list_events(etype or None, status or None, job_id or None, limit)


@app.post('/api/events/{eid}/status')
def event_status(eid: int, payload: dict):
    st = payload.get('status')
    if st not in ('new', 'confirmed', 'rejected'):
        raise HTTPException(400, 'status: new|confirmed|rejected')
    storage.set_event_status(eid, st)
    return {'ok': True}


@app.get('/api/analytics')
def analytics():
    return storage.analytics()


@app.get('/api/performance')
def performance():
    bench = []
    if BENCH_DIR.exists():
        for p in sorted(BENCH_DIR.glob('bench_*.json'), reverse=True)[:8]:
            try:
                bench.append(json.loads(p.read_text(encoding='utf-8')))
            except Exception:
                pass
    return {
        'live': worker.get_metrics() if worker else {},
        'bench': bench,
        'programs': [{'name': s.name, 'file': s.file} for s in PROGRAMS.values()],
    }


@app.get('/api/models_info')
def models_info():
    p = BASE / 'config' / 'models_info.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


@app.get('/api/zones')
def zones():
    p = BASE / 'config' / 'zones.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'zones': []}

@app.post('/api/jobs/{job_id}/retry')
def retry_job(job_id: str):
    if storage.get_job(job_id) is None:
        raise HTTPException(404)
    storage.update_job(job_id, status='queued', error=None,
                       frames_done=0, started_at=None, finished_at=None)
    worker.submit(job_id)
    return {'ok': True}