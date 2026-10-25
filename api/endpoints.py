"""
План таков:
    В api/endpoints.py, что логично, будут написаны все эндпоинты и объявлен роутер.
    Уже потом, в main.py, будет создан app (:class:`fastapi.FastAPI`).
    Полностью убрать код, сгенерированый нейросетью.

Для запуска:
    uvicorn main:app --reload --port 8000
Если АПИ не работает, попробуйте убрать :code:`root_path` из :data:`app`.
"""

from __future__ import annotations
from sqlite3 import connect, Row
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage" / "evidence"
DB_PATH = BASE_DIR / "db.db"
DEDUP_COOLDOWN_SEC = 60  # "одно нарушение = одно оповещение": пауза повторов для пары (камера, тип, трек)


# справочники
class EventType(str, Enum):
    running = "running"        # бег / резкие перемещения
    proximity = "proximity"    # опасное сближение с техникой
    phone = "phone"            # телефон
    climbing = "climbing"      # перелез / пролез под
    fall = "fall"              # падение / неподвижность
    smoking = "smoking"        # курение / огонь
    harness = "harness"        # высота без привязи

EVENT_TYPE_LABELS = {
    EventType.running: "Бег / резкие перемещения",
    EventType.proximity: "Опасное сближение с техникой",
    EventType.phone: "Использование телефона",
    EventType.climbing: "Перелезание / пролезание под",
    EventType.fall: "Падение / неподвижность",
    EventType.smoking: "Курение / открытый огонь",
    EventType.harness: "Работа на высоте без привязи",
}
DEFAULT_SEVERITY = {
    EventType.running: "medium", EventType.proximity: "high", EventType.phone: "medium",
    EventType.climbing: "high", EventType.fall: "critical", EventType.smoking: "medium",
    EventType.harness: "high",
}

class Severity(str, Enum):
    low = "low"; medium = "medium"; high = "high"; critical = "critical"

class EventStatus(str, Enum):
    new = "new"; confirmed = "confirmed"; rejected = "rejected"; closed = "closed"

DEFAULT_CAMERAS = [
    ("CAM-01", "Линия сборки", "Цех A", "online"),
    ("CAM-02", "Склад материалов", "Склад", "online"),
    ("CAM-03", "Конвейер 2", "Цех Б", "online"),
    ("CAM-04", "Погрузочная зона", "Участок C", "offline"),
]


# модели данных
class EventIn(BaseModel):
    """То, что присылает нейросеть/воркер на одно обнаруженное нарушение."""
    type: EventType
    camera_id: str
    zone: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    severity: Optional[Severity] = None
    track_id: Optional[int] = None          # id трека объекта — ключ дедупликации
    video_time_sec: Optional[float] = None  # время события внутри видео, сек
    frame_index: Optional[int] = None
    timestamp: Optional[datetime] = None    # момент фиксации (по умолчанию — сейчас)
    details: Optional[str] = None

class StatusUpdate(BaseModel):
    """Действие оператора в карточке события."""
    status: EventStatus
    comment: Optional[str] = None

class PerfIn(BaseModel):
    """Замер скорости инференса на LinQ (обязательное требование кейса)."""
    model: str
    device: str = "linq_tpu_h1m"
    fps: float
    latency_ms: float
    total_s: Optional[float] = None
    frames_processed: Optional[int] = None
    repeats: Optional[int] = None
    errors: int = 0
    notes: Optional[str] = None
    measured_at: Optional[datetime] = None

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    zone: Optional[str] = None
    status: Optional[Literal["online", "offline"]] = None


# база
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()

@contextmanager
def db():
    con = connect(DB_PATH)
    con.row_factory = Row
    try:
        yield con
        con.commit()
    finally:
        con.close()

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_code TEXT UNIQUE,
  type TEXT NOT NULL, camera_id TEXT NOT NULL, zone TEXT,
  severity TEXT NOT NULL, confidence REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',
  track_id INTEGER, video_time_sec REAL, frame_index INTEGER,
  details TEXT, operator_comment TEXT, evidence_path TEXT,
  timestamp_iso TEXT NOT NULL, created_iso TEXT NOT NULL, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cameras(
  id TEXT PRIMARY KEY, name TEXT, zone TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS performance(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model TEXT NOT NULL, device TEXT, fps REAL, latency_ms REAL, total_s REAL,
  frames_processed INTEGER, repeats INTEGER, errors INTEGER DEFAULT 0,
  notes TEXT, measured_iso TEXT NOT NULL, ts REAL NOT NULL
);
"""

def init_db():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with db() as con:
        con.executescript(SCHEMA)
        if con.execute("SELECT COUNT(*) c FROM cameras").fetchone()["c"] == 0:
            con.executemany("INSERT INTO cameras(id,name,zone,status) VALUES(?,?,?,?)", DEFAULT_CAMERAS)

init_db()


# приложение
app = FastAPI(title="LinQ", description="Прототип.", root_path='/api')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

class WSManager:
    def __init__(self): self.conns = []
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.conns.append(ws)
    def disconnect(self, ws: WebSocket):
        if ws in self.conns: self.conns.remove(ws)
    async def broadcast(self, payload: dict):
        for ws in list(self.conns):
            try: await ws.send_json(payload)
            except Exception: self.conns.remove(ws)

ws_manager = WSManager()

def row_to_event(r: Row) -> dict:
    return {
        "id": r["event_code"],
        "type": r["type"],
        "type_label": EVENT_TYPE_LABELS.get(EventType(r["type"]), r["type"]),
        "camera_id": r["camera_id"],
        "zone": r["zone"],
        "severity": r["severity"],
        "confidence": r["confidence"],
        "status": r["status"],
        "track_id": r["track_id"],
        "video_time_sec": r["video_time_sec"],
        "frame_index": r["frame_index"],
        "details": r["details"],
        "operator_comment": r["operator_comment"],
        "evidence_url": f"/storage/evidence/{r['evidence_path']}" if r["evidence_path"] else None,
        "timestamp": r["timestamp_iso"],
        "created_at": r["created_iso"],
    }

def get_event(con, code: str) -> Optional[Row]:
    return con.execute("SELECT * FROM events WHERE event_code=?", (code,)).fetchone()

# ingest
@app.post("/ingest/event", status_code=201)
async def ingest_event(data: EventIn):
    ts = data.timestamp or now_utc()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    with db() as con:
        # дедупликация: то же нарушение (камера+тип+трек) внутри cooldown не плодит оповещения
        cutoff = now_utc().timestamp() - DEDUP_COOLDOWN_SEC
        candidates = con.execute(
            "SELECT * FROM events WHERE camera_id=? AND type=? AND status IN ('new','confirmed') AND ts>=?",
            (data.camera_id, data.type.value, cutoff)).fetchall()
        for r in candidates:
            if (r["track_id"] or None) == (data.track_id or None):
                return JSONResponse(status_code=200, content={
                    "created": False, "deduplicated": True,
                    "event_id": r["event_code"], "event": row_to_event(r)})
        zone = data.zone
        if zone is None:
            cam = con.execute("SELECT zone FROM cameras WHERE id=?", (data.camera_id,)).fetchone()
            zone = cam["zone"] if cam else None
        cur = con.execute(
            """INSERT INTO events(event_code,type,camera_id,zone,severity,confidence,status,track_id,
               video_time_sec,frame_index,details,operator_comment,evidence_path,timestamp_iso,created_iso,ts)
               VALUES(NULL,?,?,?,?,?,'new',?,?,?,?,NULL,NULL,?,?,?)""",
            (data.type.value, data.camera_id, zone, (data.severity or Severity(DEFAULT_SEVERITY[data.type])).value,
             data.confidence, data.track_id, data.video_time_sec, data.frame_index, data.details,
             ts.isoformat(), now_utc().isoformat(), ts.timestamp()))
        code = f"EV-{1000 + cur.lastrowid}"
        con.execute("UPDATE events SET event_code=? WHERE id=?", (code, cur.lastrowid))
        row = con.execute("SELECT * FROM events WHERE id=?", (cur.lastrowid,)).fetchone()
    event = row_to_event(row)
    await ws_manager.broadcast({"kind": "new_event", "event": event})
    return {"created": True, "deduplicated": False, "event_id": code, "event": event}

@app.post("/events/{event_id}/evidence")
async def upload_evidence(event_id: str, file: UploadFile = File(...)):
    """Кадр-доказательство от воркера. Доступен фронтенду по /storage/evidence/<имя>."""
    with db() as con:
        row = get_event(con, event_id)
        if row is None:
            raise HTTPException(404, f"Событие {event_id} не найдено")
        ext = Path(file.filename or "frame.bin").suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            ext = ".bin"
        fname = f"{event_id}{ext}"
        (STORAGE_DIR / fname).write_bytes(await file.read())
        con.execute("UPDATE events SET evidence_path=? WHERE event_code=?", (fname, event_id))
        return row_to_event(get_event(con, event_id))

@app.post("/ingest/performance", status_code=201)
async def ingest_perf(data: PerfIn):
    measured = data.measured_at or now_utc()
    with db() as con:
        cur = con.execute(
            """INSERT INTO performance(model,device,fps,latency_ms,total_s,frames_processed,repeats,errors,notes,measured_iso,ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (data.model, data.device, data.fps, data.latency_ms, data.total_s,
             data.frames_processed, data.repeats, data.errors, data.notes,
             measured.isoformat(), to_epoch(measured)))
        row = con.execute("SELECT * FROM performance WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)

# read
@app.get("/")
def root():
    return {"service": "LinQ Safety API", "version": "0.1.0", "docs": "/docs", "health": "/api/health"}

@app.get("/health")
def health():
    return {"status": "ok", "time": now_utc().isoformat()}

@app.get("/events")
def list_events(
    type: Optional[EventType] = None,
    camera_id: Optional[str] = None,
    zone: Optional[str] = None,
    status: Optional[EventStatus] = None,
    severity: Optional[Severity] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    q: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Журнал событий с фильтрами по типу, камере, зоне, статусу, времени + поиск."""
    where, params = [], []
    if type: where.append("type=?"); params.append(type.value)
    if camera_id: where.append("camera_id=?"); params.append(camera_id)
    if zone: where.append("zone=?"); params.append(zone)
    if status: where.append("status=?"); params.append(status.value)
    if severity: where.append("severity=?"); params.append(severity.value)
    if date_from: where.append("ts>=?"); params.append(to_epoch(date_from))
    if date_to: where.append("ts<=?"); params.append(to_epoch(date_to))
    if q:
        where.append("(details LIKE ? OR event_code LIKE ? OR camera_id LIKE ? OR zone LIKE ?)")
        params += [f"%{q}%"] * 4
    w = f"WHERE {' AND '.join(where)}" if where else ""
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) c FROM events {w}", params).fetchone()["c"]
        rows = con.execute(f"SELECT * FROM events {w} ORDER BY ts DESC LIMIT ? OFFSET ?",
                           params + [limit, offset]).fetchall()
    return {"items": [row_to_event(r) for r in rows], "total": total, "limit": limit, "offset": offset}

@app.get("/events/{event_id}")
def get_event_api(event_id: str):
    with db() as con:
        row = get_event(con, event_id)
    if row is None:
        raise HTTPException(404, f"Событие {event_id} не найдено")
    return row_to_event(row)

@app.post("/events/{event_id}/status")
async def set_status(event_id: str, body: StatusUpdate):
    """Подтвердить / отклонить / закрыть событие оператором."""
    with db() as con:
        row = get_event(con, event_id)
        if row is None:
            raise HTTPException(404, f"Событие {event_id} не найдено")
        comment = body.comment if body.comment is not None else row["operator_comment"]
        con.execute("UPDATE events SET status=?, operator_comment=? WHERE event_code=?",
                    (body.status.value, comment, event_id))
        event = row_to_event(get_event(con, event_id))
    await ws_manager.broadcast({"kind": "event_status", "event_id": event_id, "status": body.status.value, "event": event})
    return event

@app.get("/analytics")
def analytics():
    """Цифры для экрана «Аналитика»: всё считается из журнала событий."""
    with db() as con:
        rows = con.execute("SELECT * FROM events").fetchall()
        cams = con.execute("SELECT id FROM cameras").fetchall()
    by_type = {t.value: 0 for t in EventType}
    by_camera = {c["id"]: 0 for c in cams}
    by_severity = {s.value: 0 for s in Severity}
    by_status = {s.value: 0 for s in EventStatus}
    by_day = {(now_utc() - timedelta(days=i)).date().isoformat(): 0 for i in range(6, -1, -1)}
    by_hour = {h: 0 for h in range(24)}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        by_camera[r["camera_id"]] = by_camera.get(r["camera_id"], 0) + 1
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        dt = datetime.fromtimestamp(r["ts"], tz=timezone.utc)
        by_day[dt.date().isoformat()] = by_day.get(dt.date().isoformat(), 0) + 1
        by_hour[dt.hour] += 1
    return {
        "totals": {"all": len(rows), **{k: by_status[k] for k in by_status}},
        "by_type": by_type,
        "type_labels": {t.value: lbl for t, lbl in EVENT_TYPE_LABELS.items()},
        "by_camera": by_camera,
        "by_severity": by_severity,
        "by_status": by_status,
        "by_day": [{"day": d, "count": c} for d, c in by_day.items()],
        "by_hour": [{"hour": h, "count": c} for h, c in by_hour.items()],
    }

@app.get("/overview")
def overview():
    """Цифры для экрана «Обзор»."""
    today = now_utc().date().isoformat()
    with db() as con:
        rows = con.execute("SELECT * FROM events").fetchall()
        cams = con.execute("SELECT * FROM cameras").fetchall()
        recent = con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT 5").fetchall()
    todays = [r for r in rows if datetime.fromtimestamp(r["ts"], tz=timezone.utc).date().isoformat() == today]
    return {
        "cameras_total": len(cams),
        "cameras_online": sum(1 for c in cams if c["status"] == "online"),
        "events_today": len(todays),
        "require_check": sum(1 for r in rows if r["status"] == "new"),
        "confirmed_today": sum(1 for r in todays if r["status"] == "confirmed"),
        "rejected_today": sum(1 for r in todays if r["status"] == "rejected"),
        "by_type": {t.value: sum(1 for r in rows if r["type"] == t.value) for t in EventType},
        "cameras": [dict(c) for c in cams],
        "recent_events": [row_to_event(r) for r in recent],
    }

@app.get("/performance")
def get_performance():
    """Страница «Производительность LinQ». Пока замеров нет — not_configured (как в макете)."""
    with db() as con:
        rows = con.execute("SELECT * FROM performance ORDER BY ts DESC LIMIT 20").fetchall()
    if not rows:
        return {"status": "not_configured", "latest": None, "history": []}
    return {"status": "ok", "latest": dict(rows[0]), "history": [dict(r) for r in rows]}

@app.get("/cameras")
def list_cameras():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM cameras ORDER BY id").fetchall()]

@app.put("/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate):
    with db() as con:
        row = con.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"Камера {camera_id} не найдена")
        con.execute("UPDATE cameras SET name=?, zone=?, status=? WHERE id=?",
                    (body.name or row["name"], body.zone or row["zone"], body.status or row["status"], camera_id))
        return dict(con.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone())

# живые алерты для UI
@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # соединение удерживается, чтобы клиент мог слать ping
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)