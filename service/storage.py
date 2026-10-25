"""Журнал задач и событий: SQLite + evidence-кадры на диске (всё в data/, вне гита)."""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DB_PATH = DATA_DIR / 'linq_safety.db'
EVIDENCE_DIR = DATA_DIR / 'evidence'
UPLOAD_DIR = DATA_DIR / 'uploads'

_lock = threading.Lock()


@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, _db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, filename TEXT, kind TEXT, status TEXT,
            created_at REAL, started_at REAL, finished_at REAL,
            frames_total INTEGER DEFAULT 0, frames_done INTEGER DEFAULT 0,
            error TEXT)''')
        con.execute('''CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, etype TEXT, state TEXT,
            started_t REAL, last_t REAL, ended_t REAL, conf REAL, meta TEXT,
            evidence TEXT, status TEXT DEFAULT 'new', created_at REAL)''')


# --- jobs ---
def create_job(job_id, filename, kind):
    with _lock, _db() as con:
        con.execute('INSERT INTO jobs(id,filename,kind,status,created_at) VALUES(?,?,?,?,?)',
                    (job_id, filename, kind, 'queued', time.time()))


def update_job(job_id, **fields):
    if not fields:
        return
    cols = ', '.join(f'{k}=?' for k in fields)
    with _lock, _db() as con:
        con.execute(f'UPDATE jobs SET {cols} WHERE id=?', (*fields.values(), job_id))


def get_job(job_id):
    with _db() as con:
        return con.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()


def list_jobs(limit=20):
    with _db() as con:
        return [dict(r) for r in con.execute(
            'SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,))]


# --- events ---
def open_event(job_id, ev):
    with _lock, _db() as con:
        cur = con.execute(
            'INSERT INTO events(job_id,etype,state,started_t,last_t,conf,meta,created_at)'
            ' VALUES(?,?,?,?,?,?,?,?)',
            (job_id, ev.etype, 'open', ev.started, ev.last_t, ev.conf,
             json.dumps(ev.meta, ensure_ascii=False), time.time()))
        return cur.lastrowid


def touch_event(ev):
    with _lock, _db() as con:
        con.execute('UPDATE events SET last_t=?, conf=?, meta=? WHERE id=?',
                    (ev.last_t, ev.conf, json.dumps(ev.meta, ensure_ascii=False), ev.id))


def close_event(ev):
    with _lock, _db() as con:
        con.execute("UPDATE events SET state='closed', ended_t=? WHERE id=?",
                    (ev.ended_t, ev.id))


def set_event_evidence(event_id, name):
    with _lock, _db() as con:
        con.execute('UPDATE events SET evidence=? WHERE id=?', (name, event_id))


def set_event_status(event_id, status):
    with _lock, _db() as con:
        con.execute('UPDATE events SET status=? WHERE id=?', (status, event_id))


def list_events(etype=None, status=None, job_id=None, limit=50):
    q, args = 'SELECT * FROM events WHERE 1=1', []
    if etype:
        q += ' AND etype=?'; args.append(etype)
    if status:
        q += ' AND status=?'; args.append(status)
    if job_id:
        q += ' AND job_id=?'; args.append(job_id)
    q += ' ORDER BY id DESC LIMIT ?'; args.append(limit)
    with _db() as con:
        rows = [dict(r) for r in con.execute(q, args)]
    for r in rows:
        r['meta'] = json.loads(r['meta'] or '{}')
    return rows


def analytics():
    with _db() as con:
        by_type = {r['etype']: r['n'] for r in con.execute(
            'SELECT etype, COUNT(*) n FROM events GROUP BY etype')}
        by_status = {r['status']: r['n'] for r in con.execute(
            'SELECT status, COUNT(*) n FROM events GROUP BY status')}
    conf, rej = by_status.get('confirmed', 0), by_status.get('rejected', 0)
    return {'by_type': by_type, 'by_status': by_status,
            'precision': round(conf / (conf + rej), 3) if conf + rej else None}