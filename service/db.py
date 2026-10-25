"""Работа с базой данных для журнала событий."""
import sqlite3
import json
from datetime import datetime

DB_PATH = "linq_safety.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            event_type TEXT,
            severity TEXT,
            status TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_event(camera_id: str, event_type: str, confidence: float, image_path: str):
    # Логика критичности: падение - Высокая, телефон - Низкая
    severity = "Средняя"
    if "Падение" in event_type or "Огонь" in event_type or "страховки" in event_type:
        severity = "Высокая"
    elif "телефон" in event_type:
        severity = "Низкая"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO events (timestamp, camera_id, event_type, severity, status, confidence, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, event_type, severity, "Новое", confidence, image_path))
    conn.commit()
    conn.close()

def get_events(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(ix) for ix in rows]

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type')
    stats = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return stats