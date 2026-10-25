"""Воркер задач: читает видео/фото, гоняет FrameChain через CloudBackend (RabbitMQ->TPU),
пишет события и evidence-кадры. Один поток, очередь задач — TPU не перегружаем."""
import json
import queue
import threading
import time
from pathlib import Path

import cv2

from linq.backend.cloud import CloudBackend
from linq.config import Settings
from service import storage
from service.chain import FrameChain
from service.events import EventRegistry
from service.scenarios import ScenarioConfig, ScenarioEngine, Zone

ZONES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'zones.json'


def load_zones(path=ZONES_PATH):
    if not path.exists():
        return [], 1.0
    data = json.loads(path.read_text(encoding='utf-8'))
    zones = [Zone(z['name'], z['kind'], [tuple(p) for p in z['poly']])
             for z in data.get('zones', [])]
    return zones, float(data.get('analysis_fps', 1.0))


class _MeteredBackend:
    """Прокси поверх CloudBackend: считаем latency каждого вызова TPU."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink
        self.name = inner.name

    def run(self, program, tensors):
        out = self.inner.run(program, tensors)
        self.sink(self.inner.last_latency_s)
        return out

    def close(self):
        self.inner.close()


class Worker:
    def __init__(self, zones, analysis_fps=1.0, cfg=None):
        self.zones = zones
        self.analysis_fps = analysis_fps
        self.cfg = cfg or ScenarioConfig()
        self.queue = queue.Queue()
        self.cloud = CloudBackend(Settings())
        self._lock = threading.Lock()
        self._lat = []
        self._calls = 0
        self._frames = 0
        self._started = None
        self._stopping = False

    # --- lifecycle ---
    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stopping = True
        self.queue.put(None)

    def submit(self, job_id):
        self.queue.put(job_id)

    def _push_lat(self, v):
        with self._lock:
            self._lat.append(v)
            if len(self._lat) > 500:
                self._lat = self._lat[-500:]
            self._calls += 1

    def get_metrics(self):
        with self._lock:
            lat = sorted(self._lat[-200:])
            calls = self._calls
        n = len(lat)
        return {
            'calls': calls,
            'frames': self._frames,
            'analysis_fps': self.analysis_fps,
            'lat_avg_ms': round(sum(lat) / n * 1000, 1) if n else None,
            'lat_p50_ms': round(lat[n // 2] * 1000, 1) if n else None,
            'lat_p95_ms': round(lat[max(0, int(0.95 * n) - 1)] * 1000, 1) if n else None,
            'queue': self.queue.qsize(),
            'started_at': self._started,
        }

    # --- main loop ---
    def _loop(self):
        while not self._stopping:
            job_id = self.queue.get()
            if job_id is None:
                break
            try:
                self._run_job(job_id)
            except Exception as e:
                storage.update_job(job_id, status='error',
                                   error=f'{type(e).__name__}: {e}',
                                   finished_at=time.time())
            finally:
                self.queue.task_done()

    def _run_job(self, job_id):
        self._started = self._started or time.time()
        job = storage.get_job(job_id)
        path = Path(job['filename'])
        storage.update_job(job_id, status='running', started_at=time.time())

        registry = EventRegistry(confirm_frames=self.cfg.confirm_frames,
                                 grace_frames=self.cfg.grace_frames)
        engine = ScenarioEngine(self.zones, self.cfg, registry)
        chain = FrameChain(_MeteredBackend(self.cloud, self._push_lat), engine)

        state = {'frame': None}

        def on_open(ev):
            ev.id = storage.open_event(job_id, ev)
            if state['frame'] is not None:
                name = f'ev_{job_id}_{ev.id}.jpg'
                cv2.imwrite(str(storage.EVIDENCE_DIR / name), state['frame'],
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                storage.set_event_evidence(ev.id, name)

        registry.on_open, registry.on_hit, registry.on_close = on_open, storage.touch_event, storage.close_event

        if job['kind'] == 'image':
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f'не читается изображение {path}')
            storage.update_job(job_id, frames_total=1)
            state['frame'] = frame
            chain.process(frame, 0.0, job_id)
            registry.flush(1.0)
            with self._lock:
                self._frames += 1
            storage.update_job(job_id, frames_done=1, status='done', finished_at=time.time())
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'не открывается видео {path}')
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps_src / self.analysis_fps)))
        storage.update_job(job_id, frames_total=total)

        idx = done = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                state['frame'] = frame
                chain.process(frame, idx / fps_src, job_id)
                done += 1
                with self._lock:
                    self._frames += 1
                storage.update_job(job_id, frames_done=idx + 1)
            idx += 1
        cap.release()
        registry.flush(idx / fps_src if fps_src else idx)
        storage.update_job(job_id, status='done', finished_at=time.time())