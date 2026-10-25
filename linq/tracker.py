"""Лёгкий трекер по IoU: связывает детекции между кадрами и считает скорости."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Track:
    track_id: int
    class_id: int
    box: tuple
    last_t: float
    points: list = field(default_factory=list)   # (t, cx, cy)
    misses: int = 0

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def speed_px_s(self, window=2.0):
        pts = [p for p in self.points if p[0] >= self.last_t - window]
        if len(pts) < 2:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        dt = t1 - t0
        return float(np.hypot(x1 - x0, y1 - y0) / dt) if dt > 1e-6 else 0.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    def __init__(self, iou_thr=0.2, max_misses=2):
        self.iou_thr, self.max_misses = iou_thr, max_misses
        self.tracks = []
        self._next = 1

    def update(self, dets, t):
        for tr in self.tracks:
            tr.misses += 1
        pairs = []
        for d in dets:
            for tr in self.tracks:
                if tr.class_id == d.class_id:
                    v = iou(tr.box, d.box)
                    if v >= self.iou_thr:
                        pairs.append((v, d, tr))
        pairs.sort(key=lambda p: -p[0])
        used_d, used_t = set(), set()
        for v, d, tr in pairs:
            if id(d) in used_d or tr.track_id in used_t:
                continue
            used_d.add(id(d)); used_t.add(tr.track_id)
            tr.box, tr.last_t, tr.misses = d.box, t, 0
            tr.points.append((t, *tr.center))
        for d in dets:
            if id(d) not in used_d:
                tr = Track(self._next, d.class_id, d.box, t, [(t, *d.center)])
                self._next += 1
                self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        return self.tracks

    def active(self, class_id):
        return [tr for tr in self.tracks if tr.misses == 0 and tr.class_id == class_id]

    def active_any(self, class_ids):
        return [tr for tr in self.tracks if tr.misses == 0 and tr.class_id in class_ids]