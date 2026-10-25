"""Правила нарушений + полигоны зон. Чистая логика: без сети и без очередей."""
import math
from dataclasses import dataclass

from linq import registry as R
from service.tracker import iou


@dataclass
class Zone:
    name: str
    kind: str          # forbidden | height | smoking_allowed
    poly: list         # [(x, y), ...] в пикселях кадра


@dataclass
class ScenarioConfig:
    run_px_s: float = 220.0          # порог «бега», px/с
    machine_move_px_s: float = 15.0  # машина считается движущейся
    prox_iou: float = 0.01           # пересечение человек-машина
    phone_contain: float = 0.5       # доля телефона внутри человека
    smoke_person_iou: float = 0.02   # дым/сигарета касаются человека
    fall_torso_deg: float = 35.0     # торс к горизонтали ниже => лежит
    fall_aspect: float = 1.6         # w/h рамки как запасной признак
    still_px_s: float = 12.0         # «неподвижен»
    crop_pad: float = 0.15           # запас кропа человека
    confirm_frames: int = 2
    grace_frames: int = 3


def point_in_poly(x, y, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def foot(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def box_in_zone(box, zone):
    x1, y1, x2, y2 = box
    pts = [foot(box), (x1, y1), (x2, y1), (x1, y2), (x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2)]
    return any(point_in_poly(px, py, zone.poly) for px, py in pts)


def pad_box(box, frac, img_wh):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    W, H = img_wh
    return (max(0, x1 - w * frac), max(0, y1 - h * frac), min(W, x2 + w * frac), min(H, y2 + h * frac))


def contain(outer, inner):
    ix1 = max(outer[0], inner[0]); iy1 = max(outer[1], inner[1])
    ix2 = min(outer[2], inner[2]); iy2 = min(outer[3], inner[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def is_lying(pose_det, fall_deg):
    kps = pose_det.keypoints
    if len(kps) < 17:
        return False
    sl, sr, hl, hr = kps[R.KP_SH_L], kps[R.KP_SH_R], kps[R.KP_HIP_L], kps[R.KP_HIP_R]
    if min(sl[2], sr[2], hl[2], hr[2]) < 0.3:
        return False
    sx, sy = (sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2
    hx, hy = (hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2
    dx, dy = hx - sx, hy - sy
    ang_hor = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9))   # 90 = стоит, 0 = лежит
    return ang_hor < fall_deg


class ScenarioEngine:
    def __init__(self, zones, cfg=None, event_registry=None):
        self.zones = zones
        self.cfg = cfg or ScenarioConfig()
        self.registry = event_registry

    def _zones(self, kind):
        return [z for z in self.zones if z.kind == kind]

    def request_crops(self, person_tracks, img_wh):
        """Люди в зонах высоты -> кропы для harness-модели."""
        out = []
        for tr in person_tracks:
            if any(box_in_zone(tr.box, z) for z in self._zones('height')):
                out.append((tr, pad_box(tr.box, self.cfg.crop_pad, img_wh)))
        return out

    def evaluate(self, src, t, person_tracks, machine_tracks, phone_dets,
                 pose_dets, smoke_dets, harness_by_track):
        """Возвращает hits: {key: (etype, confidence, meta)} для EventRegistry.tick."""
        cfg, hits = self.cfg, {}

        for tr in person_tracks:
            if tr.speed_px_s() > cfg.run_px_s:                                   # 01 бег
                hits[(src, 'running', tr.track_id)] = ('running', 0.9, {})
            for mt in machine_tracks:                                            # 02 сближение
                if mt.speed_px_s() > cfg.machine_move_px_s and iou(tr.box, mt.box) > cfg.prox_iou:
                    hits[(src, 'proximity', tr.track_id)] = ('proximity', 0.8,
                                                             {'machine_track': mt.track_id})
            for z in self._zones('forbidden'):                                   # 05 перелез/пролез
                if box_in_zone(tr.box, z):
                    hits[(src, 'forbidden', tr.track_id)] = ('forbidden', 0.9, {'zone': z.name})
            if any(box_in_zone(tr.box, z) for z in self._zones('height')):       # 06 высота
                hd = harness_by_track.get(tr.track_id, [])
                protected = any(d.class_id in R.HARNESS_OK_CLASSES for d in hd)
                if not protected:
                    missing = any(d.class_id == R.CLS_HARNESS_MISSING for d in hd)
                    at_height = any(d.class_id == R.CLS_PERSON_AT_HEIGHT for d in hd)
                    hits[(src, 'height', tr.track_id)] = ('height', 0.8 if missing else 0.6, {
                        'harness_missing': missing,
                        'person_at_height': at_height,
                        'harness_dets': len(hd),
                    })
        for ph in phone_dets:                                                    # 03 телефон
            for tr in person_tracks:
                if contain(tr.box, ph.box) >= cfg.phone_contain:
                    hits[(src, 'phone', tr.track_id)] = ('phone', ph.score, {})
                    break
        for s in smoke_dets:                                                     # 04 курение/огонь
            owner = next((tr for tr in person_tracks
                          if iou(tr.box, s.box) > cfg.smoke_person_iou), None)
            if owner is not None:
                hits[(src, 'smoking', owner.track_id)] = ('smoking', s.score, {'what': s.class_id})
            else:
                cx, cy = (s.box[0] + s.box[2]) / 2, (s.box[1] + s.box[3]) / 2
                if not any(point_in_poly(cx, cy, z.poly) for z in self._zones('smoking_allowed')):
                    hits[(src, 'fire', 'open_area')] = ('fire', s.score, {'what': s.class_id})
        for pd in pose_dets:                                                     # 07 падение
            tr = next((tr for tr in person_tracks if iou(tr.box, pd.box) > 0.3), None)
            if tr is None:
                continue
            w, h = pd.box[2] - pd.box[0], pd.box[3] - pd.box[1]
            lying = is_lying(pd, cfg.fall_torso_deg) or (h > 0 and w / h > cfg.fall_aspect)
            if lying and tr.speed_px_s(window=3.0) < cfg.still_px_s:
                hits[(src, 'fall', tr.track_id)] = ('fall', pd.score, {})
        return hits