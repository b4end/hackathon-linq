"""Гоняются без VPN: pytest -q"""
from linq import registry as R
from linq.yolo import Detection
from service.events import EventRegistry
from service.scenarios import ScenarioConfig, ScenarioEngine, Zone
from service.tracker import Track


def mktrack(tid, box, speed=0.0, cls=0):
    tr = Track(track_id=tid, class_id=cls, box=box, last_t=1.0)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    tr.points = [(0.0, cx - speed, cy), (1.0, cx, cy)]
    return tr


def engine(zones=None):
    return ScenarioEngine(zones or [], ScenarioConfig(confirm_frames=1, grace_frames=1),
                          EventRegistry(confirm_frames=1, grace_frames=1))


def test_phone_smoking_fire():
    e = engine()
    p = mktrack(1, (100, 100, 200, 300))
    phone = Detection(R.CLS_PHONE, 0.9, (140, 120, 160, 150))
    cig = Detection(R.CLS_CIGARETTE, 0.7, (180, 140, 210, 180))      # у человека -> курение
    smoke = Detection(R.CLS_SMOKE, 0.6, (500, 500, 530, 530))        # сам по себе -> огонь
    hits = e.evaluate('cam', 1.0, [p], [], [phone], [], [cig, smoke], {})
    assert ('cam', 'phone', 1) in hits
    assert ('cam', 'smoking', 1) in hits
    assert ('cam', 'fire', 'open_area') in hits


def test_height_with_and_without_harness():
    zones = [Zone('height1', 'height', [(0, 0), (640, 0), (640, 200), (0, 200)])]
    e = engine(zones)
    p = mktrack(1, (100, 50, 150, 190))
    hits = e.evaluate('cam', 1.0, [p], [], [], [], [], {1: []})
    assert ('cam', 'height', 1) in hits                              # привязи не видно -> событие
    worn = [Detection(R.CLS_HARNESS_WORN, 0.8, (20, 20, 90, 180))]
    hits = e.evaluate('cam', 2.0, [p], [], [], [], [], {1: worn})
    assert ('cam', 'height', 1) not in hits                          # привязь надета -> тишина
    missing = [Detection(R.CLS_HARNESS_MISSING, 0.7, (20, 20, 90, 180))]
    hits = e.evaluate('cam', 3.0, [p], [], [], [], [], {1: missing})
    assert hits[('cam', 'height', 1)][2]['harness_missing'] is True


def test_forbidden_running_proximity():
    zones = [Zone('fence1', 'forbidden', [(300, 300), (400, 300), (400, 400), (300, 400)])]
    e = engine(zones)
    p_fence = mktrack(2, (320, 320, 360, 390))
    p_run = mktrack(3, (10, 400, 60, 500), speed=300)
    mach = mktrack(9, (55, 400, 120, 500), speed=60, cls=2)
    hits = e.evaluate('cam', 1.0, [p_fence, p_run], [mach], [], [], [], {})
    assert ('cam', 'forbidden', 2) in hits
    assert ('cam', 'running', 3) in hits
    assert ('cam', 'proximity', 3) in hits


def test_fall_by_pose():
    e = engine()
    box = (100, 240, 180, 270)
    kps = [(0, 0, 0.0)] * 17
    kps[5], kps[6] = (110, 250, 1.0), (120, 252, 1.0)
    kps[11], kps[12] = (160, 255, 1.0), (170, 256, 1.0)
    pose = Detection(0, 0.8, box, tuple(kps))
    p = mktrack(1, box)
    hits = e.evaluate('cam', 1.0, [p], [], [], [pose], [], {})
    assert ('cam', 'fall', 1) in hits