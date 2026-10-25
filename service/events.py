"""Жизненный цикл события: candidate -> open -> closed, с grace-окном «вернулось = то же»."""
from dataclasses import dataclass, field


@dataclass
class Event:
    key: tuple
    etype: str
    state: str                 # candidate | open | closed
    first_t: float
    started: float
    last_t: float
    ended_t: float | None = None
    conf: float = 0.0
    hits: int = 0
    missed: int = 0
    meta: dict = field(default_factory=dict)
    evidence: str | None = None
    id: int | None = None


class EventRegistry:
    def __init__(self, confirm_frames=2, grace_frames=3, on_open=None, on_hit=None, on_close=None):
        self.confirm_frames, self.grace_frames = confirm_frames, grace_frames
        self.on_open, self.on_hit, self.on_close = on_open, on_hit, on_close
        self.live: dict = {}
        self.closed: list = []

    def tick(self, hits: dict, t: float):
        """hits: {key: (etype, confidence, meta)} по текущему кадру."""
        for key, ev in list(self.live.items()):
            if key in hits:
                etype, conf, meta = hits[key]
                ev.last_t, ev.hits, ev.missed = t, ev.hits + 1, 0
                ev.conf = max(ev.conf, conf)
                ev.meta.update(meta)
                if ev.state == 'candidate' and ev.hits >= self.confirm_frames:
                    ev.state, ev.started = 'open', t
                    if self.on_open:
                        self.on_open(ev)
                elif ev.state == 'open' and self.on_hit:
                    self.on_hit(ev)
            else:
                ev.missed += 1
                if ev.missed > self.grace_frames:
                    self._close(ev)
        for key, (etype, conf, meta) in hits.items():
            if key in self.live:
                continue
            ev = Event(key=key, etype=etype, state='candidate', first_t=t, started=t,
                       last_t=t, conf=conf, hits=1, meta=dict(meta))
            self.live[key] = ev
            if self.confirm_frames <= 1:
                ev.state = 'open'
                if self.on_open:
                    self.on_open(ev)

    def _close(self, ev):
        ev.state, ev.ended_t = 'closed', ev.last_t
        self.live.pop(ev.key, None)
        self.closed.append(ev)
        if self.on_close:
            self.on_close(ev)

    def flush(self, t):
        """Конец видео: закрыть всё открытое, кандидатов выбросить."""
        for ev in list(self.live.values()):
            if ev.state == 'open':
                self._close(ev)
        self.live.clear()