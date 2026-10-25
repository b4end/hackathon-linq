"""Сборка кадра: 3 полных вызова TPU + точечные кропы harness, затем трекер и правила."""
from PIL import Image

from linq import registry as R
from linq.yolo import (letterbox_params, postprocess_yolo_detect, postprocess_yolo_pose,
                       preprocess_yolo, to_original)
from service.tracker import Tracker


class FrameChain:
    def __init__(self, backend, engine, tracker=None,
                 conf_detect=0.3, conf_pose=0.35, conf_smoke=0.3, conf_harness=0.3):
        self.backend, self.engine = backend, engine
        self.tracker = tracker or Tracker()
        self.conf_detect, self.conf_pose = conf_detect, conf_pose
        self.conf_smoke, self.conf_harness = conf_smoke, conf_harness

    def process(self, frame_bgr, t, src):
        img = Image.fromarray(frame_bgr[..., ::-1])
        scale, px, py = letterbox_params(img)
        tensors = {'images': preprocess_yolo(img)}

        det = self.backend.run('yolo_detect', dict(tensors))['output0']
        pose = self.backend.run('yolo_pose', dict(tensors))['output0']
        smoke = self.backend.run('smoke', dict(tensors))['output0']

        # Должно быть так:
        d_det = to_original(postprocess_yolo_detect(det, conf_thr=self.conf_detect), scale, px, py)
        d_pose = to_original(postprocess_yolo_pose(pose, conf_thr=self.conf_pose), scale, px, py)  # ← yolo_pose, не yolo_detect!
        d_smoke = to_original(postprocess_yolo_detect(smoke, conf_thr=self.conf_smoke, class_ids=R.SMOKE_CLASSES), scale, px, py)

        persons = [d for d in d_det if d.class_id == R.CLS_PERSON]
        machines = [d for d in d_det if d.class_id in R.CLS_VEHICLES]
        phones = [d for d in d_det if d.class_id == R.CLS_PHONE]

        self.tracker.update(persons + machines, t)
        ptracks = self.tracker.active(R.CLS_PERSON)
        mtracks = self.tracker.active_any(R.CLS_VEHICLES)

        # harness — только кропами людей, попавших в зоны высоты (экономим вызовы)
        harness_by_track = {}
        for tr, box in self.engine.request_crops(ptracks, img.size):
            crop = img.crop(tuple(int(v) for v in box))
            out = self.backend.run('harness', {'images': preprocess_yolo(crop)})['output0']
            harness_by_track[tr.track_id] = postprocess_yolo_detect(out, conf_thr=self.conf_harness)

        hits = self.engine.evaluate(src, t, ptracks, mtracks, phones, d_pose, d_smoke,
                                    harness_by_track)
        self.engine.registry.tick(hits, t)
        return hits