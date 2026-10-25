"""YOLO-специфичный препроцессинг и постпроцессинг.
Препроцессинг: letterbox 640×640 с паддингом 114, нормализация /255, CHW float32.
Постпроцессинг: декод (cx,cy,w,h)→(x1,y1,x2,y2), NMS по классам, возврат списка Detection.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Detection:
    class_id: int
    score: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) в пикселях исходного кадра
    keypoints: Optional[Tuple] = None        # для pose-модели: 17 точек (x, y, conf)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# --- препроцессинг (ровно как в их example.ipynb) ---
def preprocess_yolo(image: Image.Image, hw: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """PIL → float32 NCHW 640×640 letterbox."""
    img = np.asarray(image.convert("RGB"))
    h, w = img.shape[:2]
    ratio = min(hw[0] / h, hw[1] / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (hw[1] - new_w) / 2.0
    dh = (hw[0] - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1))[None]


def _letterbox_params(image: Image.Image, hw=(640, 640)):
    """Возвращает (ratio, pad_w, pad_h) для обратного масштабирования боксов."""
    w, h = image.size
    ratio = min(hw[0] / h, hw[1] / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_w = (hw[1] - new_w) / 2.0
    pad_h = (hw[0] - new_h) / 2.0
    return ratio, pad_w, pad_h


letterbox_params = _letterbox_params


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def postprocess_yolo_detect(
    raw_output: np.ndarray,
    conf_thr: float = 0.3,
    iou_thr: float = 0.45,
    class_ids: Optional[Tuple[int, ...]] = None,
) -> List[Detection]:
    """Декод сырого выхода YOLOv8 detect (shape 1,84,8400) → список Detection."""
    out = np.asarray(raw_output)
    if out.shape != (1, 84, 8400):
        raise ValueError(f"Ожидался выход (1, 84, 8400), получен {out.shape}. "
                         "Модель не YOLOv8 detect или не тот выходной тензор.")
    if not np.issubdtype(out.dtype, np.floating):
        raise ValueError("Получен не float-выход — проверьте распаковку результата.")
    if not np.isfinite(out).all():
        raise ValueError("В выходе TPU есть NaN/Inf.")

    preds = out[0].T.astype(np.float32)          # (8400, 84)
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4:]
    class_ids_arr = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    keep = confs > conf_thr
    boxes_xywh = boxes_xywh[keep]
    class_ids_arr = class_ids_arr[keep]
    confs = confs[keep]

    if class_ids is not None:
        cls_mask = np.isin(class_ids_arr, np.array(list(class_ids), dtype=np.int64))
        boxes_xywh = boxes_xywh[cls_mask]
        class_ids_arr = class_ids_arr[cls_mask]
        confs = confs[cls_mask]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    detections: List[Detection] = []
    for cls in np.unique(class_ids_arr):
        pos = np.flatnonzero(class_ids_arr == cls)
        idx = cv2.dnn.NMSBoxes(
            boxes_xyxy[pos].tolist(),
            confs[pos].tolist(),
            conf_thr, iou_thr,
        )
        for li in np.asarray(idx, dtype=np.int64).reshape(-1):
            i = pos[li]
            detections.append(Detection(
                class_id=int(class_ids_arr[i]),
                score=float(confs[i]),
                box=tuple(float(v) for v in boxes_xyxy[i]),
            ))
    return sorted(detections, key=lambda d: d.score, reverse=True)


def postprocess_yolo_pose(
    raw_output: np.ndarray,
    conf_thr: float = 0.35,
    iou_thr: float = 0.45,
) -> List[Detection]:
    """Декод выхода YOLOv8-pose. Формат зависит от экспорта:
       - если (1, 56, 8400): первые 4 — box, 5 — obj_conf, 6..22 — 17 keypoints×3, остальное — классы;
       - если (1, 17, 3, 8400) или (1, 17, 8400): только keypoints.
       Для простоты обрабатываем вариант (1, 56, 8400) — стандарт ultralytics export."""
    out = np.asarray(raw_output)
    if out.ndim != 3:
        raise ValueError(f"pose-выход должен быть 3D, получен {out.shape}")

    # Нормализуем к виду (N, 56, 8400)
    if out.shape[0] == 1 and out.shape[1] == 56 and out.shape[2] == 8400:
        pass
    elif out.shape[0] == 1 and out.shape[1] == 17 and out.shape[2] in (3, 8400):
        # только keypoints без bboxes — такой формат нам не подходит без детекта
        raise ValueError(f"Неподдерживаемый pose-формат {out.shape}; нужен (1,56,8400)")
    else:
        raise ValueError(f"Неизвестный pose-формат {out.shape}")

    preds = out[0].T.astype(np.float32)          # (8400, 56)
    boxes_xywh = preds[:, :4]
    obj_conf = preds[:, 4]
    kpts_flat = preds[:, 5:5 + 17 * 3]           # (8400, 51)
    cls_scores = preds[:, 5 + 17 * 3:]           # (8400, 0) для pose-модели обычно пусто

    keep = obj_conf > conf_thr
    boxes_xywh = boxes_xywh[keep]
    obj_conf = obj_conf[keep]
    kpts_flat = kpts_flat[keep]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    detections: List[Detection] = []
    for i in range(boxes_xyxy.shape[0]):
        kpts = kpts_flat[i].reshape(17, 3)       # (x, y, conf) для каждой точки
        detections.append(Detection(
            class_id=0,                           # person
            score=float(obj_conf[i]),
            box=tuple(float(v) for v in boxes_xyxy[i]),
            keypoints=tuple((float(k[0]), float(k[1]), float(k[2])) for k in kpts),
        ))
    return detections


def to_original(detections: List[Detection], ratio: float, pad_w: float, pad_h: float) -> List[Detection]:
    """Пересчёт боксов и keypoints из letterbox-координат обратно в исходный кадр."""
    out = []
    for d in detections:
        x1, y1, x2, y2 = d.box
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio
        new_box = (x1, y1, x2, y2)
        new_kpts = None
        if d.keypoints is not None:
            new_kpts = tuple(
                (((k[0] - pad_w) / ratio, (k[1] - pad_h) / ratio, k[2]) for k in d.keypoints)
            )
        out.append(Detection(d.class_id, d.score, new_box, new_kpts))
    return out


PREPROCESSORS = {'yolo': preprocess_yolo}
POSTPROCESSORS = {
    'yolo_detect': postprocess_yolo_detect,
    'yolo_pose': postprocess_yolo_pose,
}