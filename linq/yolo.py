"""Подготовка кадров и разбор ответов для моделей семейства YOLO."""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# точки тела модели Б (номера по стандарту COCO)
KP_NOSE, KP_SHOULDER_L, KP_SHOULDER_R = 0, 5, 6
KP_HIP_L, KP_HIP_R = 11, 12
KP_ANKLE_L, KP_ANKLE_R = 15, 16

# классы модели А, которые используем в правилах
CLASS_PERSON, CLASS_CELL_PHONE = 0, 67
CLASS_VEHICLES = (2, 3, 5, 7)  # car, motorcycle, bus, truck
CLASS_NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus',
               7: 'truck', 67: 'cell phone'}


@dataclass
class Detection:
    """Один найденный объект: класс, уверенность, рамка (x1,y1,x2,y2) в пикселях кадра."""
    class_id: int
    score: float
    box: tuple
    keypoints: tuple = field(default_factory=tuple)  # для модели Б: ((x, y, conf), ... 17)


def letterbox_params(image: Image.Image, hw=(640, 640)):
    """Как вписать кадр в квадрат: масштаб и отступы слева/сверху."""
    w, h = image.size
    th, tw = hw
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    return scale, (tw - nw) // 2, (th - nh) // 2


def preprocess_yolo(image: Image.Image, hw=(640, 640)) -> np.ndarray:
    """Кадр -> тензор: вписали с серыми полями, цвета 0..1, порядок каналов CHW."""
    th, tw = hw
    scale, pad_x, pad_y = letterbox_params(image, hw)
    nw, nh = int(image.size[0] * scale), int(image.size[1] * scale)
    canvas = Image.new('RGB', (tw, th), (114, 114, 114))
    canvas.paste(image.resize((nw, nh)), (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _nms(boxes, scores, iou_thr):
    """Убираем дубли: из перекрывающихся рамок оставляем самую уверенную."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    picked = []
    while order.size:
        i = order[0]
        picked.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return picked


def _decode_boxes(t):
    """Из «центр+размер» в «углы»."""
    x1 = t[:, 0] - t[:, 2] / 2
    y1 = t[:, 1] - t[:, 3] / 2
    x2 = t[:, 0] + t[:, 2] / 2
    y2 = t[:, 1] + t[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_yolo_detect(tensor: np.ndarray, conf_thr=0.35, iou_thr=0.5,
                            class_ids=None):
    """Ответ модели А -> список Detection (координаты пока в квадрате 640)."""
    t = tensor[0].T                       # (N, 4 + число классов)
    scores_all = t[:, 4:]
    class_id = np.argmax(scores_all, axis=1)
    score = np.max(scores_all, axis=1)
    keep = score > conf_thr
    if class_ids is not None:
        keep &= np.isin(class_id, class_ids)
    t, class_id, score = t[keep], class_id[keep], score[keep]
    if not len(t):
        return []
    boxes = _decode_boxes(t[:, :4])
    picked = _nms(boxes, score, iou_thr)
    return [Detection(int(class_id[i]), float(score[i]),
                      tuple(float(v) for v in boxes[i])) for i in picked]


def postprocess_yolo_pose(tensor: np.ndarray, conf_thr=0.4, iou_thr=0.5):
    """Ответ модели Б -> список Detection с точками тела."""
    t = tensor[0].T                       # (N, 4 + 1 + 51)
    score = t[:, 4]
    keep = score > conf_thr
    t, score = t[keep], score[keep]
    if not len(t):
        return []
    boxes = _decode_boxes(t[:, :4])
    n_kp = (t.shape[1] - 5) // 3
    kpts = t[:, 5:].reshape(-1, n_kp, 3)
    picked = _nms(boxes, score, iou_thr)
    return [Detection(0, float(score[i]), tuple(float(v) for v in boxes[i]),
                      tuple(tuple(float(x) for x in kp) for kp in kpts[i]))
            for i in picked]


def to_original(dets, scale, pad_x, pad_y):
    """Пересчёт координат из квадрата 640 обратно в пиксели исходного кадра."""
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        box = ((x1 - pad_x) / scale, (y1 - pad_y) / scale,
               (x2 - pad_x) / scale, (y2 - pad_y) / scale)
        kps = tuple(((x - pad_x) / scale, (y - pad_y) / scale, c)
                    for x, y, c in d.keypoints)
        out.append(Detection(d.class_id, d.score, box, kps))
    return out