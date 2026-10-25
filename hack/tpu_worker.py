import os, json, time, uuid, threading
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pika

from tpu_pycloud_format import (
    MESSAGE_ID_ATTR_NAME, RESPONSE_ROUTING_KEYS_ATTR_NAME,
    TARGET_URI_ATTR_NAME, PyTask, deserialize, serialize,
)

# ---- конфиг ----
LOGIN          = os.environ.get("LOGIN", "hackathon56974")
PROGRAM        = os.environ.get("PROGRAM", "yolov8m_b1_o5_h1m1.tpu")
PROGRAM_DIR    = f"/auto/cache/{LOGIN}/"
PROGRAM_PATH   = os.path.join(PROGRAM_DIR, PROGRAM)

RABBITMQ_HOST  = os.environ.get("RABBITMQ_HOST", "10.0.114.30")
AMQP_PORT      = 5672
EXCHANGE       = "tpu.direct"
INPUT_QUEUE    = "input"
CREDENTIALS    = ("guest", "guest")

INPUT_TENSOR_NAME  = "images"
OUTPUT_TENSOR_NAME = "output0"
TIMEOUT_SECONDS    = 60

# как часто брать кадры: 1 = каждый, 5 = каждый 5-й (быстрее для прототипа)
FRAME_STEP = int(os.environ.get("FRAME_STEP", "2"))
# минимум confidence для события
CONF_THR   = float(os.environ.get("CONF_THR", "0.35"))
IOU_THR    = 0.45
# целевой размер (должен совпадать с обучением YOLO)
IMG_SIZE   = 640

EVENTS_DIR = Path(__file__).parent / "events"
EVENTS_DIR.mkdir(exist_ok=True)

# ---- препроцессинг (как в примере, letterbox) ----
def preprocess(bgr_frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ratio = min(IMG_SIZE / h, IMG_SIZE / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    img = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (IMG_SIZE - new_w) / 2
    dh = (IMG_SIZE - new_h) / 2
    img = cv2.copyMakeBorder(
        img,
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

# ---- постпроцессинг YOLOv8 (как в примере) ----
def postprocess_yolov8(raw_output, conf_thr=CONF_THR, iou_thr=IOU_THR):
    out = np.asarray(raw_output)
    if out.shape != (1, 7, 8400):
        raise ValueError(f"Unexpected TPU output shape {out.shape}")
    preds = out[0].T.astype(np.float32)
    scores = preds[:, 4:]
    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)
    keep = confs > conf_thr
    boxes = preds[keep, :4].copy()
    class_ids = class_ids[keep]; confs = confs[keep]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    dets = []
    for cls in np.unique(class_ids):
        pos = np.flatnonzero(class_ids == cls)
        idx = cv2.dnn.NMSBoxes(
            boxes[pos].tolist(), confs[pos].tolist(), conf_thr, iou_thr
        )
        for li in np.asarray(idx, dtype=np.int64).reshape(-1):
            i = pos[li]
            x, y, w, h = boxes[i]
            dets.append({
                "box": [float(x), float(y), float(x+w), float(y+h)],
                "class_id": int(class_ids[i]),
                "confidence": float(confs[i]),
            })
    return sorted(dets, key=lambda d: d["confidence"], reverse=True)

# ---- масштабирование боксов обратно к исходному кадру ----
def rescale_boxes(dets, orig_h, orig_w):
    ratio = min(IMG_SIZE / orig_h, IMG_SIZE / orig_w)
    pad_w = (IMG_SIZE - orig_w * ratio) / 2
    pad_h = (IMG_SIZE - orig_h * ratio) / 2
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1 = (x1 - pad_w) / ratio; x2 = (x2 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio; y2 = (y2 - pad_h) / ratio
        out.append({**d, "box": [x1, y1, x2, y2]})
    return out

# ---- отправка одного кадра в TPU ----
def send_frame(channel, reply_queue, frame_tensor):
    task_id = uuid.uuid4().hex
    tensors = {INPUT_TENSOR_NAME: frame_tensor}
    started = time.monotonic()
    channel.basic_publish(
        exchange=EXCHANGE, routing_key=INPUT_QUEUE,
        body=serialize(PyTask(tensors)),
        properties=pika.BasicProperties(headers={
            TARGET_URI_ATTR_NAME: f"file://{PROGRAM_PATH}?raw=False",
            RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]),
            MESSAGE_ID_ATTR_NAME: task_id,
        }),
    )
    # ждём ответ
    for method, props, body in channel.consume(reply_queue, auto_ack=True, inactivity_timeout=1.0):
        if method is None:
            if time.monotonic() - started > TIMEOUT_SECONDS:
                raise TimeoutError("TPU timeout")
            continue
        channel.cancel()
        break
    if props.headers.get(MESSAGE_ID_ATTR_NAME) != task_id:
        raise RuntimeError("message id mismatch")
    data, _, worker = deserialize(body)
    if not data:
        raise RuntimeError(f"worker error: {worker.get('error_message')}")
    return data[OUTPUT_TENSOR_NAME], time.monotonic() - started

# ---- главный цикл ----
def run(video_path: str, state: dict, lock: threading.Lock, save_fn):
    try:
        # копия .tpu в кэш (как в примере)
        os.makedirs(PROGRAM_DIR, exist_ok=True)
        src = Path(__file__).parent / PROGRAM
        if src.exists():
            import shutil
            shutil.copy(src, PROGRAM_PATH)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        with lock:
            state["metrics"]["frames_total"] = total
            save_fn()

        conn = pika.BlockingConnection(pika.ConnectionParameters(
            RABBITMQ_HOST, AMQP_PORT, credentials=pika.PlainCredentials(*CREDENTIALS)
        ))
        ch = conn.channel()

        frame_idx = 0
        latencies = []
        t0 = time.monotonic()
        event_counter = 0
        last_event_time = {}  # подавление дублей по классу (анти-спам)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % FRAME_STEP != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            tensor = preprocess(frame)

            reply_q = f"notebook_{uuid.uuid4().hex[:8]}"
            ch.queue_declare(queue=reply_q, auto_delete=True)
            ch.queue_bind(exchange=EXCHANGE, queue=reply_q, routing_key=reply_q)

            try:
                raw, latency = send_frame(ch, reply_q, tensor)
            finally:
                try: ch.queue_delete(reply_q)
                except Exception: pass

            dets = postprocess_yolov8(raw)
            dets = rescale_boxes(dets, h, w)
            latencies.append(latency)

            # сохраняем событие только если есть детекты и прошло >= 2с с прошлого такого же класса
            now = time.time()
            interesting = [d for d in dets if d["confidence"] >= CONF_THR]
            if interesting:
                # агрегируем по классам
                seen_classes = set(d["class_id"] for d in interesting)
                suppressed = True
                for cls in seen_classes:
                    if now - last_event_time.get(cls, 0) > 2.0:
                        suppressed = False
                        last_event_time[cls] = now
                if not suppressed:
                    event_counter += 1
                    # рисуем боксы
                    frame_draw = frame.copy()
                    for d in interesting:
                        x1,y1,x2,y2 = [int(v) for v in d["box"]]
                        cv2.rectangle(frame_draw, (x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(frame_draw,
                                    f"c{d['class_id']} {d['confidence']:.2f}",
                                    (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,0,255), 2)
                    fname = f"ev_{event_counter:05d}_f{frame_idx}.jpg"
                    cv2.imwrite(str(EVENTS_DIR / fname), frame_draw,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])

                    evt = {
                        "id": event_counter,
                        "frame_idx": frame_idx,
                        "time_sec": round(frame_idx / fps_src, 2),
                        "timestamp": now,
                        "classes": sorted(seen_classes),
                        "detections": interesting,
                        "frame_url": f"/event_frame/{fname}",
                        "latency_ms": round(latency*1000, 1),
                    }
                    with lock:
                        state["events"].append(evt)
                        save_fn()

            # прогресс + метрики
            done = frame_idx + 1
            elapsed = max(1e-6, time.monotonic() - t0)
            fps_proc = done / elapsed
            lat_avg = float(np.mean(latencies[-200:])) * 1000
            lat_p95 = float(np.percentile(latencies[-200:], 95)) * 1000 if latencies else 0

            with lock:
                state["metrics"]["frames_done"] = done
                state["metrics"]["fps"] = round(fps_proc, 2)
                state["metrics"]["latency_ms_avg"] = round(lat_avg, 1)
                state["metrics"]["latency_ms_p95"] = round(lat_p95, 1)
                state["progress"] = done / max(1, total)
                save_fn()

            frame_idx += 1

        cap.release()
        conn.close()

        with lock:
            state["status"] = "done"
            state["progress"] = 1.0
            state["metrics"]["finished_at"] = time.time()
            save_fn()

    except Exception as e:
        with lock:
            state["status"] = "error"
            state["metrics"]["error"] = f"{type(e).__name__}: {e}"
            state["metrics"]["finished_at"] = time.time()
            save_fn()
