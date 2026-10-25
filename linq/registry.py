"""Реестр программ TPU и имена классов.
Файлы программ — ровно как лежат в /auto/cache/<логин>/ (см. ls)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    file: str            # файл программы в /auto/cache/<логин>/
    inputs: dict         # имя тензора -> shape (tuple)
    outputs: tuple       # имена выходных тензоров
    image_hw: tuple      # (H, W) входа модели
    preprocess: str      # ключ linq.vision.PREPROCESSORS
    postprocess: str     # ключ linq.vision.POSTPROCESSORS


PROGRAMS = {
    'resnet50': ProgramSpec('resnet50', 'torch_resnet50_b1_o5_h1m.tpu',
                            {'input.1': (1, 3, 224, 224)}, ('495',),
                            (224, 224), 'resnet', 'resnet'),
    'yolo_detect': ProgramSpec('yolo_detect', 'yolov8m_b1_o5_h1m1.tpu',  # ← добавить суффикс
                               {'images': (1, 3, 640, 640)}, ('output0',),
                               (640, 640), 'yolo', 'yolo_detect'),
    'yolo_pose': ProgramSpec('yolo_pose', 'yolov8m-pose_b1_o5_h1m1.tpu',  # ← добавить суффикс
                             {'images': (1, 3, 640, 640)}, ('output0',),
                             (640, 640), 'yolo', 'yolo_pose'),
    'smoke': ProgramSpec('smoke', 'smoke_model_b1_o5_h1m1.tpu',  # ← добавить суффикс
                         {'images': (1, 3, 640, 640)}, ('output0',),
                         (640, 640), 'yolo', 'yolo_detect'),
    'harness': ProgramSpec('harness', 'fall_protection_b1_o5_h1m.tpu',  # ← добавить суффикс
                           {'images': (1, 3, 640, 640)}, ('output0',),
                           (640, 640), 'yolo', 'yolo_detect'),
}


def validate_inputs(spec: ProgramSpec, tensors: dict) -> None:
    """Проверка тензоров ДО отправки: ловим баги препроцессинга без сети."""
    for name, shape in spec.inputs.items():
        if name not in tensors:
            raise KeyError(f'нет входного тензора {name!r} для программы {spec.name!r}')
        if tuple(tensors[name].shape) != tuple(shape):
            raise ValueError(f'{spec.name}:{name} shape {tuple(tensors[name].shape)} != {tuple(shape)}')


# --- имена классов ---
# yolov8m / yolov8m-pose: COCO (вывод names подтверждён)
CLS_PERSON = 0
CLS_VEHICLES = (2, 3, 5, 7)   # car, motorcycle, bus, truck
CLS_PHONE = 67                # cell phone
# smoke_model: {0: 'cigarette', 1: 'person', 2: 'smoke'} (вывод names подтверждён)
CLS_CIGARETTE = 0
CLS_SMOKE = 2
SMOKE_CLASSES = (0, 2)         # что берём из smoke-модели в правила
# fallprotection: порядок по README Thalos (подтвердить выводом YOLO(...).names!):
# 0 harness_worn, 1 harness_missing, 2 lanyard_attached,
# 3 lanyard_not_attached, 4 person_at_height, 5 edge_zone
CLS_HARNESS_WORN = 0
CLS_HARNESS_MISSING = 1
CLS_LANYARD_ATTACHED = 2
CLS_PERSON_AT_HEIGHT = 4
HARNESS_OK_CLASSES = (0, 2)    # привязь надета ИЛИ строп пристёгнут -> защищён
# keypoints позы (COCO): плечи и бёдра
KP_SH_L, KP_SH_R, KP_HIP_L, KP_HIP_R = 5, 6, 11, 12