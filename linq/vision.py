"""Пре-/постпроцессинг. Логика resnet взята 1:1 из их примера example_torch_resnet50.py."""
import numpy as np
from PIL import Image

IMAGENET_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_OFFSET = 1  # в их примере классы ImageNet нумеруются с 1


def load_image(path) -> Image.Image:
    return Image.open(path).convert('RGB')


def _central_crop(image: Image.Image, percent: float) -> Image.Image:
    width, height = image.size
    min_dim = int(min(height, width) * percent)
    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    return image.crop((left, top, left + min_dim, top + min_dim))


def preprocess_resnet(image: Image.Image, hw: tuple = (224, 224)) -> np.ndarray:
    cropped = _central_crop(image, 0.85).resize(size=(hw[1], hw[0]))
    arr = (np.asarray(cropped) / 255.0 - IMAGENET_RGB_MEAN) / IMAGENET_RGB_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.array(np.expand_dims(arr, axis=0)).astype(np.float32)


def postprocess_resnet(scores: np.ndarray, top_k: int = 5):
    """-> [(class_id, score), ...] по убыванию; class_id с офсетом, как в их примере."""
    flat = np.asarray(scores).reshape(-1)
    order = np.argsort(flat)[::-1][:top_k]
    return [(int(i) + CLASS_OFFSET, float(flat[i])) for i in order]


PREPROCESSORS = {'resnet': preprocess_resnet}
POSTPROCESSORS = {'resnet': postprocess_resnet}